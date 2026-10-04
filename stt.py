"""
stt.py — speech-to-text for one recorded utterance (a WAV from the browser).

Used instead of the browser's Web Speech API when JARVIS_STT is "local"
(faster-whisper on this machine, free) or "openai". The expected vocabulary
goes in as the prompt, which biases the model towards these spellings;
whatever it still gets wrong is fixed after the fact by
server.apply_speech_corrections.
"""
from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import Optional

import httpx

log = logging.getLogger("jarvis.stt")

OPENAI_STT_URL = "https://api.openai.com/v1/audio/transcriptions"
MAX_WAV_BYTES = 10 * 1024 * 1024

VOCAB = ["JARVIS", "Claude Code", "Obsidian", "leads", "prospectos", "War Room", "Cockpit", "GIM"]
PROMPT = "Conversación en español con términos en inglés: " + ", ".join(VOCAB) + "."


def is_wav(data: bytes) -> bool:
    return len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


async def transcribe(wav: bytes, *, api_key: str, model: str,
                     client: Optional[httpx.AsyncClient] = None, timeout: float = 30.0) -> Optional[str]:
    """The text of one utterance, or None if the request failed."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        r = await client.post(
            OPENAI_STT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"model": model, "prompt": PROMPT},
            timeout=timeout,
        )
        if r.status_code != 200:
            log.error(f"STT {r.status_code}: {r.text[:200]}")
            return None
        return str(r.json().get("text", "")).strip()
    except (httpx.HTTPError, OSError, ValueError) as e:
        log.error(f"STT error: {e}")
        return None
    finally:
        if own:
            await client.aclose()


_whisper = None
_whisper_lock = threading.Lock()


def _local_model(name: str):
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            # Optional dependency, only for JARVIS_STT=local: pip install faster-whisper
            from faster_whisper import WhisperModel
            _whisper = WhisperModel(name, device="auto", compute_type="int8")
            log.info(f"local STT: loaded whisper model {name!r}")
    return _whisper


async def transcribe_local(wav: bytes, *, model: str) -> Optional[str]:
    """The text of one utterance via faster-whisper on this machine, or None."""
    def run() -> str:
        segments, _ = _local_model(model).transcribe(io.BytesIO(wav), initial_prompt=PROMPT, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
    try:
        return await asyncio.to_thread(run)
    except Exception as e:      # missing package, bad model name, undecodable audio
        log.error(f"local STT error: {e}")
        return None
