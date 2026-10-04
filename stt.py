"""
stt.py — OpenAI speech-to-text for one recorded utterance (a WAV from the browser).

Used instead of the browser's Web Speech API when OPENAI_API_KEY is set. The
expected vocabulary goes in as the transcription `prompt`, which biases the
model towards these spellings; whatever it still gets wrong is fixed after the
fact by server.apply_speech_corrections.
"""
from __future__ import annotations

import logging
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
