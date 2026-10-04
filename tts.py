"""
tts.py — Fish Audio (or ElevenLabs) synthesis, one request per sentence chunk.

The response is streamed so time-to-first-byte can be measured; the chunk is
returned whole because the browser decodes one complete MP3 per chunk.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger("jarvis.tts")

FISH_TTS_URL = "https://api.fish.audio/v1/tts"
# mp3 at 44.1 kHz / 128 kbps — the same kind of chunk the browser already decodes for Fish.
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"


@dataclass
class SynthResult:
    audio: bytes
    first_byte_sec: float
    total_sec: float


async def synthesize_chunk(text: str, *, api_key: str, voice_id: str,
                           client: Optional[httpx.AsyncClient] = None,
                           latency: str = "balanced", timeout: float = 15.0) -> Optional[SynthResult]:
    return await _synthesize(
        text, api_key, FISH_TTS_URL,
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {"text": text, "reference_id": voice_id, "format": "mp3", "mp3_bitrate": 128, "latency": latency},
        client, timeout)


async def synthesize_chunk_elevenlabs(text: str, *, api_key: str, voice_id: str, model: str,
                                      client: Optional[httpx.AsyncClient] = None,
                                      timeout: float = 15.0) -> Optional[SynthResult]:
    """Same contract as synthesize_chunk, voiced by ElevenLabs."""
    return await _synthesize(
        text, api_key, ELEVENLABS_TTS_URL.format(voice_id=voice_id),
        {"xi-api-key": api_key, "Content-Type": "application/json"},
        {"text": text, "model_id": model},
        client, timeout)


async def _synthesize(text: str, api_key: str, url: str, headers: dict, body: dict,
                      client: Optional[httpx.AsyncClient], timeout: float) -> Optional[SynthResult]:
    text = (text or "").strip()
    if not text or not api_key:
        return None
    own = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    t0 = time.monotonic()
    first: Optional[float] = None
    buf = bytearray()
    try:
        async with client.stream(
            "POST", url, headers=headers, json={**body, "text": text}, timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                log.error(f"TTS {resp.status_code} for {text[:40]!r}")
                return None
            async for part in resp.aiter_bytes():
                if first is None:
                    first = time.monotonic() - t0
                buf.extend(part)
    except (httpx.HTTPError, OSError) as e:
        log.error(f"TTS error: {e}")
        return None
    finally:
        if own:
            try:
                await client.aclose()
            except Exception as e:      # never turn a clean None into an exception
                log.debug(f"TTS client close failed: {e}")
    if not buf:
        return None
    return SynthResult(bytes(buf), first if first is not None else 0.0, time.monotonic() - t0)
