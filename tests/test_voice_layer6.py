"""Layer 6 voice: OpenAI STT, ElevenLabs TTS, Spanish corrections, fast route."""
import io
import wave

import httpx
import pytest

import preflight
import stt
import tts
from tests.test_voice_ws import client, _drain_until  # noqa: F401  (fixture)


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0" * 3200)
    return buf.getvalue()


def test_leads_are_corrected_only_in_a_sales_sentence(client):
    _, server = client
    fix = server.apply_speech_corrections
    assert fix("cuántos leeds nuevos tengo") == "cuántos leads nuevos tengo"
    assert fix("dame los ELITs del pipeline") == "dame los leads del pipeline"
    assert fix("las elites del deporte") == "las elites del deporte"
    assert fix("abre el guar rum") == "abre el War Room"
    assert fix("open cloud code") == "open Claude Code"


def test_fast_route_answers_without_a_brain_turn(client):
    c, server = client
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_json({"type": "transcript", "text": "¿Qué hora es?", "isFinal": True})
        seen = _drain_until(ws, lambda m: m["type"] == "audio")
        assert seen[-1]["text"].startswith("Son las ")
        assert server.brain_instance.turns == []


def test_a_longer_question_still_goes_to_the_brain(client):
    c, server = client
    assert server.fast_reply("qué hora es la reunión con los leads") is None


def test_config_announces_openai_stt_when_keyed(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "STT_ENGINE", "openai")
    with c.websocket_connect("/ws/voice") as ws:
        assert ws.receive_json()["stt"] == "openai"


def test_stt_endpoint(client, monkeypatch):
    c, server = client
    assert c.post("/api/stt", content=_wav()).status_code == 409        # no key
    monkeypatch.setattr(server, "OPENAI_API_KEY", "sk-test")
    assert c.post("/api/stt", content=b"not a wav").status_code == 400
    seen = {}

    async def fake(wav, **kw):
        seen.update(kw, wav=wav)
        return "hola"
    monkeypatch.setattr(stt, "transcribe", fake)
    r = c.post("/api/stt", content=_wav())
    assert r.json() == {"text": "hola"} and seen["api_key"] == "sk-test" and seen["wav"] == _wav()

    async def failing(wav, **kw):
        return None
    monkeypatch.setattr(stt, "transcribe", failing)
    assert c.post("/api/stt", content=_wav()).status_code == 502


def test_stt_endpoint_refuses_a_foreign_origin(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "OPENAI_API_KEY", "sk-test")
    r = c.post("/api/stt", content=_wav(), headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_transcribe_sends_the_vocabulary_prompt():
    captured = {}

    def handler(req: httpx.Request):
        captured["auth"] = req.headers["authorization"]
        captured["body"] = req.read()
        return httpx.Response(200, json={"text": " cuántos leads "})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        text = await stt.transcribe(_wav(), api_key="k", model="gpt-4o-transcribe", client=c)
    assert text == "cuántos leads"
    assert captured["auth"] == "Bearer k"
    assert "War Room".encode() in captured["body"]


@pytest.mark.asyncio
async def test_elevenlabs_request_shape():
    captured = {}

    def handler(req: httpx.Request):
        captured["url"] = str(req.url)
        captured["key"] = req.headers["xi-api-key"]
        captured["json"] = req.read()
        return httpx.Response(200, content=b"MP3")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        r = await tts.synthesize_chunk_elevenlabs("Hola.", api_key="el", voice_id="v1",
                                                  model="eleven_flash_v2_5", client=c)
    assert r.audio == b"MP3"
    assert captured["url"].startswith("https://api.elevenlabs.io/v1/text-to-speech/v1?output_format=mp3")
    assert captured["key"] == "el" and b"eleven_flash_v2_5" in captured["json"]


def test_preflight_accepts_elevenlabs_instead_of_fish(monkeypatch):
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "v1")
    assert preflight._check_fish_api_key_sync().status == preflight.STATUS_OK
