"""Layer 6 voice: local whisper / OpenAI STT, Piper / ElevenLabs TTS, Spanish corrections, fast route."""
import io
import sys
import types
import wave

import httpx
import pytest

import preflight
import stt
import tts
from tests.test_voice_ws import client, _drain_until  # noqa: F401  (fixture)
import server as _server

# The `client` fixture swaps _synth_for_speech for a fake; keep the real one.
# reload() re-runs the module in the same namespace, so it still sees the
# monkeypatched settings.
REAL_SYNTH = _server._synth_for_speech


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
    assert c.post("/api/stt", content=_wav()).status_code == 409        # browser STT: endpoint off
    monkeypatch.setattr(server, "STT_ENGINE", "openai")
    assert c.post("/api/stt", content=_wav()).status_code == 409        # openai without a key
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


def test_stt_endpoint_uses_local_whisper_when_asked(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "STT_ENGINE", "local")

    async def fake_local(wav, **kw):
        assert kw["model"] == server.WHISPER_MODEL
        return "cuántos leeds nuevos"
    monkeypatch.setattr(stt, "transcribe_local", fake_local)
    assert c.post("/api/stt", content=_wav()).json() == {"text": "cuántos leeds nuevos"}


def test_stt_endpoint_refuses_a_foreign_origin(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "STT_ENGINE", "local")
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


@pytest.mark.asyncio
async def test_transcribe_local_passes_the_vocabulary_to_whisper(monkeypatch):
    calls = {}

    class FakeModel:
        def __init__(self, name, **kw):
            calls["name"] = name

        def transcribe(self, audio, **kw):
            calls.update(kw, audio=audio.read())
            return iter([types.SimpleNamespace(text=" cuántos "), types.SimpleNamespace(text="leads ")]), None
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    monkeypatch.setattr(stt, "_whisper", None)
    assert await stt.transcribe_local(_wav(), model="small") == "cuántos leads"
    assert calls["name"] == "small" and calls["audio"] == _wav()
    assert "War Room" in calls["initial_prompt"] and calls["vad_filter"] is True


@pytest.mark.asyncio
async def test_transcribe_local_without_the_package_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)     # import raises ImportError
    monkeypatch.setattr(stt, "_whisper", None)
    assert await stt.transcribe_local(_wav(), model="small") is None


@pytest.mark.asyncio
async def test_piper_returns_a_wav_with_a_readable_length(monkeypatch):
    import speech

    class FakeVoice:
        @staticmethod
        def load(path):
            assert path == "/voices/es.onnx"
            return FakeVoice()

        def synthesize_wav(self, text, w):
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
            w.writeframes(b"\0" * 44100)                            # one second
    monkeypatch.setitem(sys.modules, "piper", types.SimpleNamespace(PiperVoice=FakeVoice))
    monkeypatch.setattr(tts, "_piper_voices", {})
    r = await tts.synthesize_chunk_piper("Hola.", model_path="/voices/es.onnx")
    assert r.audio[:4] == b"RIFF"
    assert speech.wav_seconds(r.audio) == pytest.approx(1.0)
    assert speech.ack_floor_seconds(r.audio) > 0


@pytest.mark.asyncio
async def test_piper_wins_over_the_cloud_voices(client, monkeypatch):
    _, server = client
    monkeypatch.setattr(server, "PIPER_VOICE", "/voices/es.onnx")
    monkeypatch.setattr(server, "ELEVENLABS_API_KEY", "el")
    monkeypatch.setattr(server, "ELEVENLABS_VOICE_ID", "v1")

    async def fake_piper(text, **kw):
        return tts.SynthResult(b"WAV:" + text.encode(), 0.0, 0.0)

    async def never(*a, **kw):
        raise AssertionError("a cloud voice was called while PIPER_VOICE is set")
    monkeypatch.setattr(tts, "synthesize_chunk_piper", fake_piper)
    monkeypatch.setattr(tts, "synthesize_chunk_elevenlabs", never)
    monkeypatch.setattr(tts, "synthesize_chunk", never)
    assert await REAL_SYNTH("Hola") == b"WAV:Hola"


def test_preflight_accepts_piper_instead_of_fish(monkeypatch):
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.setenv("PIPER_VOICE", "/voices/es.onnx")
    assert preflight._check_fish_api_key_sync().status == preflight.STATUS_OK
