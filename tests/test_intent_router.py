"""Layer 7: intent routing before the brain."""
import pytest

import intent_router as ir
from tests.test_voice_ws import client, _drain_until  # noqa: F401  (fixture)


@pytest.mark.parametrize("text, intent, dest, query, risk", [
    ("jarvis, busca en Obsidian la base de datos de leads", "obsidian", "obsidian", "base de datos de leads", "bajo"),
    ("busca la base de datos de leads en obsidian", "obsidian", "obsidian", "base de datos de leads", "bajo"),
    ("qué recuerdas sobre el proyecto Atlas", "memoria", "memory", "proyecto Atlas", "bajo"),
    ("recuerda que Ana prefiere llamadas por la tarde", "memoria", "brain", "Ana prefiere llamadas por la tarde", "bajo"),
    ("agenda una reunión con Pedro mañana a las 10", "calendario", "calendar", "agenda una reunión con Pedro mañana a las 10", "alto"),
    ("qué tengo hoy en el calendario", "calendario", "calendar", "qué tengo hoy en el calendario", "bajo"),
    ("envía un correo a Juan diciendo que llego tarde", "gmail", "gmail", "envía un correo a Juan diciendo que llego tarde", "alto"),
    ("tengo correos nuevos", "gmail", "gmail", "tengo correos nuevos", "bajo"),
    ("cómo está el clima", "clima", "weather", "", "bajo"),
    ("cómo está el clima en Santiago", "clima", "weather", "Santiago", "bajo"),
    ("abre el command center", "command_center", "command_center", "abre el command center", "bajo"),
    ("busca en google precio del dólar hoy", "web", "web", "precio del dólar hoy", "bajo"),
    ("abre Spotify", "accion_local", "local", "abre Spotify", "medio"),
    ("borra la carpeta de descargas", "accion_local", "local", "borra la carpeta de descargas", "alto"),
])
def test_rules(text, intent, dest, query, risk):
    assert ir.route(text) == ir.Route(intent, dest, query, risk)


def test_unmatched_goes_to_the_agent():
    assert ir.route("explícame por qué falla el build de chitauri") is None


def test_high_risk_hint_asks_for_confirmation():
    hint = ir.brain_hint(ir.route("manda un email a Laura"))
    assert "riesgo: alto" in hint and "confirma" in hint


def test_obsidian_search_runs_without_a_brain_turn(client, tmp_path, monkeypatch):
    c, server = client
    (tmp_path / "CRM").mkdir()
    (tmp_path / "CRM" / "Base de datos de leads.md").write_text("Tabla con 40 leads de inmobiliarias.\n")
    (tmp_path / "Recetas.md").write_text("Pan de masa madre.\n")
    monkeypatch.setattr(server, "OBSIDIAN_VAULT", str(tmp_path))
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_json({"type": "transcript", "text": "jarvis, busca en Obsidian la base de datos de leads",
                      "isFinal": True})
        seen = _drain_until(ws, lambda m: m["type"] == "audio")
        assert "Base de datos de leads" in seen[-1]["text"]
        assert server.brain_instance.turns == []


def test_high_risk_reaches_the_brain_with_the_route(client):
    c, server = client
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_json({"type": "transcript", "text": "envía un correo a Juan", "isFinal": True})
        _drain_until(ws, lambda m: m["type"] == "audio")
    assert "[ruta: gmail" in str(server.brain_instance.turns[0])
