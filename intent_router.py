"""Layer 7: intent routing.

Every final transcript is classified before it reaches the brain: which kind
of action it is (intent), who handles it (destination), what to act on
(query) and how risky acting on it is (risk). Keyword rules decide almost
everything in microseconds; a cheap LLM classifier is an opt-in fallback for
what the rules miss. Execution lives in server.py: low-risk routes with a
direct handler run without a brain turn, everything else reaches the brain
with the route prepended as a hint.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger("jarvis.router")

LOW, MEDIUM, HIGH = "bajo", "medio", "alto"
GENERAL = "agente"


@dataclass(frozen=True)
class Route:
    intent: str
    destination: str
    query: str      # "" when the utterance names nothing to act on
    risk: str


# (intent, destination, risk, pattern). First match wins, so specific rules
# sit above broad ones. Named groups hold the query; a pattern without any
# uses the whole utterance.
_ARTS = r"(?:(?:el|la|los|las|un|una|mi|mis)\s+)?"
_NOTES = r"(?:(?:mi\s+)?obsidian|mis notas)"
_RULES = [
    ("obsidian", "obsidian", LOW, re.compile(
        rf"\b(?:busca|encuentra|mira)\w*\s+en\s+{_NOTES}\s+{_ARTS}(?P<q>.+)"
        rf"|\b(?:busca|encuentra)\w*\s+{_ARTS}(?P<q2>.+?)\s+en\s+{_NOTES}\W*$", re.I)),
    ("memoria", "brain", LOW, re.compile(   # a write: the brain picks the title
        r"\b(?:recuerda|apunta|anota|guarda en (?:tu|la) memoria)\s+que\s+(?P<q>.+)", re.I)),
    ("memoria", "memory", LOW, re.compile(
        rf"\b(?:qu[eé] (?:recuerdas|sabes|tienes anotado) (?:de|sobre|acerca de)|"
        rf"busca en (?:tu|la|mi) memoria|te acuerdas de)\s+{_ARTS}(?P<q>.+)", re.I)),
    ("calendario", "calendar", HIGH, re.compile(
        r"\b(?:crea|agenda|programa|pon|a[ñn]ade|mueve|cancela|borra)\w*\s+(?:\w+\s+){0,3}"
        r"(?:evento|reuni[oó]n|cita|llamada|recordatorio)\b", re.I)),
    ("calendario", "calendar", LOW, re.compile(
        r"\b(?:calendario|agenda|qu[eé] tengo (?:hoy|ma[ñn]ana|el|esta)|mis (?:reuniones|eventos))\b", re.I)),
    ("gmail", "gmail", HIGH, re.compile(
        r"\b(?:env[ií]a|manda|responde|contesta|reenv[ií]a|borra|elimina)\w*\b.*"
        r"\b(?:correo|email|e-mail|mail|gmail)", re.I)),
    ("gmail", "gmail", LOW, re.compile(
        r"\b(?:correos?|emails?|e-mails?|mails?|gmail|bandeja de entrada)\b", re.I)),
    ("clima", "weather", LOW, re.compile(
        r"\b(?:clima|pron[oó]stico|temperatura|qu[eé] tiempo hace|va a llover)\b"
        r"(?:.*?\ben\s+(?P<q>[^\W\d][\w ]*?)\W*$)?", re.I)),
    ("command_center", "command_center", LOW, re.compile(
        r"\b(?:command center|centro de (?:mando|control)|dashboard)\b", re.I)),
    ("web", "web", LOW, re.compile(
        r"\b(?:busca|googlea|investiga)\w*\s+(?:en\s+)?(?:google|internet|la web)\s+(?:sobre\s+)?(?P<q>.+)", re.I)),
    ("accion_local", "local", HIGH, re.compile(
        r"\b(?:borra|elimina|apaga|reinicia|formatea|mata)\w*\b", re.I)),
    ("accion_local", "local", MEDIUM, re.compile(
        r"^(?:jarvis,?\s+)?(?:abre|cierra|lanza|ejecuta|sube|baja|silencia|pausa|reproduce)\b", re.I)),
]

_DEST = {intent: dest for intent, dest, _, _ in reversed(_RULES)} | {GENERAL: "brain"}
INTENTS = set(_DEST)


def route(text: str) -> Optional[Route]:
    """The keyword rules alone. None when none matches."""
    t = text.strip()
    for intent, dest, risk, pattern in _RULES:
        m = pattern.search(t)
        if m:
            groups = m.groupdict()
            q = next((v for v in groups.values() if v), "" if groups else t)
            return Route(intent, dest, q.strip(" .,?!¿¡"), risk)
    return None


_LLM_PROMPT = (
    "Clasifica la orden de voz de un asistente. Responde SOLO JSON: "
    '{"intent": uno de ' + json.dumps(sorted(INTENTS)) + ', "query": el objeto de la acción, '
    '"risk": "bajo"|"medio"|"alto"}. alto = envía, borra, compra o cambia algo fuera del equipo. '
    "Si no encaja en ninguno, intent=agente.")


async def classify(text: str, *, api_key: str, model: str, client: httpx.AsyncClient,
                   timeout: float = 2.0) -> Optional[Route]:
    """Cheap-LLM fallback. None on any failure: the caller just uses the brain."""
    try:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "temperature": 0, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": _LLM_PROMPT},
                               {"role": "user", "content": text}]},
            timeout=timeout)
        r.raise_for_status()
        data = json.loads(r.json()["choices"][0]["message"]["content"])
        intent = data.get("intent")
        if intent not in INTENTS:
            return None
        risk = data.get("risk") if data.get("risk") in (LOW, MEDIUM, HIGH) else MEDIUM
        return Route(intent, _DEST[intent], str(data.get("query") or "").strip(), risk)
    except Exception as e:
        log.warning(f"router LLM fallback failed: {e}")
        return None


def brain_hint(r: Route) -> str:
    """The route, as the brain is told it when it has to carry it out itself."""
    hint = f"[ruta: {r.intent} · destino: {r.destination} · query: {r.query or '-'} · riesgo: {r.risk}"
    if r.risk == HIGH:
        hint += " · confirma conmigo antes de ejecutar"
    return hint + "] "
