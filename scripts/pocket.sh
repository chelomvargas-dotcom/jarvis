#!/usr/bin/env bash
# Jarvis de bolsillo: the voice page on your phone, over Tailscale.
#
# The server stays bound to 127.0.0.1. `tailscale serve` is the only thing
# that reaches it, and only devices signed in to your tailnet can reach that,
# with a real HTTPS cert (the mic needs a secure context). The phone's page
# arrives with Origin https://<machine>.<tailnet>.ts.net, so that one origin
# is added to JARVIS_ALLOWED_ORIGINS; nothing else is opened.
#
# Usage: scripts/pocket.sh [server.py args...]   (off: tailscale serve reset)
set -euo pipefail
cd "$(dirname "$0")/.."

command -v tailscale >/dev/null || { echo "Instala Tailscale (gratis): https://tailscale.com/download"; exit 1; }
host=$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')
[ -n "$host" ] || { echo "Tailscale no está conectado: corre 'tailscale up'"; exit 1; }

# The phone loads the built frontend off the API port, not Vite.
[ -f frontend/dist/index.html ] || (cd frontend && npm run build)

port=8340
if [ -f cert.pem ] && [ -f key.pem ]; then target="https+insecure://localhost:$port"; else target="http://localhost:$port"; fi
tailscale serve --bg "$target"

echo
echo "  Jarvis de bolsillo: https://$host"
echo "  Ábrelo en el teléfono (Chrome en Android, Safari en iPhone) y añádelo a la pantalla de inicio."
echo
export JARVIS_ALLOWED_ORIGINS="${JARVIS_ALLOWED_ORIGINS:+$JARVIS_ALLOWED_ORIGINS,}https://$host"
exec python3 server.py --port "$port" "$@"
