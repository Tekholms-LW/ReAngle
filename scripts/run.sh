#!/usr/bin/env bash
# Usage: RESHOOT_BACKEND=mock|comfy|local PORT=8000 scripts/run.sh
set -euo pipefail; cd "$(dirname "$0")/.."
(cd frontend && npm run build)
exec .venv/bin/uvicorn reshoot.main:app --app-dir backend --host 127.0.0.1 --port "${PORT:-8000}"
