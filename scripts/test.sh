#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."
(cd backend && ../.venv/bin/python -m pytest -q tests)
(cd comfyui-reshoot-nodes && ../.venv/bin/python -m pytest -q tests)
