VENV ?= .venv
PY := $(VENV)/bin/python
PORT ?= 8000

.PHONY: setup ui test run dev clean
setup:            ## create venv, install backend + UI deps
	python3 -m venv $(VENV)
	$(PY) -m pip install -q -r backend/requirements.txt
	cd frontend && npm install
ui:               ## build the web UI into frontend/dist (served by FastAPI)
	cd frontend && npm run build
test:             ## run all tests
	cd backend && ../$(PY) -m pytest -q tests
	cd comfyui-reshoot-nodes && ../$(PY) -m pytest -q tests
run: ui           ## serve UI + API on http://127.0.0.1:$(PORT) (RESHOOT_BACKEND=mock|comfy|local)
	$(VENV)/bin/uvicorn reshoot.main:app --app-dir backend --host 127.0.0.1 --port $(PORT)
dev:              ## API with reload; run `cd frontend && npm run dev` in another shell (http://localhost:5173)
	$(VENV)/bin/uvicorn reshoot.main:app --app-dir backend --reload --port $(PORT)
clean:
	rm -rf data frontend/dist
