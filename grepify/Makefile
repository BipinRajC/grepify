.PHONY: setup scrape index search qdrant sidecar server clean infra up down

VENV := .venv
PY   := $(VENV)/bin/python

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .

scrape:
	$(PY) scripts/scrape.py $(ARGS)

index:
	$(PY) scripts/index.py $(ARGS)

search:
	$(PY) scripts/search.py "$(q)"

# Start infrastructure (Postgres + Qdrant)
infra:
	docker compose up -d

up: infra sidecar server

down:
	docker compose down

sidecar:
	$(PY) -m uvicorn api.sidecar:app --host 0.0.0.0 --port 8001 --reload

server:
	cd api/server && go run .

clean:
	docker compose down -v

clean:
	rm -rf $(VENV) qdrant_storage data/raw data/processed
