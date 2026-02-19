.PHONY: setup scrape index search qdrant clean

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

qdrant:
	docker run -p 6333:6333 -p 6334:6334 \
		-v $(PWD)/qdrant_storage:/qdrant/storage \
		qdrant/qdrant

clean:
	rm -rf $(VENV) qdrant_storage data/raw data/processed
