DATE ?=

ingest:
	uv run python src/ingest/ingest.py $(if $(DATE),--date $(DATE))

test:
	uv run pytest

.PHONY: ingest test
