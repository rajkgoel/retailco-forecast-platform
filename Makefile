DATE ?=

ingest:
	uv run python -m src.ingest.ingest $(if $(DATE),--date $(DATE))

test:
	uv run pytest

.PHONY: ingest test
