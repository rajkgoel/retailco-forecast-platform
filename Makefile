DATE ?=

ingest:
	uv run python -m src.ingest.ingest $(if $(DATE),--date $(DATE))

validate:
	uv run python -m src.validate.validate $(if $(DATE),--date $(DATE))

test:
	uv run pytest

.PHONY: ingest validate test
