cod:
	uv run Detector_parqueaderos.py

.PHONY: test
test:
	uv run python -m pytest -vv -s
