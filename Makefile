cod:
	uv run streamlit run app.py
#uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true

.PHONY: test
test:
	uv run python -m pytest -vv -s