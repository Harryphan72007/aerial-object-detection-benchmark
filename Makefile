.PHONY: install test lint format typecheck notebooks links secrets verify clean
install:
	python -m pip install -e . -r requirements.txt

test:
	pytest -q

lint:
	ruff check src scripts tests

format:
	ruff format src scripts tests

typecheck:
	mypy src

notebooks:
	python scripts/validate_notebooks.py

links:
	python scripts/validate_doc_links.py

secrets:
	python scripts/scan_repository_secrets.py

verify: lint test notebooks links secrets

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
