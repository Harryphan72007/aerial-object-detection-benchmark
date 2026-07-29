.PHONY: install test lint format typecheck clean
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

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
