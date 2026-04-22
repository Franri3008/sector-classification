.PHONY: install install-dev run test lint format cache-info

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	python -m src.cli run-all

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

cache-info:
	python -m src.cli cache-info
