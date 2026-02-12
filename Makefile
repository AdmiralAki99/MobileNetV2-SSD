.PHONY: install dev test test-unit test-integration test-regression test-all test-cov lint format clean

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

test: test-unit # Default run unit tests only

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-regression:
	pytest -m regression

test-all:
	pytest -m "unit or integration or regression"

test-cov:
	pytest -m unit --cov=mobilenetv2ssd --cov-report=term-missing

lint:
	ruff check .

format:
	black .

clean:
	find . -name "__pycache__" -o -name "*.pyc" -delete