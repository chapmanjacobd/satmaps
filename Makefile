.PHONY: help install dev-install lint typecheck test tune clean clean-cache

PYTHON := python3
PIP := $(PYTHON) -m pip

help:
	@echo "Available commands:"
	@echo "  install      : Install production dependencies"
	@echo "  dev-install  : Install development dependencies (linting, types, etc.)"
	@echo "  lint         : Run ruff for linting"
	@echo "  typecheck    : Run mypy for type checking"
	@echo "  test         : Run pytest"
	@echo "  tune         : Start the Tuner UI (port 5001)"
	@echo "  clean        : Remove temporary files and build artifacts"
	@echo "  clean-cache  : Remove cached tiles"

install:
	$(PIP) install .

dev-install:
	$(PIP) install ".[dev]"

lint:
	$(PYTHON) -m ruff check --fix .

typecheck:
	$(PYTHON) -m mypy *.py

test:
	$(PYTHON) -m pytest

tune:
	$(PYTHON) tuner_ui.py

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache .temp
	find . -name "*.vrt" -delete
	@echo "Cleaned up temporary files."

clean-cache:
	rm -rf .cache/*
	@echo "Cache cleared."
