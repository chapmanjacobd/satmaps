.PHONY: help install dev-install lint typecheck test tune hawaii vrt clean clean-cache

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
	@echo "  hawaii       : Build the Hawaii ocean background and PMTiles"
	@echo "  vrt          : Build the Hawaii PMTiles VRT"
	@echo "  clean        : Remove temporary files and build artifacts"
	@echo "  clean-cache  : Remove cached tiles"

all: test lint typecheck

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

hawaii: clean
	$(PYTHON) ocean.py --grade --bbox -158.4172265727475519,20.7947063146676037,-156.1288551802102802,21.8768578466807000
	$(PYTHON) satmaps.py --grade --bbox -158.4172265727475519,20.7947063146676037,-156.1288551802102802,21.8768578466807000 --output hawaii.pmtiles

vrt:
	$(PYTHON) satmaps.py --grade --bbox -158.4172265727475519,20.7947063146676037,-156.1288551802102802,21.8768578466807000 --output hawaii.pmtiles --vrt --resume

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache .temp
	find . -name "*.vrt" -delete
	@echo "Cleaned up temporary files."

clean-cache:
	rm -rf .cache/*
	@echo "Cache cleared."
