.PHONY: help install dev-install lint typecheck test tune demo hawaii clean clean-cache

PYTHON := python3
PIP := $(PYTHON) -m pip

DEMO_CACHE := /mnt/d5/archive/datasets/satmaps/.cache
DEMO_BBOX_HAWAII := -158.4172265727475519,20.7947063146676037,-156.1288551802102802,21.8768578466807000
DEMO_BBOX_GREENLAND := -37.482423,80.717238,-37.13086,84.8099
DEMO_BBOX_EASTSOUND := -123.203075,48.633139,-122.785320,48.857579

help:
	@echo "Available commands:"
	@echo "  install      : Install production dependencies"
	@echo "  dev-install  : Install development dependencies (linting, types, etc.)"
	@echo "  lint         : Run ruff for linting"
	@echo "  typecheck    : Run mypy for type checking"
	@echo "  test         : Run pytest"
	@echo "  tune         : Start the Tuner UI (port 5001)"
	@echo "  demo         : Regenerate the demo land/ocean PMTiles under demo/"
	@echo "  hawaii       : Build the Hawaii ocean background and PMTiles"
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

demo:
	$(PYTHON) satmaps.py --yes --bbox $(DEMO_BBOX_HAWAII) --cache $(DEMO_CACHE) -o demo/hawaii.pmtiles
	$(PYTHON) satmaps.py --yes --bbox $(DEMO_BBOX_GREENLAND) --cache $(DEMO_CACHE) -o demo/greenland.pmtiles
	$(PYTHON) satmaps.py --yes --bbox $(DEMO_BBOX_EASTSOUND) --cache $(DEMO_CACHE) -o demo/eastsound.pmtiles

hawaii: clean
	$(PYTHON) ocean.py --grade --bbox -158.4172265727475519,20.7947063146676037,-156.1288551802102802,21.8768578466807000 hawaii.ocean.tif
	$(PYTHON) satmaps.py --grade --max-zoom 14 --bbox -158.4172265727475519,20.7947063146676037,-156.1288551802102802,21.8768578466807000 --output hawaii.pmtiles

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache .temp
	find . -name "*.vrt" -delete
	@echo "Cleaned up temporary files."
