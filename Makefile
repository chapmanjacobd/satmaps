.PHONY: help install dev-install lint format typecheck test serve generate clean clean-cache

PYTHON := python3
PIP := $(PYTHON) -m pip

help:
	@echo "Available commands:"
	@echo "  install      : Install production dependencies"
	@echo "  dev-install  : Install development dependencies (linting, types, etc.)"
	@echo "  lint         : Run ruff for linting"
	@echo "  typecheck    : Run mypy for type checking"
	@echo "  test         : Run pytest"
	@echo "  serve        : Start the local server for the viewer (port 8000)"
	@echo "  generate     : Run the batch combination generator"
	@echo "  clean        : Remove temporary files and VRTs"
	@echo "  clean-cache  : Remove cached tiles"

install:
	$(PIP) install .

dev-install:
	$(PIP) install ".[dev]"

lint:
	$(PYTHON) -m ruff check --fix .

typecheck:
	$(PYTHON) -m mypy satmaps.py generate_combinations.py

test:
	$(PYTHON) -m pytest

serve:
	$(PYTHON) serve.py

generate:
	$(PYTHON) generate_combinations.py

clean:
	rm -f *.vrt temp* tile* chunk*
	rm -rf __pycache__ .ruff_cache .mypy_cache combinations_output
	@echo "Cleaned up temporary files."
