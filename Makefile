.PHONY: help install dev-install lint typecheck serve generate clean

PYTHON := python3
PIP := $(PYTHON) -m pip

help:
	@echo "Available commands:"
	@echo "  install      : Install production dependencies"
	@echo "  dev-install  : Install development dependencies (linting, types, etc.)"
	@echo "  lint         : Run ruff for linting and formatting"
	@echo "  typecheck    : Run mypy for type checking"
	@echo "  serve        : Start the local server for the viewer (port 8000)"
	@echo "  generate     : Run the batch combination generator"
	@echo "  clean        : Remove temporary files and VRTs"

install:
	$(PIP) install .

dev-install:
	$(PIP) install ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy satmaps.py generate_combinations.py

serve:
	$(PYTHON) serve.py

generate:
	$(PYTHON) generate_combinations.py

clean:
	rm -f *.vrt temp.mbtiles temp.vrt temp_warped.vrt
	rm -rf __pycache__
	@echo "Cleaned up temporary files."
