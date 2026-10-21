# Name That LLM — development tasks.
#
#   make install    set up the environment (uv)
#   make check      lint + self-test + notebook/source consistency
#   make notebook   regenerate notebook 02 and re-inline the package
#   make run-02     execute notebook 02 and store its outputs
#
.DEFAULT_GOAL := help
SHELL := /bin/bash

NB1 := notebooks/01_eda.ipynb
NB2 := notebooks/02_style_and_structure.ipynb

.PHONY: help install lint format test check notebook bundle run-01 run-02 data clean

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## create the virtualenv and install dev dependencies
	uv sync --extra dev

lint: ## run ruff over the package and the scripts
	uv run ruff check src scripts

format: ## auto-format the package and the scripts
	uv run ruff format src scripts
	uv run ruff check --fix src scripts

test: ## run the fast invariant checks
	uv run python scripts/selftest.py

notebook: ## regenerate notebook 02 and inline the package into both notebooks
	uv run python scripts/make_notebook.py
	uv run python scripts/sync_notebook.py --notebook $(NB2)
	uv run python scripts/sync_notebook.py --notebook $(NB1)
	uv run python scripts/check_notebooks.py

bundle: ## rebuild the flat single-file bundle in dist/
	uv run python scripts/build_bundle.py

check: lint test notebook ## everything CI runs

run-01: ## execute notebook 01 and save the outputs
	uv run python scripts/run_notebook.py $(NB1) --save

run-02: ## execute notebook 02 and save the outputs
	uv run python scripts/run_notebook.py $(NB2) --save

data: ## download the competition data into data/raw
	uv run python scripts/download_data.py

clean: ## remove caches and build artefacts
	rm -rf dist .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
