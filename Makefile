# Name That LLM — development tasks.
#
#   make install    set up the environment (uv)
#   make check      lint + self-test + notebook/source consistency
#   make notebook   regenerate 02, 03 and the five variants; re-inline the package
#   make run-02     execute notebook 02 and store its outputs
#   make run-03     execute notebook 03 and store its outputs
#   make track      show local CV vs public LB per variant
#
.DEFAULT_GOAL := help
SHELL := /bin/bash

NB1 := notebooks/01_eda.ipynb
NB2 := notebooks/02_style_and_structure.ipynb
NB3 := notebooks/03_facets.ipynb
VARIANTS := $(sort $(wildcard notebooks/04_v*.ipynb))

.PHONY: help install lint format test check notebook bundle run-01 run-02 run-03 run-variants track data clean

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## create the virtualenv and install dev dependencies
	uv sync --extra dev

lint: ## run ruff over the package and the scripts
	uv run ruff check src scripts

format: ## auto-format the package and the scripts
	uv run ruff format src scripts
	uv run ruff check --fix src scripts

test: ## run the fast invariant checks
	uv run python scripts/selftest.py

notebook: ## regenerate 02, 03, the variants, and inline the package into all
	uv run python scripts/make_notebook.py
	uv run python scripts/make_facets_notebook.py
	uv run python scripts/make_variants.py
	uv run python scripts/sync_notebook.py --all
	uv run python scripts/check_notebooks.py

bundle: ## rebuild the flat single-file bundle in dist/
	uv run python scripts/build_bundle.py

check: lint test ## everything CI runs (verification only; never rewrites notebooks)
	uv run python scripts/check_notebooks.py
	uv run python scripts/build_bundle.py

run-01: ## execute notebook 01 and save the outputs
	uv run python scripts/run_notebook.py $(NB1) --save

run-02: ## execute notebook 02 and save the outputs
	uv run python scripts/run_notebook.py $(NB2) --save

run-03: ## execute notebook 03 and save the outputs
	uv run python scripts/run_notebook.py $(NB3) --save

run-variants: ## execute all five variant notebooks and save their outputs
	@for nb in $(VARIANTS); do \
		echo "== $$nb"; \
		uv run python scripts/run_notebook.py $$nb --save || exit 1; \
	done

track: ## show the latest local CV vs public LB per variant
	uv run python scripts/track.py report

data: ## download the competition data into data/raw
	uv run python scripts/download_data.py

clean: ## remove caches and build artefacts
	rm -rf dist .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
