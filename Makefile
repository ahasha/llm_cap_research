#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
POETRY := $(shell command -v poetry 2> /dev/null)
POETRY_ENV_DIR := $(shell $(POETRY) env info -p)
POETRY_RUN := $(POETRY) run
PROJECT_NAME = llm_cap_research
PYTHON_VERSION = 3.11
SHELL := /bin/bash

#################################################################################
# COMMANDS                                                                      #
#################################################################################
.PHONY: check_poetry
check_poetry:
	@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.rg/docs/ for installation instructions"; exit 2; fi
	@echo $(POETRY)

poetry.lock: pyproject.toml check_poetry ## Once per new project, update poetry.lock to align with pyproject.toml
	$(POETRY) lock --no-update

.PHONY: environment
environment: poetry.lock check_poetry ## Install Python Dependencies
	$(POETRY) install

clean-environment: ## Delete project virtual environment
	$(POETRY) env remove $(POETRY_ENV_DIR)/bin/python

.PHONY: git
git: ## DVC is designed to run inside a git repository.  Initialize a git repository if not done already
	@git status >/dev/null 2>&1; \
	if [[ $$? -ne 0 ]]; then \
		git init; \
		git add --all; \
		git commit -m "Initial commit"; \
	else \
		echo "Git repo already initialized, skipping..."; \
	fi

.PHONY: pre-commit-hooks
pre-commit-hooks: check_poetry ## Install pre-commit hooks
	$(POETRY_RUN) pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit

.PHONY: initialize
initialize: git environment pre-commit-hooks ## Initialize project environment, dbt profiles, and pre-commit hooks


.PHONY: test
test: check_poetry ## Run tests
	$(POETRY_RUN) pytest

.PHONY: clean
clean: ## Delete all compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: lint
lint: check_poetry ## Lint using flake8 and brunette (use `make format` to do formatting)
	$(POETRY_RUN) flake8 llm_cap_research
	$(POETRY_RUN) brunette --check --config pyproject.toml llm_cap_research

.PHONY: format
format: check_poetry ## Format source code with brunette
	$(POETRY_RUN) brunette --config pyproject.toml llm_cap_research


#################################################################################
# Automated documentation generation                                            #
#################################################################################

.PHONY: docs
docs: check_poetry clean-docs  ## Build all project documentation
	$(POETRY_RUN) sphinx-apidoc -o docs/ llm_cap_research
	$(POETRY_RUN) $(MAKE) -C docs clean
	$(POETRY_RUN) $(MAKE) -C docs html



.PHONY: clean-docs
clean-docs: ## Remove auto-generated document elements
	rm -f docs/llm_cap_research.rst || \
	rm -f docs/modules.rst || true

.PHONY: start-doc-server stop-doc-server

server.pid:
	$(POETRY_RUN) python -m http.server -d docs/_build/html > server.log 2>&1 & echo $$! > $@

start-doc-server: server.pid ## Serve documentation website over http
	@echo Documentation is being served at http://localhost:8000
	@echo Run "make stop-server" to stop the server.

stop-doc-server: ## Shut down documentation server
	test -f server.pid && kil `cat server.pid` && rm server.pid || true

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
