## CONFIG
export PYTHONPATH := $(PYTHONPATH):$(pwd)
SHELL := /bin/bash -v

## COMMANDS
install:
	@poetry install

shell:
	@poetry shell

ready:
	@poetry run python src/__ready__.py

run:
	@poetry run python src/main.py

watch:
	@watchman-make -p 'src/**/*.py' -r 'make run'

cleanup:
	@rm -rf **/__pycache__
	@poetry env remove python

test:
	@pytest test --cov=src
