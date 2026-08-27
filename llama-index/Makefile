GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

lint:	## Run linters: ruff and codespell.
	ruff check llama_index_agensgraph tests
	codespell llama_index_agensgraph tests README.md

test:	## Run tests via pytest.
	pytest tests
