GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# Server legs for the test matrix, and the AgensGraph image each test image builds on.
LEGS ?= 2.14.1 2.15.0 2.16.0 2.17.0 2.18.4.0 2.18.6.1
PGVECTOR ?= 0.8.7

# The compose containers, for the shape that runs the server's own pg_dump.
COMPAT_DOCKER ?= 2.14=agensgraph-test-ag2141-1,2.15=agensgraph-test-ag2150-1,2.16=agensgraph-test-ag2160-1,2.17=agensgraph-test-ag2170-1,2.18.4=agensgraph-test-ag21840-1,2.18.6=agensgraph-test-ag21861-1
# The compose legs (tests/compose.yml) as label=conninfo pairs for the compat shapes.
COMPAT_DSNS ?= 2.14=postgresql://postgres:agens@127.0.0.1:55414/postgres,2.15=postgresql://postgres:agens@127.0.0.1:55415/postgres,2.16=postgresql://postgres:agens@127.0.0.1:55416/postgres,2.17=postgresql://postgres:agens@127.0.0.1:55417/postgres,2.18.4=postgresql://agens:agens@127.0.0.1:55419/postgres,2.18.6=postgresql://agens:agens@127.0.0.1:55418/postgres
# The compose legs 0.3.1 runs on, for recording what it returns (tests/goldens).
GOLDEN_DSNS ?= 2.17=postgresql://postgres:agens@127.0.0.1:55417/postgres 2.18.4=postgresql://agens:agens@127.0.0.1:55419/postgres 2.18.6=postgresql://agens:agens@127.0.0.1:55418/postgres

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

test: unit_tests integration_tests ## Run all tests

unit_tests: ## Run unit tests
	pytest tests/unit_tests

integration_tests: ## Run integration tests
	pytest tests/integration_tests

test-images: ## Build the agensgraph-test image for every leg (local only)
	@for leg in $(LEGS); do \
		base="skaiworldwide/agensgraph:v$$leg-bookworm"; \
		echo "building agensgraph-test:$$leg-pgvector$(PGVECTOR) from $$base"; \
		docker build --build-arg BASE=$$base --build-arg AGVERSION=$$leg --build-arg PGVECTOR=$(PGVECTOR) \
			-t agensgraph-test:$$leg-pgvector$(PGVECTOR) tests/images || exit 1; \
	done

test-compat: ## Run the phase-0 shape matrix against the compose legs
	AGENSGRAPH_COMPAT_DSNS="$(COMPAT_DSNS)" AGENSGRAPH_COMPAT_DOCKER="$(COMPAT_DOCKER)" \
		REQUIRE_SERVER=1 pytest tests/compat

goldens: ## Record what the released 0.3.1 returns, on the compose legs it runs on
	uv run --no-project --python 3.13 --with langchain-agensgraph==0.3.1 \
		--with agensgraph-python==2.0.0 python -P tests/goldens/record.py $(GOLDEN_DSNS)
