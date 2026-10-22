SHELL := /bin/bash
COMPOSE      := docker compose
COMPOSE_GPU  := docker compose -f docker-compose.yml -f docker-compose.gpu.yml
IMAGE        := nyaya-sahayak:local

GPU_AVAILABLE := $(shell docker info 2>/dev/null | grep -qi 'nvidia' && echo 1 || echo 0)

.PHONY: help build up up-gpu up-cpu down logs shell test test-integration eval lint format clean rebuild

help:
	@echo "NyayaSahayak — available targets:"
	@echo "  make build              Build the Docker image"
	@echo "  make up                 Start (auto-detect GPU; falls back to CPU)"
	@echo "  make up-gpu             Force GPU mode"
	@echo "  make up-cpu             Force CPU mode"
	@echo "  make down               Stop the container"
	@echo "  make logs               Tail logs"
	@echo "  make shell              Open a shell inside the container"
	@echo "  make test               Run pure-unit tests (no vault required)"
	@echo "  make test-integration   Run all tests incl. vault-dependent ones"
	@echo "  make eval               Run the 25-query retrieval evaluation"
	@echo "  make lint               Ruff check"
	@echo "  make format             Ruff format"
	@echo "  make clean              Remove container + image"
	@echo "  make rebuild            down + build + up"

build:
	$(COMPOSE) build

up:
ifeq ($(GPU_AVAILABLE),1)
	@echo ">> NVIDIA runtime detected — starting with GPU"
	$(COMPOSE_GPU) up --build
else
	@echo ">> No NVIDIA runtime — starting CPU-only"
	$(COMPOSE) up --build
endif

up-gpu:
	$(COMPOSE_GPU) up --build

up-cpu:
	$(COMPOSE) up --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

shell:
	$(COMPOSE) exec nyaya-sahayak bash

test:
	$(COMPOSE) run --rm --no-deps nyaya-sahayak \
		python -m pytest -v -m "not integration"

test-integration:
	$(COMPOSE) run --rm nyaya-sahayak \
		python -m pytest -v

eval:
	$(COMPOSE) run --rm nyaya-sahayak \
		python scripts/tests/run_evaluation.py

lint:
	$(COMPOSE) run --rm --no-deps nyaya-sahayak \
		ruff check .

format:
	$(COMPOSE) run --rm --no-deps nyaya-sahayak \
		ruff format .

clean:
	-$(COMPOSE) down -v --remove-orphans
	-docker rmi $(IMAGE)

rebuild: clean build up