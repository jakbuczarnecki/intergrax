        # Build & deploy — Intergrax Legal API

        Tier-3 application package: ``applications/legal_application``. This document is the **operational runbook** for local development, verification, and container deployment.

        > Quick overview: [`README.md`](../README.md) · Layout canon: [`applications/USAGE.md`](../../USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../../intergrax/applications/USAGE.md)

        ---

        ## Prerequisites

        | Tool | Purpose |
        |------|---------|
        | [uv](https://docs.astral.sh/uv/) | Python deps from repo root ``pyproject.toml`` / ``uv.lock`` |
        | Repo clone | Monorepo; **build context is always repository root** |
        | Docker (optional) | Image build via ``docker`` |
        | Docker Buildx (recommended) | Per-app ``.dockerignore`` via ``--ignorefile`` |

        Tier-2 agents used by this host: **legal** (under ``agents`` on ``PYTHONPATH``).

        ---

        ## 1. Configuration

        ```bash
        cp applications/legal_application/.env.example applications/legal_application/.env
        ```

        Edit ``.env`` (gitignored). Variables use the application prefix **`LEGAL_`** — do not put app secrets only in the repository-root ``.env``.

        | Variable | Default | Role |
        |----------|---------|------|
        | ``INTERGRAX_ENV`` | ``dev`` | ``prod`` for production-like runs |
        | ``LEGAL_BACKEND_HOST`` | see ``.env.example`` | Bind address |
        | ``LEGAL_BACKEND_PORT`` | ``8000`` | HTTP port |

        Agent roster and integrations: ``manifest.py``, ``host/wiring.py``, ``host/tool_wiring.py``. Runtime platform: validating event store + ``bootstrap_nexus_platform()`` (see ``applications/USAGE.md``).

        ### LLM provider

        | Variable | Default | Role |
        |----------|---------|------|
        | ``LEGAL_LLM_PROVIDER`` | ``ollama`` | :class:`~intergrax.llm_adapters.contracts.llm_provider.LLMProvider` slug |
        | ``LEGAL_LLM_MODEL`` | (empty) | Optional model/deployment override for :class:`~intergrax.llm_adapters.registry.profile.LLMProfile` |
        | ``INTERGRAX_LLM_METRICS_ENABLED`` | ``false`` | Per-tenant/provider token/latency counters |
        | ``calls_per_minute`` (in profile options) | — | Optional LLM rate limit per provider |
        | ``circuit_breaker_threshold`` (in profile options) | — | Optional fail-fast after N errors |

        Metrics HTTP (optional): ``register_llm_metrics_routes(app)`` → ``GET /metrics/llm``. See [architecture/LLM_ADAPTERS.md](../../../docs/project/architecture/LLM_ADAPTERS.md).

        ### Tool catalog (optional)

        | Variable | Default | Role |
        |----------|---------|------|
        | ``LEGAL_ENABLE_RAG`` | ``false`` | Register ``rag.retrieve`` on ``RuntimeConfig`` |
        | ``LEGAL_ENABLE_RAG_INGEST`` | ``false`` | Register ``rag.ingest_document`` (index local files) |
        | ``LEGAL_ENABLE_WEBSEARCH`` | ``false`` | Register ``websearch.query`` |
        | ``LEGAL_USE_TOOL_DECISION`` | ``false`` | LLM tool-decision step before Nexus bridge |
        | ``LEGAL_TOOLS_MODE`` | ``off`` | ToolsAgent planner mode when tools enabled |
        | ``LEGAL_ENABLED_TOOLS`` | (empty) | Comma-separated extra catalog tool_ids |

        ``LEGAL_PRODUCT_PROFILE=research`` enables RAG, websearch, and tool-decision by default (override with env).

        Wire vectorstore / websearch backends in ``host/tool_wiring.py`` when enabling RAG/websearch in production. Set ``INTERGRAX_INTEGRATION_DOCUMENT_PARSER=docling`` (and optional ``INTERGRAX_INTEGRATION_RERANK_PROVIDER``) for ingestion/rerank governance. See [`intergrax/tools/USAGE.md`](../../../intergrax/tools/USAGE.md).

        ---

        ## 2. Local run (development)

        From **repository root**:

        ```bash
        uv run uvicorn legal_application.host.main:app --host 127.0.0.1 --port 8000
        ```

        Or use the module CLI (reads ``LEGAL_BACKEND_*`` from ``.env``):

        ```bash
        uv run python -m legal_application.host.main
        ```

        ### Smoke check

        ```bash
        curl -s http://127.0.0.1:8000/health
        ```

        ### Product API

Routes are mounted under ``/v1/legal``. See ``serving`` and application README for contract details.

        ---

        ## 3. Verify before deploy

        ```bash
        uv run pytest applications/legal_application/tests -q
        uv run pytest tests/unit/applications/ -q -k "legal" --ignore-glob="*" 2>/dev/null || true
        ```

        Gate (repo CI):

        ```bash
        uv run pytest -m gate -q
        ```

        ---

        ## 4. Container image

        Build context = **monorepo root** (``.``). Dockerfile lives under this application only as a path reference.

        ### Build scripts (recommended)

        Run from **repository root** or from ``applications/legal_application/docker`` (scripts ``cd`` to repo root):

        ```bash
        # Linux / macOS / Git Bash
        applications/legal_application/docker/build-docker.sh

        # Windows (cmd)
        applications\legal_application\docker\build-docker.bat
        ```

        Override image tag: ``IMAGE_TAG=my-registry/legal:1.0.0`` (sh) or ``build-docker.bat my-registry/legal:1.0.0`` (bat).

        ### Manual — BuildKit

        ```bash
        docker buildx build -f applications/legal_application/docker/Dockerfile \
          --ignorefile applications/legal_application/docker/.dockerignore \
          -t legal-application .
        ```

        ### Manual — classic Docker

        ```bash
        cp applications/legal_application/docker/.dockerignore .dockerignore
        docker build -f applications/legal_application/docker/Dockerfile -t legal-application .
        ```

        **Notes:**

        - First build can take several minutes (full ``uv sync`` inside the image).
        - The image adjusts ``tool.uv.environments`` from ``win32`` to ``linux`` during build (dev lockfile targets Windows).
        - Image ``HEALTHCHECK`` probes ``/health``.
        - Scripts use BuildKit when ``docker buildx`` is available; otherwise they fall back to ``docker build``.

        ---

        ## 5. Run container

        ```bash
        docker run --rm \
          --env-file applications/legal_application/.env \
          -e INTERGRAX_ENV=prod \
          -e LEGAL_BACKEND_HOST=0.0.0.0 \
          -e LEGAL_BACKEND_PORT=8000 \
          -p 8000:8000 \
          legal-application
        ```

        ### Docker Compose

        From **repository root**:

        ```bash
        docker compose -f applications/legal_application/docker/docker-compose.yml up --build
        ```

        Ensure ``applications/legal_application/.env`` exists (compose uses ``env_file: ../.env``).

        ---

        ## 6. Production checklist

        - [ ] ``INTERGRAX_ENV=prod`` and application-prefixed secrets in orchestrator / ``.env``, not committed
        - [ ] ``LEGAL_*`` reviewed against ``host/settings.py``
        - [ ] Image tagged and pushed to your registry: ``docker tag legal-application <registry>/legal-application:<version>``
        - [ ] Health check wired to ``GET /health`` (or orchestrator equivalent)
        - [ ] Agent roster in ``manifest.py`` matches agents copied in ``docker/Dockerfile`` / ``.dockerignore``

        ---

        ## 7. Troubleshooting

        | Issue | What to try |
        |-------|-------------|
        | ``unknown flag: --ignorefile`` | Use **Buildx** or copy ``docker/.dockerignore`` to repo root |
        | Import errors for agents | Confirm ``agents/<slug>`` is listed in ``docker/.dockerignore`` exceptions |
        | Slow rebuild | Use BuildKit cache; avoid copying whole repo without per-app ``.dockerignore`` |
        | Wrong agents in registry | Check ``manifest.py`` flags / ``host/wiring.py`` and ``LAB_INCLUDE_*`` (lab) |

        ---

        *Generated for Intergrax Tier-3 scaffold (profile: product).*

## Application dependency project

Canonical packaging: [docs/project/architecture/APPLICATION_DEPENDENCY_MODEL.md](../../../docs/project/architecture/APPLICATION_DEPENDENCY_MODEL.md).

```bash
uv sync --project applications/legal_application
uv run --project applications/legal_application python -m legal_application.host.main
```

The application `pyproject.toml` selects Intergrax platform extras. Docker uses the same application project (`uv sync --frozen --no-dev --project applications/legal_application`); do not pass root `--extra` flags in the Dockerfile.

## Application runtime graph (isolated images)

Canonical packaging and image isolation: [docs/project/architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md](../../../docs/project/architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md).

```bash
uv sync --project applications/legal_application
uv run python scripts/build/build_application_image.py --application legal_application --tag legal-application:local
```

Compose uses docker/runtime-context/ produced by the same builder (--context-dir ... --keep-context). Do not build with repository-root context.
