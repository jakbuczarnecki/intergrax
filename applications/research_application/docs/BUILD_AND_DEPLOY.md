        # Build & deploy - Research Application

        Tier-3 application package: ``applications/research_application``. This document is the **operational runbook** for local development, verification, and container deployment.

        > Quick overview: [`README.md`](../README.md) · Layout canon: [`applications/USAGE.md`](../../USAGE.md) · Engine API: [`intergrax/applications/USAGE.md`](../../../intergrax/applications/USAGE.md)

        ---

        ## Prerequisites

        | Tool | Purpose |
        |------|---------|
        | [uv](https://docs.astral.sh/uv/) | Python deps from repo root ``pyproject.toml`` / ``uv.lock`` |
        | Repo clone | Monorepo; **build context is always repository root** |
        | Docker (optional) | Image build via ``docker`` |
        | Docker Buildx (recommended) | Per-app ``.dockerignore`` via ``--ignorefile`` |

        Tier-2 agents used by this host: **research** (under ``agents`` on ``PYTHONPATH``).

        ---

        ## 1. Configuration

        ```bash
        cp applications/research_application/.env.example applications/research_application/.env
        ```

        Edit ``.env`` (gitignored). Variables use the application prefix **`RESEARCH_`** - do not put app secrets only in the repository-root ``.env``.

        | Variable | Default | Role |
        |----------|---------|------|
        | ``INTERGRAX_ENV`` | ``dev`` | ``prod`` for production-like runs |
        | ``RESEARCH_BACKEND_HOST`` | see ``.env.example`` | Bind address |
        | ``RESEARCH_BACKEND_PORT`` | ``8010`` | HTTP port |

        Agent roster and integrations: ``manifest.py``, ``host/wiring.py``, ``host/integration_wiring.py``, ``host/tool_wiring.py``.

        ### Tool catalog (optional)

        | Variable | Default | Role |
        |----------|---------|------|
        | ``RESEARCH_ENABLE_WEBSEARCH`` | ``true`` | Register ``websearch.query`` on ``RuntimeConfig`` |
        | ``RESEARCH_ENABLE_RAG`` | ``false`` | Register ``rag.retrieve`` |
        | ``RESEARCH_ENABLE_RAG_INGEST`` | ``false`` | Register ``rag.ingest_document`` |
        | ``RESEARCH_ENABLED_TOOLS`` | (empty) | Comma-separated extra catalog tool_ids |

        ``IntegrationProfile.research_product()`` sets ``document_parser=docling``, ``search_provider=google_cse``, ``rerank_provider=jina_rerank``. Override via ``INTERGRAX_INTEGRATION_*`` env vars (see repo-root ``.env.example``).

        Wire ``websearch_executor`` (or integration ``SearchProvider``) in ``host/tool_wiring.py`` for live search. See [`intergrax/tools/USAGE.md`](../../../intergrax/tools/USAGE.md).

        ---

        ## 2. Local run (development)

        From **repository root**:

        ```bash
        uv run uvicorn research_application.host.main:app --host 127.0.0.1 --port 8010
        ```

        Or use the module CLI (reads ``RESEARCH_BACKEND_*`` from ``.env``):

        ```bash
        uv run python -m research_application.host.main
        ```

        ### Smoke check

        ```bash
        curl -s http://127.0.0.1:8010/health
        ```

        ### Product API

Routes are mounted under ``/v1/research``. See ``serving`` and application README for contract details.

        ---

        ## 3. Verify before deploy

        ```bash
        uv run pytest applications/research_application/tests -q
        uv run pytest tests/unit/applications/ -q -k "research" --ignore-glob="*" 2>/dev/null || true
        ```

        Gate (repo CI):

        ```bash
        uv run pytest -m gate -q
        ```

        ---

        ## 4. Container image

        Build context = **monorepo root** (``.``). Dockerfile lives under this application only as a path reference.

        ### Build scripts (recommended)

        Run from **repository root** or from ``applications/research_application/docker`` (scripts ``cd`` to repo root):

        ```bash
        # Linux / macOS / Git Bash
        applications/research_application/docker/build-docker.sh

        # Windows (cmd)
        applications\research_application\docker\build-docker.bat
        ```

        Override image tag: ``IMAGE_TAG=my-registry/research:1.0.0`` (sh) or ``build-docker.bat my-registry/research:1.0.0`` (bat).

        ### Manual - BuildKit

        ```bash
        docker buildx build -f applications/research_application/docker/Dockerfile \
          --ignorefile applications/research_application/docker/.dockerignore \
          -t research-application .
        ```

        ### Manual - classic Docker

        ```bash
        cp applications/research_application/docker/.dockerignore .dockerignore
        docker build -f applications/research_application/docker/Dockerfile -t research-application .
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
          --env-file applications/research_application/.env \
          -e INTERGRAX_ENV=prod \
          -e RESEARCH_BACKEND_HOST=0.0.0.0 \
          -e RESEARCH_BACKEND_PORT=8010 \
          -p 8010:8010 \
          research-application
        ```

        ### Docker Compose

        From **repository root**:

        ```bash
        docker compose -f applications/research_application/docker/docker-compose.yml up --build
        ```

        Ensure ``applications/research_application/.env`` exists (compose uses ``env_file: ../.env``).

        ---

        ## 6. Production checklist

        - [ ] ``INTERGRAX_ENV=prod`` and application-prefixed secrets in orchestrator / ``.env``, not committed
        - [ ] ``RESEARCH_*`` reviewed against ``host/settings.py``
        - [ ] Image tagged and pushed to your registry: ``docker tag research-application <registry>/research-application:<version>``
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
uv sync --project applications/research_application
uv run --project applications/research_application python -m research_application.host.main
```

The application `pyproject.toml` selects Intergrax platform extras. Docker uses the same application project (`uv sync --frozen --no-dev --project applications/research_application`); do not pass root `--extra` flags in the Dockerfile.

## Application runtime graph (isolated images)

Canonical packaging and image isolation: [docs/project/architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md](../../../docs/project/architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md).

```bash
uv sync --project applications/research_application
uv run python scripts/build/build_application_image.py --application research_application --tag research-application:local
```

Compose uses docker/runtime-context/ produced by the same builder (--context-dir ... --keep-context). Do not build with repository-root context.
