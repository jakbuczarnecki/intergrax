# legal_application — architecture

## Purpose

Product-style Tier-3 host for scaffold `LegalAgent` — auth, FastAPI core, MCP, deploy triad.

## Manifest

- `environment=ApplicationEnvironmentProfile.product_defaults(...)` inline
- Single agent binding: `LegalAgent`

## Factory

- `build_harness_host_runtime` + `wire_application_environment`
- `host/wiring.py` mounts `LegalAgent()` without legacy runtime bridge

## Serving

- FastAPI router under `serving/`
- Compliance policy stub route for gate smoke

## Observability

- Trace/events DB paths from product environment profile (see `.env.example`)

## Deploy triad

- `docker/Dockerfile`, `docker/docker-compose.yml`, `BUILD_AND_DEPLOY.md`
- Gate: `tests/unit/applications/test_application_deploy_triad.py`

## Dependencies

- `uv sync` with harness extras; LLM via env (`LEGAL_*` / shared harness vars)
- LangGraph not required (`langgraph-legacy` extra optional elsewhere)

## Next steps (Band 3)

Port UAEP steps from `agents/legal/SPEC_FROM_LEGACY.md` — out of Phase AA platform scope.
