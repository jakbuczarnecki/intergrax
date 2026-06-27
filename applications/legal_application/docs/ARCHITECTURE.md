# legal_application — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)

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

Port UAEP steps from `agents/legal/docs/SPEC_FROM_LEGACY.md` — out of Phase AA platform scope.

## Runtime recovery (APP-EVOL-5)

| Scenario | Host action |
|----------|-------------|
| Host restart | `resume_scheduler` via `ReliabilityProfile.recovery_contract` |
| Task interrupted | `resume` with checkpoint + idempotency store |
| Graph node failure | `retry_node` via Nexus orchestration retries |
| Corrupt checkpoint | `replay_from_snapshot` using `environment_snapshot.v1` |

- **Checkpoint store:** SQLite task checkpoints (see `.env.example` / `BUILD_AND_DEPLOY.md`)
- **Scheduler:** `long_running_scheduler_enabled` for async and HITL paths
- **In-flight tasks on deploy:** drain via checkpoint + `resume_token`; do not abort without operator ack
