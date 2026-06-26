# research_application — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)

## Purpose

Multi-agent HTTP host for **ResearchAgent** + **SummaryAgent** with Nexus loop enabled by default.

## Manifest

- Explicit `environment=` from `host/environment_profile.py`
- Agent bindings for both research agents

## Settings

- `RESEARCH_USE_NEXUS_LOOP` (default `true`) — legacy agent-engine flag removed

## Factory

- `build_harness_host_runtime` assembly
- Registry via `build_application_registry`

## Deploy triad

- `docker/`, `BUILD_AND_DEPLOY.md` — verified by deploy triad gate

## Dependencies

- Monorepo `uv sync`; LLM provider env per `BUILD_AND_DEPLOY.md`

## Tests

- `research_application_tests/` host smoke + `test_research_manifest_wiring`

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
