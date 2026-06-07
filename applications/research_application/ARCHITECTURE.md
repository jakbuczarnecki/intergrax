# research_application — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

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
