# lab_application — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)

## Purpose

Universal **Harness lab** — multi-agent roster, debug API, interactions, scheduler, strict harness profile.

## Factory

- `wire_lab_integrations` for SQLite trace/events/checkpoints
- `build_harness_host_runtime` (Phase AA-LABAPP.2) for Nexus assembly
- Retains `integration_wiring.py` / `tool_wiring.py` (lab superset)

## Manifest

- Dynamic roster from `LabApplicationSettings` flags

## Deploy triad

- `docker/`, `BUILD_AND_DEPLOY.md` — see gate `test_application_deploy_triad`

## Manifest environment

- `build_lab_environment_profile(settings)` embedded in `manifest.environment`
- **Adaptive (L4-O):** `AdaptiveProfile(enabled=True, mode=observe)` by default — `LAB_ADAPTIVE_OBSERVE=false` to disable signal collection

## Dependencies

- Full monorepo `uv sync` (torch, integrations catalog)
- `INTERGRAX_HARNESS_API_KEY` when strict/stage/prod
- Optional `[dev-ci]` for gate tests under `tests/`

## Application dependency project

Canonical packaging: [docs/project/architecture/APPLICATION_DEPENDENCY_MODEL.md](../../../architecture/APPLICATION_DEPENDENCY_MODEL.md).

```bash
uv sync --project applications/lab_application
uv run --project applications/lab_application python -m lab_application.host.main
```
