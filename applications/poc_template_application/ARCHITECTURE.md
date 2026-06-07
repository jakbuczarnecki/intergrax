# poc_template_application — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

## Purpose

**Canonical Tier-3 lab shell** — minimal H-APP host for new applications (Phase AA reference).

## Manifest

- `ApplicationEnvironmentProfile.lab_defaults` + `IntegrationProfile.lab_stack()`
- Roster: `EchoAgent`

## Factory

- `build_harness_host_runtime` → debug API + lab routes + optional MCP

## Deploy triad

| Piece | Location |
|-------|----------|
| Docker | `docker/` |
| Deploy | `BUILD_AND_DEPLOY.md` |

## Dependencies (pyproject.toml)

- Core `Intergrax-ai` install from repo root
- Optional: `[harness-author]` for external repos
- Tests: `[dev-ci]` or `[dev]`

## Run

```bash
uv run uvicorn poc_template_application.host.main:app --port 8092
```
