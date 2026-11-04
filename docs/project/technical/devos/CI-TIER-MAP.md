# CI tier map (DevOS ↔ InterGRAX)

| DevOS tier | InterGRAX mechanism |
|------------|---------------------|
| gate-fast | pre-push / `igx_gov ticket` with profile `fast` |
| governance-fast | CI jobs `governance-fast` + advisory `governance-policy-pin`; `validate_policy_registry.py`; `tests/unit/governance/` |
| smoke / RISK | CI job `ci_smoke`; local `igx_gov ticket --profile smoke` |
| enterprise | `workflow_dispatch` enterprise profile |
| full | `workflow_dispatch` full profile |

Workflow: [`.github/workflows/unit-tests.yml`](../../../../.github/workflows/unit-tests.yml).

Default `[tool.intergrax.push_gate] profile = "fast"` in `pyproject.toml` — do not change without operator decision.

## Agent push flow

`igx_gov ticket` → `git push` → CI smoke on PR.
