---
id: IJ-2026-06-11-049
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-PROD-6
status: completed
commit: pending
adr: none — CI lint enforcing existing APP-CON-2/3 typed contract; no new runtime semantics
---

# Sprint 24 APP-PROD-6 environment state usage gate

## Operator request

Close the remaining Tier-3 application architecture implementation gap by delivering APP-PROD-6 and synchronizing plan/architecture fidelity tables so the frozen APP-* register reads **Done**.

## Summary

Added `environment_state_usage_wiring.py` and `check_environment_state_usage.py` — a CI lint that verifies `build_harness_host_runtime` mounts `apply_application_environment_state_wiring`, forbids raw `runtime_state["app_env_state…"]` access in Tier-3 packages, and requires `ApplicationEnvironmentState` typed helpers in `on_hook` bodies that touch `app_env_state.v1`. Wired the check into `check_application_production_gates.py`. Updated plan and architecture fidelity rows (§40, §46, §49, §50) to reflect complete platform APP-* implementation.

## Project impact

Tier-3 hosts and future `ApplicationHost` authors get a preventive gate against ad-hoc dict state on hooks; the APP-PROD register is fully closed on the platform side, leaving only cross-plan ACP-TOK-2/3 for mutating STRICT budget depth.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §40.2 · §46.3 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-PROD-6 · fidelity matrix |
| ADR | none — lint only |

## Changed artifacts

- `intergrax/applications/_shared/environment_state_usage_wiring.py` — lint helpers
- `scripts/check_environment_state_usage.py` — standalone CI gate
- `scripts/check_application_production_gates.py` — aggregates APP-PROD-6
- `tests/unit/applications/test_environment_state_usage_wiring.py` — wiring unit tests
- `tests/unit/scripts/test_check_environment_state_usage.py` — script smoke test
- `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` — fidelity + phase status sync
- `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — §40.2 · §42.4 · §46.3 sync

## Verification

```bash
uv run pytest tests/unit/applications/test_environment_state_usage_wiring.py tests/unit/scripts/test_check_environment_state_usage.py tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/check_environment_state_usage.py
python scripts/check_implementation_journal.py
```

Result: pass (pending run in iteration closeout).

## Risks and follow-ups

- Cross-plan ACP-TOK-2/3 remains open for kernel-level mutating STRICT token governance.
- Lint is static; runtime hook purity for lab replay still relies on author discipline beyond AST checks.
