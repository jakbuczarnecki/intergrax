---
id: IJ-2026-06-11-019
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-CON-1
  - APP-CON-2
  - H-APP-CON-DOC.1
status: completed
commit: pending
adr: none — wiring follows existing middleware pattern; ADR-APP-001 optional for hook semantics
---

# APP-CON-1 ApplicationHost Nexus wiring + environment state

## Operator request

Close APP-CON-1 hook mounting gap and enrich Tier-3 architecture with hook lifecycle matrix, ApplicationEnvironmentState, budget reactions, test matrix, composition separation, and production acceptance criteria before further implementation.

## Summary

Implemented `apply_application_host_wiring` mounted from `build_harness_host_runtime(application_host=...)`, wired `HarnessApplication.hooks()` to runtime, added `ApplicationEnvironmentState` contract, gate tests for host middleware mount and agent-selection block. Expanded architecture §25.3.1, §32.1, §41–§46; updated plan and hub.

## Project impact

Tier-3 environments can register real dynamic `ApplicationHost` reactions on Nexus boundaries — not documentation-only. Typed host state key `app_env_state.v1` gives hook authors a structured state model. Production readiness criteria are explicit (~7.5/10 until ACP-TOK-2/3).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §32, §41–§46 |
| Plan | `APP-CON-1`, `APP-CON-2` in H-APP-CON |

## Changed artifacts

- `intergrax/applications/_shared/application_host_wiring.py` — mount middleware
- `intergrax/applications/_shared/harness_host_runtime.py` — `application_host` param
- `intergrax/applications/contracts/environment_state.py` — typed state
- `intergrax/harness/app.py` — pass hooks to runtime
- `tests/unit/applications/test_application_host_wiring.py` — gate tests
- `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — canon expansion

## Verification

```bash
uv run pytest tests/unit/applications/test_application_host_wiring.py tests/unit/harness/test_harness_application_minimal.py -q
python scripts/maintenance/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- ACP-TOK-2/3: kernel budget enforcement + notify paths.
- APP-PROD-1: `check_application_host_wiring.py` CI gate.
- Optional: seed `app_env_state.v1` on intake in Nexus lifecycle.
