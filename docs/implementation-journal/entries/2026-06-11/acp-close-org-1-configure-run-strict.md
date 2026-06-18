---
id: IJ-2026-06-11-010
date: 2026-06-11
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-ORG-1
status: completed
commit: pending
adr: none — enforcement of existing §30.6 / §39.4 STRICT posture contract
---

# ACP-CLOSE ORG-1 — STRICT configure_run widen deny

## Operator request

Execute next ACP-CLOSE sprint: per-agent STRICT deny when `configure_run` or request overrides attempt to widen tools or org posture.

## Summary

Added `configure_run_strict` module with execution-mode resolution, overlay validation, and `environment_overrides` clamping. `merge_environment` enforces ceiling from contract+binding before applying request overrides or `configure_run` overlay. `acp_run` fails fast with `POLICY_DENIED` when violations occur. Scoreboard policy and security dimensions no longer list STRICT widen blockers.

## Project impact

Agents in STRICT hosts cannot bypass organizational or contract tool ceilings via `configure_run` or per-run overrides. GAP org-rule bypass via configure_run closed per §39.4.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §30.6 · §39.4 |
| Plan | `ACP-CLOSE-ORG-1` |
| ADR | none |

## Changed artifacts

- `intergrax/agents/configure_run_strict.py` — STRICT widen deny (new)
- `intergrax/agents/run_environment.py` — clamp overrides + sanitize overlay
- `intergrax/agents/authoring/acp_run.py` — fail-fast on violation
- `intergrax/agents/readiness/scoreboard.py` — policy/security blockers cleared
- `tests/unit/agents/test_configure_run_strict.py` — unit + acp_run test (new)

## Verification

```bash
uv run pytest tests/unit/agents/test_configure_run_strict.py tests/unit/agents/test_run_environment.py tests/unit/agents/test_org_policy_merge_acp_org.py -q
```

Result: pass.

## Risks and follow-ups

- BALANCED mode still allows configure_run widen when host policy permits (by design §30.6).
- ACP-CLOSE-ORG-2 product-host UC-11 compliance golden tests remain open.
