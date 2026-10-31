# EBH-7 — W1/W2 qualification reconciliation record

| Field | Value |
| --- | --- |
| Start HEAD | `5df074fcbbe9d33cb3766ebaab95f92b9cee9d98` |
| Scope | Qualification/tests/docs only — **no** `intergrax/` edits |
| OD-EBH7-01 | **OPEN** (unchanged) |
| B6 / OD-EBH7-08 | **OPEN** |

## W1 production delta (frozen)

| **Status:** `W1-PROD-DELTA-PENDING-OPERATOR-REVIEW` @ `5df074fcb` — `RuntimeToolInvoker._require_current_attempt_authorization` rejects `invocation_context.tenant_id` ≠ `state.tenant_id`.

**Operator @ OD-EBH7-01 (2026-10-30):** **RETAIN — pending architectural reconciliation.** Historical introduction while OD-EBH7-01 was OPEN = authorization deviation (acknowledged, not retroactive approval). Targeted arch review + minimal canonical remediation **authorized**; no automatic revert; no duplicate tenant/governance authority.

| Aspect | Assessment |
| --- | --- |
| Security effect | Fail-closed deny before handler when optional invocation context carries mismatched tenant |
| Canonical admission | **Does not** replace Gov→Exec admission; complements Exec→Tool path when context tenant is set |
| Duplicated authority | Partial overlap with marketplace intent tenant checks and governance projection — **REFACTOR** candidate to single seam after ADR |
| OD-EBH7-01 | Prod change landed while entry **OPEN** — **unauthorized for program closure**; visible in audit trail |

**Recommendation (not executed):** **REFACTOR** after operator ADR — consolidate tenant binding at sanctioned wiring resolution; interim **RETAIN** if operator accepts delta; **REVERT** if ADR rejects second guard.

## Disposition corrections

| ID | Was @ gap W1 | Corrected |
| --- | --- | --- |
| TEN-03 | EXACT (projection) | **PARTIAL** — launcher policy DENY probe only |
| TEN-04 | EXACT (invoker only) | **PARTIAL** — ToolRuntime→catalog dispatch→invoker guard probe |
| TEN-05 | EXACT (scoped effect) | **PARTIAL** — `resolve_from_profile` sqlite only |
| TEN-13 / XDM-018 | PARTIAL | **PARTIAL** — host activation **SKIPPED/blocker** |
| RPL `integration.relational_store` | false QUALIFIED | **PARTIAL** — sqlite/duckdb slug test **skipped** without `duckdb` dep |

**TEN semantic @ SSOT:** **14/26 EXACT**, **12 OPEN** (derived from registry).

## New qualification nodes

- `tests/qualification/ebh_7/tenant_platform_e2e/test_ebh7_w1w2_qualification_reconciliation.py`
- `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py` (PARTIAL RPL probe)

## Operator gate (next Agent prod lane)

See `EBH_7_OPERATOR_DECISION_REGISTER.md` — W1-PROD-DELTA + W1/W2 correction acceptance before further `intergrax/` changes.
