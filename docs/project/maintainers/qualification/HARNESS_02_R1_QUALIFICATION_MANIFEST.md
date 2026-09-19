# HARNESS-02-R1 — Qualification manifest (pre-implementation)

**Parent ADR:** [`HARNESS_02_ADR1_CANONICAL_EXECUTION_TIME_AUTHORITY.md`](../architecture/HARNESS_02_ADR1_CANONICAL_EXECUTION_TIME_AUTHORITY.md)  
**Status:** Planned (gates to be implemented in HARNESS-02-R1 / R2)  
**Catalog baseline:** `tests/qualification/harness_02/catalog.py`

---

## Scope

Proofs below close HARNESS-02 blockers **B1**, **B2**, **B3** and debt **D1** per accepted ADR. Each proof must use contract-driven admission, not ad-hoc monotonic checks in consumers.

---

## Required proofs

| ID | Proof | Closes |
| --- | --- | --- |
| Q01 | Root creates durable `deadline_at_utc` exactly once per `run_id` (CAS) | B3 |
| Q02 | Same run resume preserves `deadline_at_utc` | B3 |
| Q03 | Different `attempt_id`, same `run_id` preserves deadline | B3 |
| Q04 | New `run_id` receives new authority | lifecycle |
| Q05 | Child cannot extend parent deadline | B1 |
| Q06 | Grandchild cannot extend root deadline | B1 |
| Q07 | Expired parent cannot launch child (admission EXPIRED) | B1 |
| Q08 | Cancelled parent cannot launch first tool physical attempt | B2 |
| Q09 | Expired execution cannot invoke tool (before submit) | B1 |
| Q10 | Expired execution cannot invoke LLM | B1 |
| Q11 | Provider timeout ≤ remaining execution time | provider |
| Q12 | Retry backoff does not cross global deadline | retry |
| Q13 | Corrupt/missing durable deadline on resume fails closed | B3 |
| Q14 | Parallel workers same run load identical `deadline_at_utc` | concurrency |
| Q15 | `enforce_wall_time_budget` / Nexus ticks align with projection (M7) | D1 |
| Q16 | Idempotency claim after admission guard on tool path | ordering |
| Q17 | Custom admission cannot override canonical EXPIRED/CANCELLED | hard invariant |

---

## Suggested pytest homes

| Area | Path |
| --- | --- |
| Durable authority | `tests/unit/runtime/execution/deadline_authority/` (new) |
| Admission guard | `tests/unit/runtime/execution/test_protected_work_admission.py` (new) |
| Harness catalog | extend `tests/qualification/harness_02/` rows to **CANONICAL** after R1 |
| Redelivery | extend `tests/unit/runtime/execution/budget/test_ue_9ar1_preserve_run_budget_across_redelivery.py` or sibling |

---

## Gate policy

- HARNESS-02 qualification status moves from **BLOCKED** only when Q01–Q14 pass in CI.
- Q15–Q16 may land in HARNESS-02-R2 if split for risk; must pass before HARNESS-02 closure.
