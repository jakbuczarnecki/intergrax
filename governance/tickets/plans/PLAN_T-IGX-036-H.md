# PLAN T-IGX-036-H — PDC-P5-DOC documentation reconciliations (wave A)

Parent: PDC-X master plan [`PLAN_T-IGX-036-A.md`](PLAN_T-IGX-036-A.md) §G "PDC-P5-DOC". The rows come from `PDC_DOMAIN_REQUIREMENTS_MATRIX.json` (class `DOCUMENTATION_RECONCILIATION`). This lane is documentation only: no product, script or test changes. It can run before MCP-SERVER-X P8 because it touches no product path.

## Deliverables (one change per requirement, each citing its requirement ID)

1. **PDC-D1-R009:** `docs/project/architecture/AGENT_DISTRIBUTION.md`. The AC-4 status still reads "Phase 10 READY_FOR_INDEPENDENT_AUDIT". Change it to CLOSED, citing `docs/project/maintainers/audits/AGENT_PLATFORM_AC1_AC6_FINAL_ARCHITECTURE_AUDIT.md` (audited SHA `67f57331`), matching the plan.
2. **PDC-D4-R016:** `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`. TL-FIX-D is ACCEPTED/PLANNED in the plan but implemented in code (`bind_execution_adapter`). Mark it Done with the code reference.
3. **PDC-D4-R019:** the sandbox/shadow isolation claim. Make `TIER3_APPLICATION_ENVIRONMENT.md` (architecture) and the plan consistent. The bounded claim is "not a universal production isolation guarantee". Keep the plan's APP-CON-8 "Done" scoped to lifecycle wiring.
4. **PDC-D5-R016:** `docs/project/maintainers/plans/CRITIC_VERIFICATION.md`. Add a reconciliation note: the domain is LEGACY/HISTORICAL, canonical authority is DECISION_SYSTEM, and open critic findings are legacy lineage, re-dispositioned by PDC-X (no deletion of the historical findings).
5. **PDC-D7-R011 and PDC-D7-R021:** `docs/project/architecture/BACKGROUND_TASKS.md`. Add short sections on at-least-once delivery consequences (idempotency key, duplicate handling, safe terminal redelivery as certified by BG-01) and on the horizontal-scaling boundary (DIST-SCALE-X scoped envelope; multi-node correctness depends on the provider; SCHED-01 two-scheduler proof).
6. **PDC-D7-R023:** `docs/project/maintainers/plans/BACKGROUND_TASKS.md`. Reconcile stale track statuses with what exists in code (TaskRegistry/WorkerRuntime, BG-01 and SCHED-01 closures). Point remaining gaps to PDC-P5-D7 / ADR-PDC-07.
7. **PDC-P1-F01:** register `BACKGROUND_TASKS` and `COLLABORATIVE_WORK` in `docs/project/architecture/intergrax_runtime_architecture.md` (additional canonical DOMAIN pairs table), each exactly once.
8. **Broken link (part of PDC-D6-R021):** the plan link to the missing `docs/project/architecture/IDENTITY_TRUST.md`. Repoint it to the existing EE-B3 threat model or mark it historical. The consolidated security satellite itself stays with the D6 lane (ADR-PDC-06).
9. **PDC-D6-R025:** add a reconciliation note next to the "ENTERPRISE QUALIFIED" rows of `HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md`. The note states the scoped meaning and points to the open FRZ-SEC rows. The historical certification text is **not** rewritten.

Excluded, because they belong to ADR-bound lanes: PDC-D5-R017 (D5), the PDC-D6-R021 satellite (D6) and PDC-D6-R023 (D6).

## Acceptance

- Each listed requirement has a doc change citing its ID.
- `check_docs_domain_pairs.py` stays OK.
- No historical certification text is rewritten; only notes are added.
- English only.
- No files outside `docs/`.

## Test contract

```text
uv run python scripts/docs/check_docs_domain_pairs.py
uv run python scripts/qualification/pdc_x_inventory.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product code; scripts; tests; ADR-bound documents; FRZ changes; rewriting historical certification records.
