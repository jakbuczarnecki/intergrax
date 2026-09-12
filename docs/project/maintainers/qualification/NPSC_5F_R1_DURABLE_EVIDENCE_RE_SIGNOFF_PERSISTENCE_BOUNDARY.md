# NPSC-5F/R1 — Durable Evidence Re-Signoff (Persistence Boundary Evolution)

**Status:** `PASS / QUALIFIED`

**Task:** NPSC-5F/R1 Durable Evidence Re-Signoff After Persistence Boundary Evolution

**Branch:** `development`

---

## Purpose

Re-confirm that R1 Durable Evidence invariants hold after introducing `EvidencePersistencePort` and `RuntimeEventPersistenceEvidenceAdapter`. Execution Engine behavior and durable semantics are unchanged; only the responsibility boundary moved.

```text
Execution producers (RuntimeEventBus)
        |
        v
EvidencePersistencePort
        |
        v
RuntimeEventPersistenceEvidenceAdapter
        |
        v
RuntimeEventPersistence  (canonical store contract)
        |
        v
Storage
```

---

## Qualified baseline

| Label | SHA |
| ----- | --- |
| Evidence persistence port introduction | `39ac19d5ba9be3d7b7498d20539a8b2defed676c` |
| Port failure isolation + bus resilience wiring | `3d298f651` … `8fb9c0fb5` |
| **R1 re-signoff baseline** (`R1_POST_R2_QUALIFIED_BASELINE_SHA`) | `df677b5b37e0dcaa1e280b7a98324ee59ea24878` |

Post re-signoff, `collect_r1_protected_production_drift(from_sha=R1_POST_R2_QUALIFIED_BASELINE_SHA)` must remain empty unless a new qualified R1 change is recorded.

---

## R1 invariants verified (no production behavior change)

| Invariant | Result |
| --------- | ------ |
| Durable storage — required execution facts persisted when gated | PASS |
| Evidence source of truth — history from runtime events → persistence, not reconstruction/recovery/state | PASS |
| Immutable evidence — idempotent accept; payload conflict blocked | PASS |
| Append delegation — `EvidencePersistencePort.append` ≡ inner `RuntimeEventPersistence.append` | PASS |
| Read delegation — same history, order, tenant scope | PASS |
| Tenant isolation — tenant A evidence ≠ tenant B evidence | PASS |
| Run-local ordering — append order preserved per run | PASS |
| Execution Engine — no `RuntimeEventPersistence` import; uses port at bus boundary only | PASS |
| Single evidence flow — bus → port → adapter/store; no parallel durable writers | PASS |
| No lifecycle mutation — evidence path does not drive execution/retry/workflow | PASS |

---

## Regression gates (re-signoff)

| Suite | Role |
| ----- | ---- |
| `tests/unit/runtime/events/test_evidence_persistence_boundary.py` | Port, adapter parity, ordering, tenant isolation |
| `tests/unit/runtime/architecture/test_npsc5f_enterprise_evidence_certification.py` | Architecture freeze gates |
| `tests/unit/runtime/architecture/test_npsc5f_r1_durable_evidence_commit_tenant_integrity.py` | R1 durability + tenant matrix |
| `tests/unit/runtime/architecture/test_npsc5f_r1_durable_evidence_persistence_boundary_resignoff.py` | Re-signoff qualification record + drift sentinel |
| `test_npsc5f_p0_r1_protected_evidence_surfaces_have_no_unqualified_post_r1_drift` | P0 ownership-scoped drift |

---

## Production surface (authority unchanged)

| Layer | Role |
| ----- | ---- |
| `EvidencePersistencePort` | Stable interchange contract for producers and read models |
| `RuntimeEventPersistenceEvidenceAdapter` | Single adapter path to existing stores |
| `RuntimeEventPersistence` | Store-level durable authority (tenant routing, idempotency, positions) |
| `RuntimeEventBus` | Transport; persist-before-history/subscribers via port |

---

## Verdict

**R1 Durable Evidence re-signoff:** PASS on baseline `df677b5b37e0dcaa1e280b7a98324ee59ea24878`. Persistence boundary evolution does not alter R1 durable commit, tenant integrity, ordering, or execution lifecycle ownership.

**Next:** Continue NPSC-5F evidence plane gates (R2–R4) on `development`; treat this document as the qualification anchor for port-boundary changes through the re-signoff baseline.
