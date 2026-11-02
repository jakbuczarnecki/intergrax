# EVOL-X → ENT-AUDIT-X handoff record

**Schema:** intake record for **ENT-AUDIT-X** (no ENT implementation in this wave)  
**Status:** **PREPARED — FINALIZE @ descendant of `FORMAL_CLOSURE_PUBLICATION_SHA`**  
**Authority:** Master plan v3 W7-06 · D10 digest reconciliation

---

## 1. Handoff disposition

| Field | Value |
|-------|--------|
| Source stage | EVOL-X |
| Target stage | ENT-AUDIT-X |
| Certification profile accepted | `AUTOMATION_FIRST_SCOPED` |
| EVOL-X closure SHA | *TBD — `FORMAL_CLOSURE_PUBLICATION_SHA`* |
| B6 acceptance SHA | *TBD — `B6_ACCEPTANCE_PUBLICATION_SHA`* |

Until closure SHAs exist, this record is **PREPARED** for intake drafting only.

---

## 2. Accepted audit baseline (pinned evidence)

| Role | SHA |
|------|-----|
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| Qualified CI | `41337077334` |
| Plan publication | `b4fca4bc413a650b2e5a8dd8826669a9025864c5` |

---

## 3. D10 digest chain (reconciliation)

ENT-AUDIT-X must reconcile publication digests against accepted baseline:

| Artifact class | Location / binding |
|----------------|-------------------|
| Evidence MANIFEST | @ FINAL commit — fields `test_execution_sha`, `plan_publication_sha` |
| ARTIFACT_DIGESTS | Content digests @ evidence HEAD (Mode A procedure, master plan v3 V-0) |
| VL3 manifest | `evol_vl3_execution_manifest.v1.json` @ TES tree |
| FRZ bundle | `evol_frz_evidence.v1.json` @ FINAL/TES bundle |
| Metrics | `evol_metrics.v1.json` — M-09A MEASURED, M-09H NOT_MEASURED |
| Experiments | `experiments/` arms S1..S8 @ FINAL bundle |
| Impact / selection | Qualification modules under `tests/qualification/evol_x/` — contracts for ENT orchestration |

**Reconciliation rule:** `merge-base --is-ancestor TEST_EXECUTION_SHA EVIDENCE_FINAL_PUBLICATION_SHA` — required.

**Drift policy:** Material change to closed-world inventory or VL3 manifest after TES → new EVOL re-qualification — not ENT substitute.

---

## 4. Accepted impact graph and regression-selection contract

| Deliverable | Consumer |
|-------------|----------|
| Change-impact graph builder outputs @ TES | ENT-AUDIT-X regression selection |
| `SelectionDecision` / conservative policy (FRZ-EVOL-06) | ENT baseline runs |
| COMPAT-at-change adapter evidence (FRZ-EVOL-04) | ENT static checks |

Do **not** re-implement in ENT-AUDIT-X — **orchestrate** existing qualification gates.

---

## 5. Evidence oracle

| Oracle | Purpose |
|--------|---------|
| Closed-world gates CW-01..10 | Structural drift |
| FRZ behavioral rows | Change-safety claims |
| P03 experiment verdict JSON | Experiment-class regression |
| CI JUnit canonical hash `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4` | Replay parity |

---

## 6. Change-safety certification record

Independent B6 record @ `B6_ACCEPTANCE_PUBLICATION_SHA` attests scoped change-safety qualification — ENT-AUDIT-X treats this as **historical acceptance**, not live enforcement.

---

## 7. Remaining exclusions (ENT must respect)

- PROD-Q live integration OPEN
- M-09H human drill NOT_MEASURED on scoped profile
- Qualification-only FRZ-EVOL-07 (no production rollback hardening claim)
- Vendor live qualification (Dimension B) unchanged

---

## 8. Freshness and drift detection (ENT-AUDIT-X requirements)

ENT-AUDIT-X intake must implement:

- Baseline SHA comparison vs accepted enterprise baseline post-freeze
- Classification of drift (BLOCKER / WARN / INFO)
- Mandatory vendor hardcoding checks per roadmap §3.1 (reference EVOL-X accepted artifacts, do not duplicate EVOL-X closure)

---

## 9. Required ENT-AUDIT-X intake evidence checklist

- [ ] This handoff record @ closure descendant SHA
- [ ] Copy of B6 Protocol 03 PASS (scoped)
- [ ] OD-EVOLX-08 signature record
- [ ] SHA ladder closure row
- [ ] FRZ-EVOL promoted rows snapshot
- [ ] Link to qualification tree paths (no `.tmp/` session logs)

---

## 10. Finalization procedure (W7-06)

1. Complete W7-05 → obtain `FORMAL_CLOSURE_PUBLICATION_SHA`.
2. Update §1 closure/B6 SHA fields in this document.
3. Commit @ descendant of closure SHA.
4. Set `batch_5_ent_handoff.status: COMPLETE` in [EVOL_X_CLOSURE_EXECUTION_STATUS.json](./EVOL_X_CLOSURE_EXECUTION_STATUS.json).

**Do not start ENT-AUDIT-X implementation commits in the same closure wave.**
