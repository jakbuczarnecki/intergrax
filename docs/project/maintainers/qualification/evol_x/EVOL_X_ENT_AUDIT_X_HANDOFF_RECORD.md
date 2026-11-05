# EVOL-X → ENT-AUDIT-X handoff record

**Schema:** intake record for **ENT-AUDIT-X**  
**Status:** **`EVOL-X → ENT-AUDIT-X INTERNAL HANDOFF COMPLETE`** (2026-11-05)  
**Formal W7-06 pin:** **BLOCKED** — requires formal `FORMAL_CLOSURE_PUBLICATION_SHA` (unset)  
**Campaign:** [`EVOL_X_INTERNAL_CLOSURE_CAMPAIGN_TERMINAL_REPORT.md`](./EVOL_X_INTERNAL_CLOSURE_CAMPAIGN_TERMINAL_REPORT.md)  
**Authorization:** `OPERATOR-DIRECTED INTERNAL CLOSURE CAMPAIGN`

---

## 1. Handoff disposition

| Field | Value |
|-------|--------|
| Source stage | EVOL-X |
| Target stage | ENT-AUDIT-X |
| Certification profile accepted | `AUTOMATION_FIRST_SCOPED` |
| EVOL-X **internal** closure SHA | **`INTERNAL_CLOSURE_PUBLICATION_SHA`** — [EVOL_X_SHA_LADDER.md](./EVOL_X_SHA_LADDER.md) |
| EVOL-X **formal** closure SHA | **unset** — `FORMAL_CLOSURE_PUBLICATION_SHA` |
| B6 **formal** acceptance SHA | **unset** — `B6_ACCEPTANCE_PUBLICATION_SHA` |
| Internal B6 verdict | **`INTERNAL_B6_TECHNICAL_ACCEPTED`** — [EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md](./EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md) |
| Operator technical acceptance | **OD-EVOL-18** (preserved) |

ENT-AUDIT-X may consume **internal** handoff artifacts for intake drafting. **Formal** dependency **`EVOL-X` = CLOSED — INDEPENDENTLY ACCEPTED** under ECAP remains **OPEN**.

---

## 2. Accepted audit baseline (pinned evidence)

| Role | SHA |
|------|-----|
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| Qualified CI | `41337077334` |
| R2 global reconciliation | `167583ad1d33d9eb6e82589661e6d18aee15fa92` |
| Local B6 unattended | `42363732d9f450bfe4329d76a96c646d2ab730b2` |
| Plan publication | `b4fca4bc413a650b2e5a8dd8826669a9025864c5` |

---

## 3. Internal acceptance package (ENT intake)

| Artifact | Purpose |
| --- | --- |
| [EVOL_X_INTERNAL_FRZ_EVOL_ACCEPTANCE_MATRIX.md](./EVOL_X_INTERNAL_FRZ_EVOL_ACCEPTANCE_MATRIX.md) | FRZ scoped acceptance — formal rows OPEN |
| [EVOL_X_INTERNAL_PARENT_CLOSURE_RECORD.md](./EVOL_X_INTERNAL_PARENT_CLOSURE_RECORD.md) | Internal parent close |
| [EVOL_X_INTERNAL_FINDINGS_RECONCILIATION_MATRIX.md](./EVOL_X_INTERNAL_FINDINGS_RECONCILIATION_MATRIX.md) | Outstanding formal-only flags |
| [ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md](../ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md) | Cross-stage gates |

---

## 4. D10 digest chain (reconciliation)

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

**Drift policy:** Material change to closed-world inventory or VL3 manifest after TES → new EVOL requalification — not ENT substitute.

---

## 5. Accepted impact graph and regression-selection contract

| Deliverable | Consumer |
|-------------|----------|
| Change-impact graph builder outputs @ TES | ENT-AUDIT-X regression selection |
| `SelectionDecision` / conservative policy (FRZ-EVOL-06) | ENT baseline runs |
| COMPAT-at-change adapter evidence (FRZ-EVOL-04) | ENT static checks |

Do **not** re-implement in ENT-AUDIT-X — **orchestrate** existing qualification gates.

---

## 6. Evidence oracle

| Oracle | Purpose |
|--------|---------|
| Closed-world gates CW-01..10 | Structural drift |
| FRZ behavioral rows | Change-safety claims |
| P03 experiment verdict JSON | Experiment-class regression |
| CI JUnit canonical hash `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4` | Replay parity |

---

## 7. Change-safety certification record

**Internal:** [EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md](./EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md) — scoped technical acceptance only.

**Formal:** Independent B6 @ `B6_ACCEPTANCE_PUBLICATION_SHA` — **not recorded**. ENT must not treat internal record as formal Protocol 03 attestation.

---

## 8. Remaining exclusions (ENT must respect)

- PROD-Q live integration OPEN
- M-09H human drill NOT_MEASURED on scoped profile
- Qualification-only FRZ-EVOL-07 (no production rollback hardening claim)
- Vendor live qualification (Dimension B) unchanged
- **Formal ECAP EVOL-X closure OPEN**

---

## 9. Freshness and drift detection (ENT-AUDIT-X requirements)

ENT-AUDIT-X intake must implement:

- Baseline SHA comparison vs accepted enterprise baseline post-freeze
- Classification of drift (BLOCKER / WARN / INFO)
- Mandatory vendor hardcoding checks per roadmap §3.1 (reference EVOL-X accepted artifacts, do not duplicate EVOL-X closure)

---

## 10. Mandatory future requalification triggers

Per FRZ-EVOL-10 / post-freeze governance — VL3 manifest or closed-world inventory change after TES requires new EVOL qualification wave; internal closure does not waive.

---

## 11. ENT-AUDIT-X intake checklist (internal handoff)

- [x] This handoff record @ internal closure descendant SHA
- [x] Internal B6 acceptance record (not formal PASS)
- [x] OD-EVOLX-08-INTERNAL companion
- [x] SHA ladder internal closure row
- [x] Internal FRZ matrix (formal checklist still OPEN)
- [x] Qualification tree paths (no `.tmp/` session logs as SSOT)

**Formal checklist (still open):**

- [ ] Copy of **signed** Protocol 03 PASS (scoped)
- [ ] Formal OD-EVOLX-08 signature
- [ ] `FORMAL_CLOSURE_PUBLICATION_SHA`

---

## 12. Finalization procedure

**Internal handoff:** complete @ `EVOL-X-INTERNAL-FINAL-CLOSURE`.

**Formal W7-06 pin:** remains blocked until W7-05 publishes `FORMAL_CLOSURE_PUBLICATION_SHA` under authorized independent certification.

**Do not** mark ENT formal predecessor satisfied by internal handoff alone.
