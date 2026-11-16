# EVOL-X → ENT-AUDIT-X handoff record

**Schema:** intake record for **ENT-AUDIT-X**  
**Status:** **`EVOL-X → ENT-AUDIT-X FORMAL HANDOFF PUBLISHED`** (W7-06, 2026-11-08; internal handoff complete 2026-11-05)  
**Formal W7-06 pin:** published as `W7_06_HANDOFF_PUBLICATION_SHA` in [EVOL_X_SHA_LADDER.md](./EVOL_X_SHA_LADDER.md) (two-phase, non-self-referential); formal closure `FORMAL_CLOSURE_PUBLICATION_SHA` = `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca`  
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
| EVOL-X **formal** closure SHA | `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca` — `FORMAL_CLOSURE_PUBLICATION_SHA` |
| B6 **formal** acceptance SHA | `1dd769331506149a1ee2f411c2bc62e85e5bbf25` — `B6_ACCEPTANCE_PUBLICATION_SHA` (IIAC, PASS_SCOPED) |
| Internal B6 verdict | **`INTERNAL_B6_TECHNICAL_ACCEPTED`** — [EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md](./EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md) |
| Operator technical acceptance | **OD-EVOL-18** (preserved) |

The formal predecessor **`EVOL-X` = CLOSED — INDEPENDENTLY ACCEPTED (AUTOMATION_FIRST SCOPED) — PUBLISHED** is satisfied at `FORMAL_CLOSURE_PUBLICATION_SHA` (IIAC B6 acceptance; OD-EVOL-12 and OD-EVOLX-08 signed). The operator signature lines were written by operator-designated signing sessions under IIAC Amendment 01 (operator-adopted, not audited, not mechanically enforced; assurance `process_enforced`). This does not make ENT-AUDIT-X, ARCH-FREEZE or PROD-Q closed.

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

**Formal:** Independent B6 acceptance under IIAC-2026-11-05.1 @ `B6_ACCEPTANCE_PUBLICATION_SHA` `1dd769331506149a1ee2f411c2bc62e85e5bbf25` (`iiac_evidence/EVOL-X/b6_acceptance/EVOL_X_IIAC_B6_ACCEPTANCE_RECORD.md`, scope PASS_SCOPED with 12 bound limitations). The internal record above stays scoped technical acceptance only and is not the formal acceptance.

---

## 8. Remaining exclusions (ENT must respect)

- PROD-Q live integration OPEN
- M-09H human drill NOT_MEASURED on scoped profile
- Qualification-only FRZ-EVOL-07 (no production rollback hardening claim)
- Vendor live qualification (Dimension B) unchanged
- ENT-AUDIT-X, ARCH-FREEZE and PROD-Q live qualification are not closed by the EVOL-X closure
- Legacy human Protocol 03 is superseded by the IIAC B6 acceptance for EVOL-X (see ladder `FORMAL_ECAP_INDEPENDENT_B6`)

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

**Formal checklist (W7-06):**

- [x] Independent B6 acceptance (IIAC record, replaces the legacy signed Protocol 03 PASS for EVOL-X)
- [x] Formal OD-EVOLX-08 signature (`iiac_evidence/EVOL-X/operator_decisions/EVOL_X_OD_EVOLX_08_DECISION_RECORD.md`, delegated, process_enforced)
- [x] `FORMAL_CLOSURE_PUBLICATION_SHA` `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca`
- [x] D10 digest chain attested by an independent verifier session ([EVOL_X_D10_DIGEST_CHAIN_ATTESTATION.md](./EVOL_X_D10_DIGEST_CHAIN_ATTESTATION.md))

---

## 12. Finalization procedure

**Internal handoff:** complete @ `EVOL-X-INTERNAL-FINAL-CLOSURE`.

**Formal W7-06 pin:** this record and the D10 attestation are published in the phase-1 commit; the phase-2 commit binds that commit's SHA in the ladder row `W7_06_HANDOFF_PUBLICATION_SHA` and in `EVOL_X_CLOSURE_EXECUTION_STATUS.json` (`w7_06_handoff_publication_sha`). The record itself does not contain its own SHA.

**Do not** treat the internal handoff alone as the formal predecessor; the formal predecessor is the closure at `FORMAL_CLOSURE_PUBLICATION_SHA`.
