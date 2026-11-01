# EVOL-X — Independent B6 audit package (E3 PREP)

**Disposition:** preparation only — **no B6 PASS**, **no FRZ promotion**, **no agent signature**.

**Authority:** [EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md](../EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md) Part V–VII; [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](../ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) B6; closure plan E3 / §7.5–7.6.

## Active SHA ladder (closure chain)

| Role | SHA |
|------|-----|
| TEST_EXECUTION_SHA (TES) | `dcd7ae6e264d435994c61162e0d41aba25f71283` |
| EVIDENCE_TECH_PUBLICATION_SHA | *see EVOL_X_SHA_LADDER.md after TECH commit* |
| EVIDENCE_FINAL_PUBLICATION_SHA | *pending E2* |
| B6_ACCEPTANCE_PUBLICATION_SHA | *pending E3* |
| DOC / prep tip @ session | `0231ff9a198dddb3fcc7bac1a5f1725c731ef1c7` |

**Superseded (forbid in active records):** TES `6fec9ec…`, TECH `43a44b…`; historical TES `cb407403…`, TECH `20a3bc63…`.

**Prerequisite:** Named **OD-EVOL-11** auditor (≠ implementer, ≠ Cursor agent) recorded in [EVOL_X_OPERATOR_DECISION_REGISTER_W1.md](./EVOL_X_OPERATOR_DECISION_REGISTER_W1.md) before W7-01 / B6 sign-off.

---

## Certification matrix reference

Embedded **102×ID** matrix: EVOL-X Final Closure plan §5 (5-A … 5-J). Agent E0 notes:

- **M-03R / M-03C / M-03E** — osobne suplementy w macierzy (wiersz zbiorczy M-03R/C/E); w B6 review nie scalać z **M-03** precision bez jawnego odwołania do `m03_causal_audit.v1.json` i testów `test_evol_x_m03_causal_audit.py`.
- **M-09T** vs **M-09H** — M-09T MEASURED @ TECH ≠ human attribution; final wymaga M-09H.

Deliverable B6: reconcile Part VII rows → PASS (SCOPED) or **EXACT BLOCKERS** list with evidence SHA citations.

---

## Mode A — evidence integrity (final disposition)

**When:** @ **EVIDENCE_FINAL_PUBLICATION_SHA** (after E2), clean checkout.

```powershell
cd <repo>
git checkout <EVIDENCE_FINAL_SHA>
uv sync --extra dev-ci --frozen
uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish
```

**Technical disposition (already valid @ TECH lineage, not closure):**

```powershell
uv run python scripts/qualification/evol_x_p03_evidence.py verify-evidence-publish --technical
```

**Auditor checks (G-05):**

- Ancestor: TES ancestor of evidence HEAD.
- `MANIFEST.test_execution_sha` = TES.
- `ARTIFACT_DIGESTS.json` matches bytes @ HEAD.
- Final: `evol_m09_human_drill.v2.json` (schema v2, `measurement_policy_id` OD-EVOL-08-M1; dual SLA: tool ≤120 s/case, median attribution ≤900 s; runbook [EVOL_X_M09H_HUMAN_DRILL_RUNBOOK.md](./EVOL_X_M09H_HUMAN_DRILL_RUNBOOK.md)) + M-09H 3/3 MEASURED PASS + worksheet digest chain.
- **STOP:** final Mode A @ TECH-only HEAD for closure claim.

---

## Mode B — independent replay @ TES (G-06)

**When:** HEAD = TES; auditor-only (not implementer convenience replay as B6 substitute).

```powershell
git checkout 6fec9ec387089dfcee1039089ba5ad901fab39d6
uv sync --extra dev-ci --frozen
uv run pytest tests/qualification/evol_x -q --junitxml=.tmp/session/evol-x-b6/junit.xml
uv run python scripts/qualification/evol_x_p03_evidence.py verify-test-execution `
  --evidence-ref <NEW_TECH or EVIDENCE_FINAL SHA> `
  --certification --replay-junit .tmp/session/evol-x-b6/junit.xml
```

**Expected:** match `evol_junit_outcomes.v1.json` @ bundle (279 passed population @ NEW_TES; N-M09-17 anti-regression included).

---

## FRZ-EVOL-07 — W7-03 checklist (Part X, G-08)

**Not sufficient alone:** S7± green, M-08 @ TECH, OD-EVOL-05b Option 1.

**Mandatory five-step audit** (named auditor) — document **Outcome A** or **Outcome B** in B6 worksheet:

1. **Checklist wording** — [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](../PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) FRZ-EVOL-07 @ L275: czy scoped qual-layer PASS jest normatywnie uzasadniony **bez** live PROD-Q enforcement?
2. **Part X bundle** — S7+ **and** S7− (DM-04 REFUSE); DM-01..07; `evol_frz_evidence.v1` footnote qual-only; forbidden claims absent.
3. **Enforcement boundary** — [EVOL_X_ENFORCEMENT_BOUNDARY_W1.md](./EVOL_X_ENFORCEMENT_BOUNDARY_W1.md): B6 text **must not** claim production `ActivationService.rollback` refuses DM-04 in operator envs.
4. **W4-04** — confirm **not implemented** (Option 1); production rollback hardening claims → STOP.
5. **Decision record** — Outcome A (scoped PASS text for B6 only) or Outcome B (OPEN + operator path).

**FRZ-EVOL-07 checklist column → PASS** only in **E4** after Outcome A + operator authorization — **never** E3 alone.

---

## CI evidence slot (W6-04 / G-01)

Parallel gate — not substitute for Mode B.

| Slot | Source |
|------|--------|
| WORKFLOW_DEF_SHA | `main` after workflow-only merge |
| Dispatch TES | `dcd7ae6…` (NEW_TES) input; `6fec9ec…` dispatch INVALIDATED |
| Run proof | [EVOL_X_CI_QUALIFICATION_RECORD.md](./EVOL_X_CI_QUALIFICATION_RECORD.md) |
| Procedure | [EVOL_X_CI_QUALIFICATION_PROCEDURE.md](./EVOL_X_CI_QUALIFICATION_PROCEDURE.md) |

Operator runbook (prep): `.tmp/session/evol-x-final-cert/w6-04-operator-runbook.md`

---

## B6 deliverable template

**Path (to create @ B6 SHA):**  
`docs/project/maintainers/qualification/evol_x/EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`

**Minimum sections:**

1. **Publication SHA** — `B6_ACCEPTANCE_PUBLICATION_SHA`
2. **Evidence refs** — TES, EVIDENCE_FINAL (or documented chain), `independent_replay_sha` = TES
3. **Mode A** — command, exit status, evidence HEAD
4. **Mode B** — JUnit replay match statement
5. **Semantic review (G-14)** — M-02/M-03, CW-09/10, S− arms — scoped findings
6. **W7-03 FRZ-07** — Outcome A or B (verbatim)
7. **CI** — C-CI-01 row or EXACT BLOCKER if G-01 open
8. **Disposition** — `PASS (SCOPED)` **or** `EXACT BLOCKERS` (numbered)
9. **Signatures** — named independent auditor only

---

## Findings register (template)

| Finding ID | Severity | Norm / Gate | Evidence SHA | Disposition | Operator action |
|------------|----------|-------------|--------------|-------------|-----------------|
| F-001 | | | | OPEN / SCOPED PASS / N/A | |
| … | | | | | |

**Agent row forbidden:** any line stating «B6 PASS by agent» or FRZ promotion.

---

## G-xx blockers @ prep time (execution)

| ID | Blocks B6 / closure |
|----|---------------------|
| G-01 | W6-04 CI OPEN |
| G-02 | OD-EVOL-11 name missing |
| G-03 | M-09H drill |
| G-04 | EVIDENCE_FINAL empty |
| G-05 | Mode A final not @ FINAL |
| G-06 | Mode B not independent |
| G-07 | No B6 record |
| G-08 | W7-03 not decided |
| G-09–G-10 | OD-12 / FRZ rows |
| G-13 | SHA chain incomplete |

---

## E4 sequence (auditor read-only until B6 DONE)

1. B6 published  
2. OD-EVOL-12  
3. FRZ-EVOL-01..10 (FRZ-07 only if W7-03 Outcome A)  
4. OD-EVOLX-08  
5. Roadmap §5 + FORMAL_CLOSURE_SHA  
6. ENT-AUDIT-X handoff non-template  

**Agent:** E4 **BLOCKED** until B6.

---

*Prepared by Cursor agent session EVOL-X Final Certification E3 PREP — not an independent audit.*
