# EVOL-X — OD-EVOL-17 independent technical review

**Disposition:** **TECHNICAL REVIEW PUBLISHED / PROTOCOL 03 PENDING**  
**This file is not** `EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`.  
**No** `auditor_id`, human identity, or acceptance signature is recorded here.

**Session:** isolated Cursor agent session on `development`.  
**START_HEAD / review base:** `a20acf4d5fc23887fc3e331bdd5bd03cdb9f48f1`  
**OD-EVOL-17 publication SHA:** `7ab634dcf33bf4954a942b29bd573c9931617fb8`  
**Repository:** `jakbuczarnecki/intergrax`

---

## Authority determination

| Question | Result |
|----------|--------|
| OD-EVOL-17 | **OPERATOR APPROVED — LOCAL PATH A** (register). Active mechanism is ECAP-§2.B. |
| OD-EVOL-16 | Preserved as historical Path B selection. |
| `ECAP-P4-EVOL-SCOPE-AUTO-B6` | **`NOT_IN_FORCE`**. Machine attestation cannot accept B6. |
| OD-EVOL-11 `auditor_id` | **NOT RECORDED**. Appointment forbids agent self-appointment. |
| This session as Protocol 03 auditor | **NOT AUTHORIZED**. A separate session verifies process separation only. |
| Protocol 03 | **PENDING** |
| `B6_ACCEPTANCE_PUBLICATION_SHA` | **unset** |

---

## Technical verdict

**Verdict:** qualified TES/FINAL/CI evidence is internally consistent for `AUTOMATION_FIRST_SCOPED`. No W7-02 corrective defect found. Formal B6 acceptance is **not eligible**.

| Check | Evidence | Result |
|-------|----------|--------|
| Identity | branch `development`; HEAD `a20acf4d…`; remote `jakbuczarnecki/intergrax`; TES and FINAL are ancestors of HEAD | PASS |
| Mode A @ FINAL | `p03.verify_mode_a` on worktree HEAD `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` | `EVIDENCE_INTEGRITY_OK` |
| Bundle drift | `git diff --stat FINAL..HEAD` on `evol_x_p03_independent_replay` | empty |
| Profile | MANIFEST @ FINAL `certification_profile=AUTOMATION_FIRST_SCOPED`, `test_execution_sha=3c4a641b…` | PASS |
| M-09A / M-09H | metrics @ FINAL: M-09A `MEASURED` 3/3 PASS; M-09H `NOT_MEASURED` (`passed` null) | PASS (scoped) |
| FRZ behavioral | `evol_frz_evidence.v1.json` @ FINAL: FRZ-EVOL-01..10 `passed=true` | behavioral only |
| Manifest nodes | `evol_test_execution_manifest.v1.json` @ FINAL | 290 |
| CI | run `41337077334` conclusion `success`; job `evol-x-p3-l3-qualification`; step `Checkout TEST_EXECUTION_SHA` success; workflow head `a00674d6ba6af9b5676c0341823cd23b162bbb99` | PASS |
| CI JUnit | artifact `evol-x-p3-l3-3c4a641b3e48a13f437e830a0562e2a80b03aabb`; SHA256 `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4`; 290 testcases; 0 `failure`/`error` | PASS |
| Mode B full local pytest | not re-run | retained CI replay accepted |
| W7-02 | no technical failure of pinned evidence | **NOT APPLICABLE** |

Orchestrator `--skip-mode-b` reported Mode A `evidence_head=a20acf4d…` because `uv run --directory` left the process on the orchestrator checkout. The direct FINAL-worktree call above is the Mode A result used for this verdict. The evidence bundle bytes match between FINAL and that HEAD.

Adversarial `--evidence-ref` equal to TES exited 2. The recorded reason was `HEAD 3c4a641b3 != MANIFEST test_execution_sha 2b1655164` (the TES commit still carries the superseded manifest). That is fail-closed. It is not a defect in the FINAL bundle.

## Gates G-01..G-14

| Gate | This review |
|------|-------------|
| G-01 | PASS — CI run and JUnit hash re-checked |
| G-02 | **BLOCKED** — `auditor_id` absent (F-003) |
| G-03 | PASS scoped — M-09H NOT_MEASURED |
| G-04 | PASS — Mode A @ FINAL |
| G-05 | technical PASS; formal open until Protocol 03 |
| G-06 | technical PASS via retained CI replay; formal open until Protocol 03 |
| G-07 | **BLOCKED** — no signed PASS (F-004) |
| G-08 | **BLOCKED** — FRZ-EVOL-07 Outcome A/B not selected (F-005) |
| G-09 | **BLOCKED** — OD-EVOL-12 PENDING (F-006) |
| G-10 | **BLOCKED** — OD-EVOLX-08 PENDING (F-007) |
| G-11, G-12 | not defined as standalone gates; covered by W7-01 / G-14 |
| G-13 | ladder consistent; B6 SHA still empty |
| G-14 | section rollup sums to the nominal 102 IDs; semantic attestation remains formal-open |

102-ID source: [b6_technical_qualification/EVOL_X_B6_102_ID_COVERAGE_MATRIX.md](./b6_technical_qualification/EVOL_X_B6_102_ID_COVERAGE_MATRIX.md) (10+10+10+10+8+15+17+11+11). This review did not promote that rollup to a signed semantic acceptance.

## Scope exclusions (explicit)

Not claimed: ARCH-FREEZE, global FRZ PASS, PROD-Q live qualification, ENT-AUDIT-X CLOSED, production `ActivationService.rollback`, W4-04 port (OD-EVOL-05b Option 1), human M-09H measurement, Auto-B6 key custody, or an independent security-review signature. FRZ-EVOL-07 footnote @ FINAL limits the behavioral pass to the qualification-layer assessor (DM-01..07).

## Waves not executed

W7-01 Protocol 03 PASS, W7-03, W7-04, W7-05 closure, and W7-06 handoff stay blocked on the named OD-EVOL-11 auditor. OD-EVOL-17 does not substitute for OD-EVOL-12 or OD-EVOLX-08.

**Parent status:** **OPEN**.
