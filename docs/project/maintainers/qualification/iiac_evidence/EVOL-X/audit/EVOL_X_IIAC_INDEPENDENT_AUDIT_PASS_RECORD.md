# EVOL-X — IIAC independent audit PASS record

**Verdict:** `EVOL-X — INDEPENDENT IIAC AUDIT PASS` · manifest verdict `PASS_SCOPED`
**Record type:** independent auditor record under `IIAC-2026-11-05.1` (`IN_FORCE`). Parallel to the human Protocol 03 record; **no human signature block**.
**Status of this record:** auditor deliverable. It is **not** a certification publication, **not** a closure claim and **not** a ladder or roadmap update.

## 1. Identity

| Field | Value |
|---|---|
| Stage | `EVOL-X` |
| Profile | `iiac-v1` (certification profile of the audited evidence: `AUTOMATION_FIRST_SCOPED`) |
| Auditor role | `independent-quality-auditor` |
| Auditor session | `IIAC-C-EVOL-X-INDEPENDENT-AUDIT-2026-11-05` (Claude Code session `a18c2476-3b59-4cc7-bc0f-fa22b25edcfc`, model `claude-sonnet-5-5`) |
| Implementer / certifier | recorded separately in the handoff record (operator attestation pending) and the certifier-completed manifest |
| Audit date | 2026-11-05 (+01:00) |
| Assurance level | `process_enforced` |

## 2. Exact SHAs

| Role | SHA |
|---|---|
| `audited_implementation_sha` = `test_execution_sha` (TES) | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| `evidence_publication_sha` (FINAL) | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| `audit_execution_sha` | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` (last exact-SHA replay checkout) |
| Repository HEAD during the audit | `06a459afbab3aa8cea89e4118865d55e7b537a34` |
| IIAC activation (`effective_from_sha`) | `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` |
| Qualified CI run | `41337077334` @ TES |
| `B6_ACCEPTANCE_PUBLICATION_SHA` | not set by the auditor |
| `FORMAL_CLOSURE_PUBLICATION_SHA` | not set by the auditor |

## 3. Result

| Check | Result |
|---|---|
| Mode A @ FINAL (canonical P03 CLI, auditor-run) | PASS |
| Mode B @ TES, `--certification`, auditor's own 290-test replay | PASS; outcome map identical to the published bundle |
| Regression at HEAD `06a459afb` | 290/290 |
| CI provenance | success; TES checkout assertion verified in the job log |
| Mode C adversarial | Mode A 12/12 tamper cases detected; Mode B 5/5 replay tampers rejected; 3 CLI negatives rejected |
| Independent derivations | 27/27 artifact digests; 16/17 metric readings identical (M-09A re-derived by Mode A); 0 missed production consumers in 60 real changes; 0 under-selection in 56 real paths; 384 DM and 128 WM inputs enumerated |
| Master plan Part VII | 105 rows evaluated (`part_vii_id_matrix.json`) |
| Findings | 0 `PROVEN_ARCHITECTURAL_DEFECT`, 0 `IN_SCOPE_BLOCKER`; 11 recorded (F-C-01..F-C-11) |

## 4. FRZ-EVOL criterion-level result (no mass PASS, no promotion)

| FRZ_ID | Auditor result | Limitation |
|---|---|---|
| FRZ-EVOL-01 | PASS — SCOPED | LIM-03 |
| FRZ-EVOL-02 | PASS | — |
| FRZ-EVOL-03 | PASS | — |
| FRZ-EVOL-04 | PASS | — |
| FRZ-EVOL-05 | PASS | — |
| FRZ-EVOL-06 | PASS — SCOPED | LIM-01, LIM-03, LIM-04 (EVOL_CONSERVATIVE_FULL only; FINAL bundle footnote empty: F-C-02) |
| FRZ-EVOL-07 | PASS — SCOPED | LIM-05 (qualification layer only; no production enforcement claim) |
| FRZ-EVOL-08 | PASS — SCOPED | LIM-02 (change-time diagnostics; M-09H not measured; incident/deploy correlation = ENT-AUDIT-X) |
| FRZ-EVOL-09 | PASS — SCOPED | LIM-06 |
| FRZ-EVOL-10 | PASS — SCOPED | LIM-07 (scenario-wave audit = ENT-AUDIT-X) |

The auditor changed no checklist row. Rows stay `OPEN` until the certifier publishes them one by one. A global FRZ PASS is not supported by this record.

## 5. Limitations (all travel with the verdict)

LIM-01 `EVOL_CONSERVATIVE_FULL` selection scope only · LIM-02 `AUTOMATION_FIRST_SCOPED`, `M-09H` NOT_MEASURED · LIM-03 direct-import test ownership · LIM-04 M-03 on self-surface arcs, oracle O-X · LIM-05 FRZ-EVOL-07 qualification layer only · LIM-06 9-surface frozen-core registry · LIM-07 post-freeze requalification owned by ENT-AUDIT-X · LIM-08 audited tree is TES/FINAL · LIM-09 Windows local replay, Ubuntu via CI log · LIM-10 process-enforced assurance. Definitions: `IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md` §9.

## 6. FRZ-EVOL-07 wording determination (master plan Part X, W7-03)

The checklist wording *qualified or unsafe refusal explicit* supports a **qualification-layer scoped** claim. It does not support a claim that unsafe rollback is blocked in operator environments (S7 negative: `ad_layer_alone_succeeds=True`). Permitted text: `PASS (EVOL-X scoped)` with the qualification-layer footnote. Details: dossier §8.

## 7. Explicit non-actions

No product-path edit · no tracked-file edit · no commit or push · no FRZ promotion · no ladder or roadmap edit · no `B6_ACCEPTANCE_PUBLICATION_SHA` · no `FORMAL_CLOSURE_PUBLICATION_SHA` · no activation or amendment of any policy · no promotion of `EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md` or the earlier Protocol 03 automated record (read as context only; every conclusion re-derived).

## 8. Evidence

Directory `docs/project/maintainers/qualification/iiac_evidence/EVOL-X/audit/` (byte-preserved, `-text`): dossier, findings, scope, test matrix, FRZ and Part VII matrices, manifest draft, handoff documents, probe scripts, logs and `EVIDENCE_INDEX.json` with SHA-256 digests.

## 9. Handoff

Certification handoff to a separate `certification-governance-agent`: `EVOL_X_IIAC_CERTIFICATION_HANDOFF.md`. The manifest is a draft until the operator attests the Session C handoff record and the certifier completes its fields and runs the validators.
