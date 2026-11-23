# PLAN T-IGX-046-E — EQO-A remediation nodes PT-D04 (batch DB-C) and PT-R03 (EDER owner confirmation): stage-record notes and owner-vocabulary data

This ticket certifies nothing, promotes no FRZ criterion, changes no stage status and edits no product path. Every added sentence either records an Operator decision that is published in the repository, states a fact that was re-read at the head named below, or quotes a roadmap row with attribution; none adds an acceptance stamp or an independence claim.

## 1. Purpose

Two documentation and data nodes of the EQO-A Campaign A remediation (findings baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589), in one pull request, re-verified at `origin/development` `2727a2c2364a97492202b931375772a400279a9b`:

- **PT-D04 (batch DB-C):** append-only dated notes on closed stage records. Findings addressed: EQOA-A2-007 (GOV-X1, EBH-4, INT-CONFIG-REAL-X, S24-GAP-04), EQOA-A5-014, EQOA-A2-021, EQOA-A8-022. Operator decisions applied (published in `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_DECISIONS_EQOA_REMEDIATION.md`): DP-04 (qualified CLOSED labels LBL-L / LBL-C for stages without a preserved audit artefact, no independence implied), DP-09 (evidence basis per stage), DP-45 (MCP-SERVER-X conditions C5/C8, R-P7-04/06, R-P8-01). The notes are the program's chosen way to bring the records in line with the roadmap wording (batch DB-C, option "add notes").
- **PT-R03:** the Operator confirmed on 2026-11-17 (DP-47 option a, addendum 4 of `OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md`) the 55-ID owner vocabulary of the Enterprise Debt & Exceptions Register as canonical. This change flips the recorded status in the register data and documents it.

## 2. PT-D04 scope: 7 of 16 hunks applied (55 added lines, 0 changed or deleted lines)

| Hunk | File | Note |
|---|---|---|
| DB-C-H01 | `docs/project/maintainers/qualification/GOV_X1_GOVERNANCE_AUTHORITY_BOUNDARY_RECERTIFICATION.md` | Record reads READY FOR AUDIT (L62-L63, L67); roadmap L368/L2001 wording quoted; DP-04 wording quoted; DP-09 names an artefact or an audit; no acceptance artefact preserved |
| DB-C-H02 | `docs/project/maintainers/qualification/EBH_4_COMMUNICATION_COMPOSITION_BYPASS_CERTIFICATION.md` | Record BLOCKED / READY FOR AUDIT (L15, L445, L1391, L1393); roadmap L388/L1991 quoted; DP-04 wording quoted; DP-09 evidence basis named, not assessed |
| DB-C-H10 | `docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md` | Supersession note on row `OD-HIST04-A` (roadmap EVOL-X cell replaced by commit `e46ddcef1c0d8e9ef6d3f23d971af156713e7fb0` on 2026-11-11); matrix disposition not altered |
| DB-C-H13 | `docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_ARCH_FREEZE_HANDOFF.md` | Post-MCP-RC status of R-P8-01, R-P7-04, R-P7-06, ADR-MCP-RC-01 (PROPOSED); DP-45 recorded; no re-verification is performed |
| DB-C-H14 | `docs/project/maintainers/architecture/INT_CONFIG_REAL_X_EXISTING_CAPABILITY_CONFIGURATION_REALIZATION.md` | P0 lock record reads BLOCKED (L16); roadmap L377 and ledger L1974-L1982 quoted; DP-04 wording quoted; DP-09 evidence basis named, not assessed |
| DB-C-H15 | `docs/project/maintainers/architecture/S24_GAP_04_EXTERNAL_CONTRACT_SEMANTIC_COMPATIBILITY.md` | P0-R1 lock PASS pending audit (L733); roadmap L369, ledger L1966-L1973 quoted; DP-04 general reference only |
| DB-C-H16 | `docs/project/maintainers/qualification/TRACE_X_P6_RESTART_RESUME_TERMINAL_CAUSALITY_CERTIFICATION.md` | Path clarification for the L92 shorthand (`resume_admission.py`, `reentry_admission.py`); no status effect |

Source: `.tmp/session/eqo-a-remediation/WP-9/patches/DB-C/` (`PATCH.diff`, `hunks.jsonl`, `FACT_CHECK.md`, `NOTES.md`), re-verified and reworded as listed in section 4.

## 3. Held hunks (9 of 16, not applied)

A hunk that depends on an Operator or owner fact that is not published is dropped as a whole, not softened.

| Hunk | File | Missing fact |
|---|---|---|
| DB-C-H03 | `AW_7C_A2_SCOPED_ADAPTIVE_INTEGRATION_QUALIFICATION.md` | Whether AW-7C-P2, -P3, -P4 and -CERT are folded into the final parent certification `2fab207f3e850f1b339523587a19906ea1e9b292` (naming the SHAs that carry their gates), or closure rows with SHAs (no tracked file states either) |
| DB-C-H04 | `ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W4_PARENT_RECERTIFICATION.md` | Operator confirmation of the digests of the local replay logs `t11-a.log` and `t11-b.log` and of their custody, or their publication under `iiac_evidence/` (W3:OQ-7) |
| DB-C-H05 | `ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_PARENT_RECERTIFICATION.md` | Same for `v11-batch-a.log` and `v12-batch-b.log` |
| DB-C-H06 | `ENTERPRISE_RUNTIME_INTELLIGENCE_W6_PARENT_RECERTIFICATION.md` | Owner statement: run command, SHA and log digest of the roadmap-quoted counts (gate 10/10, batches 52/52), or that they are roadmap-asserted only |
| DB-C-H07 | `HARNESS_W7_HOST_01_CURRENT_HEAD_RECERTIFICATION.md` | Owner: node IDs and classification of the 7 failed tenant/identity tests (not recorded in the record) |
| DB-C-H08 | `HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md` | Owner: where the `npsc42` failure was finally resolved or accepted |
| DB-C-H09 | `pre_ent_enterprise_certification/PRE_ENT_WAVE6_CROSS_AUDIT_REPORT.md` | Operator: archive the local log `parent_regression_5gate.log` under `iiac_evidence/` or leave it as local evidence, and the G-SKILL decision together with SKILL-X (W3:OQ-7) |
| DB-C-H11 | `ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md` | Operator: whether the session designation `ent-sign-w3-2026-11-09` covers the W5 signing session `ent-sign-w5-od-08-2026-11-11`, or the W5 designation |
| DB-C-H12 | `HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md` | Owner: closure record for row U transport parity, classification or owner of the CE-02 compaction initiative, SHA and node IDs per row for FRZ-HRN-03..08 |

## 4. Checks and adjustments (PT-D04)

- All 16 target record files and the freeze checklist are byte-identical between the findings baseline and `2727a2c23`. The roadmap differs only in six PDC-X lines (L498, L570, L572, L1847, L1884, L2163); every roadmap row quoted by the notes (L365-L390, L487, L495, L1963-L2001, L2160) was re-read at `2727a2c23` and matches.
- The "no acceptance artefact preserved" statement was re-run at `2727a2c23` for each roadmap-cited SHA over the whole tree (not only `docs/`, `governance/`, `tests/`). The file lists changed since the baseline: the Campaign A dossier (PR #589), a historical-campaign invariant audit dossier, a scenario-factory inventory and test data now also cite some SHAs; none is an audit report or acceptance record of the stage and the notes name those classes. `iiac_evidence/` now holds `CERT-CAMPAIGN-2026-11-07`, `EQO-A-PT-D03`, `QUAL-X`, `QUAL-X-P0`, `SKILL-X` and `TOOL-LIFE-X` next to the earlier directories; none is named for GOV-X1, EBH-4, INT-CONFIG-REAL-X or INT-EXTCOMP-X.
- The prepared wording "CONDITIONAL ON W3:OQ-1 Option A being ruled by the Operator" is replaced by a reference to the published decision DP-04 and states that the roadmap row is not edited by this change. The notes quote only the published DP-04 wording ("qualified CLOSED (LBL-L / LBL-C) for stages without an audit artefact, no independence implied") and state no label text: the label wording of LBL-C is set out in the decision plan WP-3, which is not published, and is left to the roadmap batch. The qualifier "INDEPENDENCE: basis unstated" of the same plan is not named by DP-04, so DB-C-H15 refers to DP-04 only and leaves the form to the roadmap batch.
- DB-C-H01, -H02, -H14 add one bullet quoting DP-09 (GOV-X1: artefact or audit; EBH-4 and INT-CONFIG-REAL-X: ROADMAP-REPLAY-X class-A L2 evidence), stating that the evidence is not assessed here and is not treated as an independent audit.
- DB-C-H13: the placeholder bullet is replaced by the published decision DP-45; the note records it and states that it does not perform the re-verification of R-P8-01.
- The roadmap is quoted at `origin/development` `2727a2c2364a97492202b931375772a400279a9b`; the roadmap edit batch (DB-A) is not part of this change.
- The authority for "append notes on stage records" is the program decision DP-04 plus the commission of node PT-D04. W9:Q5(b) of the WP-9 plan (a numbered question, recommended option "add notes") has no separate DP number in the published record; the reviewer should confirm that DP-04 is accepted as sufficient.
- Notes are dated 2026-11-17 and are labelled "EQO-A remediation" (the prepared text said "draft"). The plan documents WP-3 and WP-9 are session artefacts and are not cited from the notes.
- Line endings: the matrix `PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md` is CRLF in the index; the inserted note keeps CRLF; all other files are LF. DB-C-H10 inserts its paragraph after the table block, before the next rule; no existing line changes.
- EBH-7 identity allowlist (`scripts/qualification/verify_ebh7_p03_immutable_evidence.py`, prefix `docs/project/maintainers/qualification/`): five touched record files lie under it; the two files under `docs/project/maintainers/architecture/` lie outside it. None of the seven is a canonical F10 input.

## 5. PT-R03 scope: owner vocabulary confirmed as data (4 files)

`owner_vocabulary.proposed.json` keeps its name: no schema (`register_schema.json`) and no script or test reads the string `owner_vocabulary` or the file name (searched in `scripts/`, `tests/`, `.github/`, `intergrax/` and `governance/`; the only hits are the tickets and plans T-IGX-047-A and this one, which cite the path); renaming would break citations in tickets and plans. Nothing in a schema depends on the strings `PROPOSED_NOT_CANONICAL` or `PROPOSED_UNCONFIRMED`; the schema does not define `owner_vocabulary_status`.

| File | Change |
|---|---|
| `docs/project/maintainers/qualification/enterprise_debt_register/owner_vocabulary.proposed.json` | `status` `PROPOSED_NOT_CANONICAL` becomes `CONFIRMED`; `confirmation` text cites DP-47 option a, addendum 4 (orchestrator's transcription, not an Operator signature), states that no person is assigned |
| `docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl` | `owner_vocabulary_status` `PROPOSED_UNCONFIRMED` becomes `CONFIRMED` on every row (the token is the only change on each line; the field is carried by every row, including the 45 rows without an `owner_ref`) |
| `docs/project/maintainers/qualification/enterprise_debt_register/reconciliation.json` | status line, `owner.vocabulary_status`, new `owner.vocabulary_confirmation_scope` and `owner.rows_by_owner_vocabulary_status`, `owner_status_note` |
| `docs/project/maintainers/qualification/enterprise_debt_register/README.md` | New section 10, dated note citing addendum 4; earlier lines unchanged |

Counts recomputed by script (`.tmp/session/batch-dr/apply_eder.py count`), before and after:

| Measure | Before | After |
|---|---:|---:|
| register rows | 1069 | 1069 |
| rows with `owner_vocabulary_status` `PROPOSED_UNCONFIRMED` | 1069 | 0 |
| rows with `owner_vocabulary_status` `CONFIRMED` | 0 | 1069 |
| vocabulary IDs (all used `owner_ref` / `co_owner_refs` values are in the vocabulary) | 55 | 55 |
| rows with `owner_ref` / without | 1024 / 45 | 1024 / 45 |
| row-level `owner_status`: SINGLE_PROPOSED / OPERATOR_ONLY / SHARED_UNRESOLVED / UNMAPPED / MISSING / ROLE_NOT_OWNER / GENERIC_NOT_AN_OWNER | 817 / 132 / 75 / 14 / 15 / 12 / 4 | 817 / 132 / 75 / 14 / 15 / 12 / 4 |
| rows with `owner_status` `CONFIRMED` | 0 | 0 |
| rows with `accountable_person` | 0 | 0 |
| rows in lifecycle state other than `INGESTED` | 0 | 0 |

Not changed: row-level `owner_status`, `accountable_person`, lifecycle state, severity, disposition, impact class. No row becomes CLOSED, EXCEPTION_ACCEPTED or verified.

Held findings: EQOA-A7-023 (rows without owner, ticket or justification) needs a named owner and ticket per row after the DP-34 classification (DP-48); EQOA-A7-033 (14 `OWNER_MISSING` entries) needs assignments by the Operator. Neither can be handled without a fact that is not published.

## 6. Independent review needed

A non-authoring, read-only reviewer (AR-9-07) confirms on the exact head: (a) no added sentence adds an independence claim, closure, PASS or FRZ promotion, and none calls anything independently verified without a preserved artefact; the words "independent" and "independently" occur in quotations of roadmap or record wording, in paraphrases of what a record says about itself (for example that an independent audit is required or pending), each attributed to the record or the roadmap, in file names of existing dossiers, and in the negated phrases "no independence" and "no ... independence claim"; (b) every cited SHA resolves with `git cat-file -e` and is 40 hex where a full SHA is given: `27af801bec916165dab85019c07cf43d18e18551`, `33490b6aa07fbd6ac9e6d85f9a187be72dd56492`, `a59744517b92847f55def1db22826d17d89ee155`, `234d3c03dce5708766f8e068afbd2ea496132002`, `e46ddcef1c0d8e9ef6d3f23d971af156713e7fb0`, `1dd769331506149a1ee2f411c2bc62e85e5bbf25`, `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca`, `2727a2c2364a97492202b931375772a400279a9b`; (c) the diff of each record file adds lines only and no existing line changes; (d) the quoted roadmap rows read as quoted at `2727a2c23`; (e) `git diff` of `register.jsonl` changes only the `owner_vocabulary_status` token on each line.

## 7. Explicitly out of scope

The roadmap, the freeze checklist, the FRZ matrices, the batches DB-A, DB-B, DB-D, DB-E, DB-F, DB-G, DB-H, DB-I, DB-J, registers generated by scripts, `PDC_*` records, the nine held hunks, findings EQOA-A7-023 and EQOA-A7-033, assignment of any person (`accountable_person`), row-level owner confirmation, triage, severity ratings, every product path, `scripts/`, `tests/`, `.github/`, FRZ promotion, Docker, vendor integration, production configuration.

## 8. Write set

```text
docs/project/maintainers/architecture/INT_CONFIG_REAL_X_EXISTING_CAPABILITY_CONFIGURATION_REALIZATION.md
docs/project/maintainers/architecture/S24_GAP_04_EXTERNAL_CONTRACT_SEMANTIC_COMPATIBILITY.md
docs/project/maintainers/qualification/EBH_4_COMMUNICATION_COMPOSITION_BYPASS_CERTIFICATION.md
docs/project/maintainers/qualification/GOV_X1_GOVERNANCE_AUTHORITY_BOUNDARY_RECERTIFICATION.md
docs/project/maintainers/qualification/TRACE_X_P6_RESTART_RESUME_TERMINAL_CAUSALITY_CERTIFICATION.md
docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_ARCH_FREEZE_HANDOFF.md
docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md
docs/project/maintainers/qualification/enterprise_debt_register/README.md
docs/project/maintainers/qualification/enterprise_debt_register/owner_vocabulary.proposed.json
docs/project/maintainers/qualification/enterprise_debt_register/reconciliation.json
docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl
governance/tickets/T-IGX-046-E.md
governance/tickets/plans/PLAN_T-IGX-046-E.md
```

None is under the IIAC product-path prefixes (`intergrax/`, `agents/`, `applications/`), nor under `scripts/`.

## 9. Test contract

One process at a time, under the shared pytest lock; logs under `.tmp/session/batch-dr/`:

- `uv run pytest tests/unit/docs/` (without `-p no:cacheprovider`)
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Data check (stdlib, no repo script exists for the register): every line of `register.jsonl` parses as JSON, all `owner_ref` and `co_owner_refs` values are in the vocabulary, counts equal the table in section 5.
- Added-link check: every relative link target added by this change exists.
- `python -I scripts/governance/verify_ticket_approvals.py --since origin/development`.
- Reviewer check: `git diff --name-only origin/development...HEAD` lists exactly the Write set.

## 10. Risks

- Five of the seven note files and the register data lie under the EBH-7 identity allowlist prefix; the two architecture documents lie outside it, so a merge after the EBH-7 pin affects the identity gate for those two. The EBH-7 and PDC owners confirm timing.
- `register.jsonl` is large (1069 lines); the diff is one token per line.
- The decision records DP-04, DP-09, DP-45, DP-47 are the orchestrator's transcriptions of chat decisions, not Operator signatures; the notes and the vocabulary file say so.
- The vocabulary keeps the file name `.proposed.json` while its status reads `CONFIRMED`; the README note explains why.

## 11. Rollback

Revert the single PR. No code path, configuration or generated file is affected.

## 12. Closure condition

PR merged after independent review; the ticket records the merge SHA. No finding is marked closed or verified by this ticket.
