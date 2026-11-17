# PLAN T-IGX-046-F - EQO-A remediation: release of the nine held DB-C hunks as dated gap and custody notes

This ticket certifies nothing, promotes no FRZ criterion, changes no stage status and edits no product path. Every added sentence either quotes wording that exists in a published record at the head named below, states a fact that was re-read at that head, or states what a file does not contain. None adds an acceptance stamp, a closure, PASS, verification or independence claim.

## 1. Purpose

Ticket T-IGX-046-E (PR #656, merged as `cb60f683f409baf2b87c27e31902a14d918e1e0b`) applied 7 of the 16 prepared DB-C notes and held 9, because each needed a fact that no tracked file stated (its plan section 3). The Operator decided on 2026-11-17 how to release them (four decisions given in chat; the chat is the original, and addendum 6 of `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md`, merged in PR #660 as `cd93d7e2b519bc2d894e2da82f4f8156d281af2d`, is the orchestrator's English rendering of that message and of how the orchestrator applies it):

1. H04, H05, H09: publish the local replay and regression logs under `iiac_evidence/` after a secret and personal-data scan, with the sha256 in the note.
2. H09: apply the published decision DP-44 on G-SKILL together with SKILL-X (SKILL-X artefact published, G-SKILL a scoped limitation, reconciled with roadmap L499, no independence implied), quoting only the published wording.
3. H11: a gap note stating that the SHA ladder does not show a designation covering the W5 signing session `ent-sign-w5-od-08-2026-11-11`; no extension of `ent-sign-w3-2026-11-09` by inference.
4. H03, H06, H07, H08, H12: dated gap notes that state only what is verifiable in the file, name the related finding and point to the register row (EDER) and its proposed owner, and add no claim.

Findings addressed: EQOA-A2-007, EQOA-A2-036, EQOA-A2-040, EQOA-A2-044, EQOA-A2-046, EQOA-A3-017 (findings baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589). Re-verified at `origin/development` `44d1e5d105beb9b62d0a12e693066b0b26c125eb`.

## 2. Scope: 9 notes in 9 stage records (98 added lines, 0 changed or deleted lines) and one evidence directory

| Hunk | File (under `docs/project/maintainers/qualification/`) | Kind | What the note states | Source |
|---|---|---|---|---|
| H03 | `AW_7C_A2_SCOPED_ADAPTIVE_INTEGRATION_QUALIFICATION.md` | gap | Record L930-L941 classifies AW-7C-P2, -P3, -P4, -CERT as `CURRENT-HEAD REVALIDATED` without SHAs; it does not state whether they are folded into the final parent certification `2fab207f3e850f1b339523587a19906ea1e9b292`; roadmap has no ledger row for them; `EDER-F-0037` | the file; roadmap L386, L1983-L1989 |
| H04 | `ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W4_PARENT_RECERTIFICATION.md` | custody (log not published) | L80, L94-L95 cite T11-A/T11-B logs under gitignored `.tmp/`; local files found, not published because the scan hit user-profile paths; sizes and sha256 of the local files; `EDER-F-0073` | the file; local files |
| H05 | `ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_PARENT_RECERTIFICATION.md` | custody (logs published) | L119 cites a directory only; two logs published with sha256; record names no auditor, report or audit SHA beyond L3 and L5; `EDER-F-0073` | the file; published logs |
| H06 | `ENTERPRISE_RUNTIME_INTELLIGENCE_W6_PARENT_RECERTIFICATION.md` | gap | Record has no run counts, command, run SHA or log reference; roadmap L1965 quotes 10/10 and 52/52; `EDER-F-0073` | the file; roadmap L1965 |
| H07 | `HARNESS_W7_HOST_01_CURRENT_HEAD_RECERTIFICATION.md` | gap | L119 reports 80 passed, 7 failed; node IDs and classification not listed; `EDER-F-0075` | the file |
| H08 | `HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md` | gap | L152, L176, L192 name the npsc42 failure as tracked debt; no resolution or acceptance stated; HARNESS-FINAL record has no occurrence; `EDER-F-0075` | the file; SCHED-01 record L166 |
| H09 | `pre_ent_enterprise_certification/PRE_ENT_WAVE6_CROSS_AUDIT_REPORT.md` | custody (log published) + decision cross-reference | log published with sha256; DP-44 quoted verbatim from `eqo_a_remediation/OPERATOR_DECISIONS_EQOA_REMEDIATION.md`; AUD-SKILLX location; roadmap L499 wording; `EDER-F-0069` | the file; published logs and records |
| H11 | `ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md` | gap | Ladder names no signing session or designation; delegation record names `ent-sign-w3-2026-11-09` as designated session id (L38); cited records listed; no inference; `EDER-F-0065` | the file; three published records |
| H12 | `HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md` | gap | Row U transport parity (L76), CE-02 (L200), FRZ-HRN-03..08 rows (L186-L193) carry no closure record, owner pointer, SHA or node IDs; `EDER-F-0097` | the file; CTRL-X record |

Every note is appended after the last line of its record, preceded by a blank line and a rule, dated 2026-11-17 and labelled "EQO-A remediation". Each says what it does not establish. Line endings: the SHA ladder is CRLF with a byte-order mark and `PRE_ENT_WAVE6_CROSS_AUDIT_REPORT.md` is CRLF; the notes appended to them use CRLF and the byte-order mark is kept; the other seven records are LF.

`git diff --numstat` of the nine records shows 0 deleted lines. The ladder parser `parse_sha_ladder_table` (`intergrax/platform/enterprise_audit/baseline.py`) returns an identical result (4 entries) for the ladder before and after the change.

## 3. Published logs (new directory `iiac_evidence/EQO-A-DBC-HELD/`)

Layout follows `iiac_evidence/EQO-A-PT-D03/EVIDENCE_MANIFEST.md`: one directory per publication, a manifest with custody statement, a `logs/` subtree. `.gitattributes` line 1 sets `-text` for `iiac_evidence/**`, so the byte copies are not normalised.

| Published path (under `iiac_evidence/EQO-A-DBC-HELD/`) | Size | sha256 | Local origin |
|---|---:|---|---|
| `logs/harness-w5-r1/v11-batch-a.log` | 2642 | `464e96b23dc216efbeefd280e27dbe3f9dec7e3b23468097e26b92de7720ea1d` | `.tmp/session/harness-w5-r1/v11-batch-a.log` |
| `logs/harness-w5-r1/v12-batch-b.log` | 2642 | `0cc6b83c3e1621837fa7c825f61df72f9943a68959222e40b3173c9fec58fec5` | `.tmp/session/harness-w5-r1/v12-batch-b.log` |
| `logs/pre-ent-w6-cross-audit/parent_regression_5gate.log` | 1646 | `7c4ab15cff100f393a73dc5a123af89700eca8ed79fd41a033ff1c4454426f6d` | `D:\Projekty\_wt\intergrax\_scratch\w6-cross-audit\parent_regression_5gate.log` |

Not published:

| Local file | Size | sha256 of the local file | Reason |
|---|---:|---|---|
| `.tmp/session/HARNESS-W4-FINAL-CONSOLIDATION/t11-a.log` | 439232 | `1c4934fa5aca892547c2a95aab489b7973a7eb0ce54700920657cfd79fb892fe` | scan hit: Windows user-profile path in 200 lines |
| `.tmp/session/HARNESS-W4-FINAL-CONSOLIDATION/t11-b.log` | 439232 | `ad3ffb562bc7c7ea2901363cefb26158ccc74bac6cd941fd996f2e1db990adc8` | same scan hit (200 lines) |

Both held files are 0.42 MiB, below the 2 MB limit, so the size rule does not apply; they are held back only because of the scan hit, and redaction is not allowed because a modified log is no longer the artefact. All five files are UTF-16 little-endian console captures; the scan was run on the decoded text (a first pass on the raw bytes was discarded because it could not see ASCII patterns in UTF-16 data). Scan patterns: key and token words, private-key headers, cloud-key prefixes, e-mail addresses, IPv4 addresses, `Users` and `AppData` paths, personal-name patterns, long hex strings and URLs. The two published W5 logs contain checkout paths of the form `D:\Projekty\intergrax\...` (no user name); the reviewer may overrule the reading that a checkout path is not personal data, in which case those two logs are withdrawn and H05 becomes a gap note.

Evidence manifest: `iiac_evidence/EQO-A-DBC-HELD/EVIDENCE_MANIFEST.md`. After the commit the blob stored in git for each published path is read back and its sha256 compared with the table above; the result is recorded in the pull request body.

## 4. Checks and adjustments

- Every line number cited in a note was checked against the file at the head named above by a script that asserts the cited substring on the cited line, and every "no occurrence" statement was re-run (searches over the record, the roadmap, the checklist and the CTRL-X record). All ten distinct 40-hex values in the added text resolve with `git cat-file -e`.
- The prepared hunk H03 would have said that the record "does not state" the status of AW-7C-P2..CERT; the re-check showed that the record classifies them at L938-L941 and that the roadmap names `AW-7C-P2` once (L1985); the note states those facts instead.
- The prepared hunk H11 would have said that no W5 designation is shown anywhere; the re-check found the delegation record scope lines (`ENT_AUDIT_X_O1_SIGNING_DELEGATION_RECORD.md` L32, L50) and the W5 signing-session record; the note cites them as they read and concludes nothing.
- The G-SKILL wording is the published decision DP-44 (decision session 4 of `OPERATOR_DECISIONS_EQOA_REMEDIATION.md`), quoted verbatim; that record is the orchestrator's transcription of chat decisions and is not an Operator signature, which the note says. Wording of the unpublished decision plan (LBL-C label text, WP-3, WP-9) is not used.
- The register rows (`EDER-F-0037`, `-0065`, `-0069`, `-0073`, `-0075`, `-0097`) are cited with their `owner_ref` and `owner_status` as the register reads at the head named above; the notes say that the vocabulary is confirmed, the row-level owner is not, and no person is assigned.
- Open pull requests at the time of writing (#653, #658, #659) touch none of the nine records, the new directory or these ticket files.
- EBH-7 identity allowlist (`scripts/qualification/verify_ebh7_p03_immutable_evidence.py`, prefix `docs/project/maintainers/qualification/`): all nine records and the new `iiac_evidence/EQO-A-DBC-HELD/` tree lie under that prefix. The sha256 of each of the nine files before the change was searched over the whole tracked tree: no match, so no digest of any of them is pinned. Name references outside `docs/`: `tests/qualification/roadmap_replay_x/_historical_rows.py` quotes roadmap ledger rows that link seven of the records by file name (replay data, no digest of a record); `scripts/qualification/pdc_x_requirements.py` and `pdc_x_wbs.py` name `HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md`; the PRE-ENT report is named by no script or test. `ENT_AUDIT_X_SHA_LADDER.md` is parsed by `intergrax/platform/enterprise_audit/baseline.py` and read by `scripts/enterprise_audit/` and `scripts/qualification/ent_audit_x_formal_closure_preflight.py` for table rows beginning with `|`; the appended prose has no such row. The reviewer re-runs these searches.
- `iiac_evidence/` paths are validated by name in `scripts/governance/iiac_activation_gate.py` (it accepts a named artefact under the directory) and by named stage directories in the preflight scripts (`evol_x_closure_publication_preflight.py` walks `iiac_evidence/EVOL-X` only); no script was found that enumerates the whole `iiac_evidence/` tree for unlisted files. The earlier PT-D03 publication added 29 files in the same way.

## 5. Independent review needed

A non-authoring, read-only reviewer (AR-9-07) confirms on the exact head: (a) no added sentence adds an independence claim, closure, PASS, verification or FRZ promotion, and none calls anything independently verified without a preserved artefact; the words "independent" and "independently" occur in quotations of roadmap or record wording, in paraphrases of what a record says about itself, in the quotation of DP-44, in file and record names, and in negated phrases; (b) every full SHA cited resolves with `git cat-file -e` and is 40 hex: `44d1e5d105beb9b62d0a12e693066b0b26c125eb`, `2fab207f3e850f1b339523587a19906ea1e9b292`, `6fda62ca792255e482895a574250af547cb18cba`, `e8b51e2a1a80eec3b62d64a179ec5e9e0856c04a`, `f38e455ccc391dad9a6a64d7867a33cfc5c9b21d`, `5a87046ba6b5dc46ffb3693417b4d2bf96a561c5`, `846af28cd8fb6889cece1e8807196014dc89cdfd`, `5d5442dae7769671a7532a973428c8203b492ec6`, `3a9b58e6fa1e14ddc9786bd9a447a1d76552d196`, `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, and in section 1 `cb60f683f409baf2b87c27e31902a14d918e1e0b`; (c) the diff of each record adds lines only; (d) every cited line number and every quoted phrase reads as cited at the head; (e) the sha256 values in the notes and the manifest equal `sha256sum` of the published files and of the git blobs, and the held files are not in the diff; (f) the scan result of the three published logs is reproduced on the decoded text; (g) the notes do not state that a held fact exists elsewhere; the H04 note and the manifest record local files, digests and last lines that no file in the repository can confirm, and they say so explicitly (the T11 logs are not preserved in the repository). This is a stated limitation for the reviewer; the digests are kept so that anyone holding the files can compare.

## 6. Explicitly out of scope

The roadmap, the freeze checklist, the FRZ matrices, `PDC_*` records, generated registers, the register data (`register.jsonl`, `reconciliation.json`), the matrix `PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md` (its G-SKILL row stays as it is), `scripts/`, `tests/`, `.github/`, every product path, publication of the two T11 logs, redaction of any log, supply of any missing owner fact, assignment of any person (`accountable_person`), row-level owner confirmation, FRZ promotion, Docker, vendor integration, production configuration.

## 7. Write set

```text
docs/project/maintainers/qualification/AW_7C_A2_SCOPED_ADAPTIVE_INTEGRATION_QUALIFICATION.md
docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W4_PARENT_RECERTIFICATION.md
docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_PARENT_RECERTIFICATION.md
docs/project/maintainers/qualification/ENTERPRISE_RUNTIME_INTELLIGENCE_W6_PARENT_RECERTIFICATION.md
docs/project/maintainers/qualification/HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md
docs/project/maintainers/qualification/HARNESS_W7_HOST_01_CURRENT_HEAD_RECERTIFICATION.md
docs/project/maintainers/qualification/HARNESS_W8_BG_01_CURRENT_HEAD_RECERTIFICATION.md
docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md
docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_WAVE6_CROSS_AUDIT_REPORT.md
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/EVIDENCE_MANIFEST.md
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/harness-w5-r1/v11-batch-a.log
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/harness-w5-r1/v12-batch-b.log
docs/project/maintainers/qualification/iiac_evidence/EQO-A-DBC-HELD/logs/pre-ent-w6-cross-audit/parent_regression_5gate.log
governance/tickets/T-IGX-046-F.md
governance/tickets/plans/PLAN_T-IGX-046-F.md
```

None is under the IIAC product-path prefixes (`intergrax/`, `agents/`, `applications/`), nor under `scripts/`, `tests/` or `.github/`.

## 8. Test contract

One process at a time, under the shared pytest lock; logs under `.tmp/session/dbc-held/`:

- `uv run pytest tests/unit/docs/`
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `python -m scripts.governance.verify_ticket_approvals --since origin/development` (the module form; `python -I` cannot import the `scripts` package)
- Reviewer check: `git diff --name-only origin/development...HEAD` lists exactly the write set; `git diff --numstat` shows 0 deleted lines for the nine records.

## 9. Risks

- All nine records and the new directory lie under the EBH-7 identity allowlist prefix, so a merge after the EBH-7 pin does not affect the identity gate for them; the EBH-7 and PDC owners confirm timing of the merge relative to the pin.
- The published W5 logs contain checkout paths; the reading that this is not personal data is the author's and is flagged for the reviewer (section 3).
- The two held T11 logs remain only on the local workstation; their digests recorded in the note and the manifest cannot be checked from the repository.
- The decision record DP-44 is the orchestrator's transcription of a chat decision, not an Operator signature; the H09 note says so.
- The four decisions of 2026-11-17 are transcribed in addendum 6 of `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md` (merged in PR #660 as `cd93d7e2b519bc2d894e2da82f4f8156d281af2d`), which is the orchestrator's English rendering, not an Operator signature, and which states that the Polish original is in the chat. The notes do not attribute any wording to these decisions; the only decision wording they quote is DP-44.

## 10. Rollback

Revert the single pull request. No code path, configuration or generated file is affected.

## 11. Closure condition

Merge of the pull request after independent review; the ticket records the merge SHA. No finding is marked closed or verified by this ticket; the register rows cited by the notes keep their status.
