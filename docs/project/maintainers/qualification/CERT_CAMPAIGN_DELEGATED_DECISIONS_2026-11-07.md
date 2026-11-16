# Certification campaign - delegated decision records (2026-11-07)

**Scope of this document:** publication of the signed decision records of SIGN-1 (QUAL-X), SIGN-2 (SKILL-X) and SIGN-5 (route / SSOT decisions), copied verbatim from the signing-session dossiers. Pointer table for SIGN-3 and SIGN-4 in section 2.

## 0. Status of these records (read first)

- **Delegated.** Every `Approved ...: operator-delegate @ ...` line below was written by a designated signing session under IIAC Amendment 01, citing the delegation record `.tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md` (sha256 `5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab`). The delegation record itself is a gitignored session artifact and is not published here.
- **`process_enforced`.** The assurance is process-level only.
- **Not recognised by IIAC / EVOL-X preflight.** No IIAC verifier, activation gate or `evol_x_closure_publication_preflight` recognises the `operator-delegate` form; nothing in this document is an input to those tools.
- **Not a cryptographic signature.** These records must not be presented as one.
- **Scope limits stand as signed.** No FRZ-* status is promoted by this document; no stage parent is closed by it; the signed conditions (in particular the SKILL-X condition on the QUAL-X regularisation and the DS-5 effectivity condition) remain in force exactly as written below.
- The source files include the signing sessions' evidence notes and "proposed publication text"; they are preserved unedited. Only the text under "Exact text signed" / "Text signed" and the signature lines are the signed content; the surrounding verification notes are the signer's rationale.

## 1. Decision index (SIGN-1, SIGN-2, SIGN-5)

| Decision | Session | Outcome | Signed at | Roadmap / SSOT effect (published by the integration branch) |
|---|---|---|---|---|
| OD-QUALX-ACCEPT-2026-11-07 | SIGN-1 | SIGNED | 2026-11-07T18:20:51+01:00 | QUAL-X row and ledger: Population A independently accepted (scoped) |
| OD-QUALXP0-VERIFIED-2026-11-07 | SIGN-1 | SIGNED (separate verification, not absorbed) | 2026-11-07T18:20:51+01:00 | QUAL-X-P0 row and ledger |
| OD-SKILLX-ACCEPT-2026-11-07 | SIGN-2 | SIGNED_WITH_CONDITIONS | 2026-11-07T18:18:48+01:00 | SKILL-X row (conditional text intact) |
| OD-ROUTE-DS0..DS4-2026-11-07 | SIGN-5 | SIGNED | 2026-11-07T18:24:31+01:00 | route decisions; no roadmap edit |
| OD-ROUTE-DS5-2026-11-07 | SIGN-5 | SIGNED_WITH_CONDITIONS | 2026-11-07T18:24:31+01:00 | predecessor deviation regularisation, conditional |
| OD-SSOT-D1-2026-11-07 | SIGN-5 | SIGNED | 2026-11-07T18:24:31+01:00 | SSOT patch 07 (eight TRACE-X rows, lineage only) |
| OD-SSOT-D2-2026-11-07 | SIGN-5 | SIGNED (exact-SHA audits; absorption not adopted) | 2026-11-07T18:24:32+01:00 | SSOT patch 10 NOT applied |
| OD-SSOT-D3-2026-11-07 | SIGN-5 | SIGNED (operator-scoped acceptance only) | 2026-11-07T18:24:32+01:00 | SSOT patch 09 |

## 2. Pointer table - decisions bound into their own tickets (not reproduced here)

| Signing session | Decisions | Outcomes | Bound to | Source dossier file (sha256) |
|---|---|---|---|---|
| SIGN-3_CTENT | OD-CTENT01-PLAN-2026-11-07, OD-CTENT02-PLAN-2026-11-07 | SIGNED_WITH_CONDITIONS (both) | plan-approval lines of tickets `T-IGX-022-A`, `T-IGX-022-B` | `.tmp/session/cert-campaign/SIGNING/SIGN-3_CTENT/SIGNED_DECISIONS.md` (`6ce86682eaad9dd1c1a304da4ec419d633a5f5696ce7c0f9e1712e1853b19650`) |
| SIGN-4_TLX | OD-TLX-A, OD-TLX-B (SIGNED); OD-TLX-C, OD-TLX-D (SIGNED_WITH_CONDITIONS); OD-TLXCORR-A-PLAN, OD-TLXCORR-B-PLAN (SIGNED); OD-TLXCORR-C-PLAN (REFUSED, no signature written) | as listed | plan-approval lines of tickets `T-IGX-023-A`, `T-IGX-023-B` | `.tmp/session/cert-campaign/SIGNING/SIGN-4_TLX/SIGNED_DECISIONS.md` (`672518014018fb4ab88eaf22d4c87b47abbf7478423418ca26448bc4cd38c32c`) |

The authoritative approval lines for these live in their tickets under `governance/tickets/`; this document does not restate them.


## 3. SIGN-1 (QUAL-X) - SIGNED_DECISIONS.md, verbatim

Source (not tracked; session dossier): `.tmp/session/cert-campaign/SIGNING/SIGN-1_QUALX/SIGNED_DECISIONS.md` - sha256 `5acbb740f953ff34c6fa684db6cdff7ef9ce0edd7c2cc89cff0695e3fa399931`. The block below is the file content byte-for-byte, unedited (fence is tildes because the source contains backtick fences).

~~~~~~markdown
# SIGNED_DECISIONS - SIGN-1_QUALX (agent sign_qualx, session ec974642-7d45-42d7-a69c-28374140189a)

Signed in the operator's name as designated signing session (IIAC Amendment 01); delegation record sha256 `5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab` verified before signing. Only the two listed decisions are signed. Signing time 2026-11-07T18:20:51+01:00.
Worktree HEAD `b288ba756ef736f3d372832d8e684dbb19901ca4` (clean); campaign base `2f6664d53e3fcd1cb6010e33506e2d22201dc9f3` is an ancestor of HEAD.

## Decision 1 - OD-QUALX-ACCEPT-2026-11-07 - SIGNED

Exact text signed (file `texts/OD-QUALX-ACCEPT-2026-11-07.txt`, sha256 `16861352602eefd242b143b19a0bee15d083f8f9f27b7dae8a9c4299ea704f47`):

```text
OD-QUALX-ACCEPT-2026-11-07
Independent acceptance of the QUAL-X formal closure "QUAL-X CLOSED - POPULATION A ONLY" (formal closure commit 0d52df6e0da10bb1ec36bcb113b54a4866b9b76f; evidence authority ACCEPTANCE_SHA d9ea0d6e87b939e4cefb9971c6bbd4fbcaecc0ce), limited to Population A.
Basis: AUD-QUALX AUDIT_REPORT.md (verdict PASS_SCOPED, sha256 3661a2cd4281b6c89407afcf703e35df1b232ae804c6a49d29a0f994df4f3a4f), FINDINGS.json F-01..F-07 (sha256 89687115902c6d3cd2f10df49152f2286d7f711bde65849b67345163b2678dbb), exact-SHA replay at d9ea0d6e8: 630 passed, 0 failed, 0 skipped (log sha256 b566d6b6ce8a8833ebbebe34ab8c414124efe9f46ade5e472405ffbec8227725), re-verified structurally by the signing session.
NOT accepted by this decision: global FRZ-REG-01..10 PASS, PROD-Q Dimension B, Population E / ENT-AUDIT-X, ARCH-FREEZE, any FRZ promotion, any product/test state after d9ea0d6e8, and the operator's original Path B decision message (A1, recorded NOT_REPRODUCED by the auditor). FRZ-REG-01..10 remain OPEN.
Also recorded: (1) the self-adjudicated A1-A5 table in the closure commit is superseded by this audit (F-01); (2) the stale in-document status text and the "32 modules" count (F-04, F-06) are corrected in a later doc revision (deferred, not performed by this decision); (3) QUAL-X-P0 is NOT absorbed by this decision (see OD-QUALXP0-VERIFIED-2026-11-07).
```

Evidence I verified myself (git at exact SHAs unless noted):
- d9ea0d6e8, 4c5faa89a, 0d52df6e0, 9a175177 resolve as commits; all are ancestors of base and of HEAD. Chain d9ea -> 4c5f -> 0d52 (`0d52^` = 4c5f, `4c5f^` = d9ea): 4c5f is the parent of 0d52 (F-07 confirmed).
- `git diff --name-status d9ea..0d52` = 3 `.md` files (roadmap, FRZ checklist, QUAL-X inventory); d9ea..4c5f = 2 `.md`; the non-`.md` diff d9ea..0d52 is empty and no `tests/` or `intergrax/` path changed. The tested tree equals the closure tree.
- 9a175177..d9ea changed 7 product files under `intergrax/runtime/` (events/store.py, execution/debug_lab_nexus_loop.py, execution/host_observability_composition.py, nexus/nexus_loop.py, nexus/observability_wiring.py, nexus/orchestration/task_events.py, observability/journal_export.py): the F-03 gap is real, the 535/535 attestation did not cover d9ea, and the 630 replay is the operative evidence. uv.lock/pyproject unchanged 9a17..d9ea; the CRLF-form sha256 of the d9ea uv.lock = `0ab9b58d92cb...a7b6` (matches the attested value; the LF form `fffcc162...` explains the F-04 "hash form" difference).
- All 41 targets in `AUD-QUALX/logs/targets.txt` exist at d9ea0d6e8; the P5 manifest SSOT `tests/qualification/qual_x/_p5_freeze_selection_catalog.py` with `qual_x_p5_dimension_a_pytest_module_targets()` exists there.
- `pytest-d9ea.log` (sha256 b566d6b6...): "collected 630 items", rootdir = scratch worktree at d9ea, last line "630 passed in 1080.39s"; no skip/fail/warning/error lines (grep count 0). Internally consistent. I did not rerun the 18-minute suite: the claim could be checked structurally, and the process-budget rule says to rerun only when it cannot.
- A4: `git grep RuntimeEventPersistence d9ea0d6e8 -- intergrax/runtime/execution` = no match; the gate test name exists in the d9ea tree.
- FRZ-REG-01..10: all ten rows OPEN at d9ea0d6e8 and at HEAD. No promotion.
- Roadmap at HEAD: L480 QUAL-X `[x] CLOSED - POPULATION A ONLY (formal closure publication READY FOR INDEPENDENT AUDIT on GitHub)`; L2057 says the same; L479 QUAL-X-P0 `[ ] READY FOR AUDIT`; L2055 "Independent audit: pending on P0 commit SHA". One git identity (jakbuczarnecki) authored every QUAL-X commit a70b..base that I listed (30), consistent with F-01.
- Not reproducible by me, and stated as limits in the signed text: the operator's original Path B message (A1); the raw 9a175177 logs; Dimension B and Populations B-E.
- The product tree changed after d9ea (214 files under intergrax/agents/applications d9ea..HEAD); the acceptance is bound to d9ea0d6e8 and is not carried forward.

Outcome: SIGNED. Rationale: F-01 (self-adjudication) and F-03 (evidence gap) are closed in substance by the exact-SHA replay; the remaining findings are LOW/INFO documentation items and none is blocking.

Signature lines (copy verbatim):

```text
Approved OD-QUALX-ACCEPT-2026-11-07: operator-delegate @ 2026-11-07T18:20:51+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

## Decision 2 - OD-QUALXP0-VERIFIED-2026-11-07 - SIGNED

Exact text signed (file `texts/OD-QUALXP0-VERIFIED-2026-11-07.txt`, sha256 `4ca3aa913899c63856425803ffdc48b199d57d1025d5eeb77ca0d49a12a38210`):

```text
OD-QUALXP0-VERIFIED-2026-11-07
QUAL-X-P0 (enterprise qualification baseline & gate inventory + Path B stage-row reconciliation; commit a70b68cf947b77de58bce8df0863d3114991cbd9, parent 497af3cdbc9d8102b95975fda362a91e25ff2b29) is accepted as INDEPENDENTLY VERIFIED SEPARATELY (not absorbed into OD-QUALX-ACCEPT-2026-11-07).
Scope: exactly as in AUD-QUALXP0 AUDITOR_PASS_RECORD_DRAFT.md (sha256 cf213d0339a8a177896eccead3490823cf21e2405c0682344922948db20d807a), verdict PASS_SCOPED: inventory content / SHAs / counts, Path B text reconciliation, non-closure and no-FRZ-promotion claims; bounded replay 202 passed / 3 failed / 2 skipped at a70b68cf9 plus 2 scripts (AUDIT_REPORT.md sha256 149647a7ea9b719ee17d17c2f9797d8aeffb25f1d2cb9df07c68b0cd4f40a574; FINDINGS.json sha256 4a2b46a87c1426271e463e3091fa4f39e361f9b09fa00e6321f2a6eaa126ab18).
Erratum accepted: F-00 (three Q01/Q03-cited gates were RED on the P0 tree: test_ebh_2a_public_contract_boundary_gate, test_ebh_2i_final_rescan_gate, test_hardening_3_layer_boundary_gate; the P0 "DIRECT" labels for Q01/Q03 are NOT accepted as true at the P0 tree; remediated by later commits, green at d9ea0d6e8); F-02 (scripts/maintenance/check_architecture_boundary_chaos.py cited in Q01 is non-functional at P0 and at base); F-03 (Q03 path is tests/unit/runtime/architecture/test_ebh_2f_r2_integration_plugin_contract_typing_gate.py).
Acknowledged: F-01 (P0's own gate "independent audit before P1" was bypassed; not retroactively repairable).
NOT covered: any FRZ-REG-01..10 promotion, QUAL-X beyond Population A, PROD-Q Dimension B, repair of the chaos script, Q05 test_npsc*, AW-7C PHYSQ, Q09/Q10 historical batches and the bulk of the governance/tenant_x/state_x/trace_x trees.
```

Evidence I verified myself:
- a70b68cf9 is a commit with parent 497af3cdb and is an ancestor of base. `git show --stat`: 2 files only (roadmap +14/-6, inventory +179), production delta 0, author jakbuczarnecki, 2026-10-22 15:30:23.
- F-00: at a70b68cf9, `intergrax/contracts/governed_external_work_create_follow_on.py:29` and `intergrax/contracts/governed_external_work_provider_invocation_identity.py:7` import `intergrax.runtime.attestation.canonical_json`, and `intergrax/contracts/tools/qualified_tool_invocation.py:17` imports `intergrax.tools.invocation_wiring`. At d9ea0d6e8 none of these imports is present (grep empty). `pytest-p0.log` (sha256 9e0a0a7a...) ends "3 failed, 202 passed, 2 skipped"; the three FAILED node ids are the ones named in F-00; the 2 skips are platform-conditional.
- F-02: `scripts/maintenance/check_architecture_boundary_chaos.py` joins `REPO_ROOT / "scripts" / <child>`; `scripts/check_intergrax_no_applications_imports.py` does not exist (only `scripts/maintenance/...`); the file is unchanged a70b..base and base..HEAD; the P0 chaos log shows "can't open file ...scripts\check_intergrax_no_applications_imports.py".
- F-01: the audit record states P1 started 10 minutes after P0 and a single git identity; I found no waiver/absorption text. The process gate cannot be repaired retroactively, so it is acknowledged and not waived.
- Overlap with AUD-QUALX: that audit did not audit P0 prose/matrix, so "verified separately, not absorbed" is the correct choice. The auditor advised against absorption, and I agree because P0's own baseline gates were red at its tree.

Outcome: SIGNED (separate verification, scoped, with erratum). Not absorbed.

Signature lines (copy verbatim):

```text
Approved OD-QUALXP0-VERIFIED-2026-11-07: operator-delegate @ 2026-11-07T18:20:51+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

## Proposed publication text (PROPOSAL ONLY - not applied)

File `docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` (line numbers as of HEAD b288ba756). Each OLD string occurs exactly once in the file and on the stated line. An authorized publisher applies these after review.

1. L479 (QUAL-X-P0 status cell)
   - OLD: `[ ] **READY FOR AUDIT**`
   - NEW: `[x] **INDEPENDENTLY VERIFIED - SCOPED** (not a stage closure; PASS_SCOPED AUD-QUALXP0 @ a70b68cf947b77de58bce8df0863d3114991cbd9; accepted as separate verification by OD-QUALXP0-VERIFIED-2026-11-07, operator-delegate; erratum F-00 / F-02 / F-03, F-01 acknowledged; no FRZ promotion)`
2. L480 (QUAL-X status cell)
   - OLD: `(formal closure publication **READY FOR INDEPENDENT AUDIT** on GitHub)`
   - NEW: `(formal closure **INDEPENDENTLY ACCEPTED - POPULATION A ONLY** by OD-QUALX-ACCEPT-2026-11-07, operator-delegate, on AUD-QUALX PASS_SCOPED @ ACCEPTANCE_SHA d9ea0d6e87b939e4cefb9971c6bbd4fbcaecc0ce; global **FRZ-REG-01..10**, **PROD-Q** Dimension **B**, Population **E** / **ENT-AUDIT-X** remain **OPEN**)`
3. L2055 (QUAL-X-P0 ledger row)
   - OLD: `**Independent audit:** pending on P0 commit SHA.`
   - NEW: `**Independent audit:** AUD-QUALXP0 PASS_SCOPED on a70b68cf947b77de58bce8df0863d3114991cbd9, accepted as separate verification (not absorbed) by OD-QUALXP0-VERIFIED-2026-11-07 (operator-delegate); erratum F-00 (Q01/Q03 gates red at the P0 tree, remediated later), F-02 (chaos script non-functional), F-03 (Q03 path); F-01 (audit-before-P1 gate bypassed) acknowledged.`
4. L2057 (QUAL-X ledger row)
   - OLD: `**Independent audit:** formal closure commit **READY FOR INDEPENDENT AUDIT** on GitHub.`
   - NEW: `**Independent audit:** AUD-QUALX PASS_SCOPED @ d9ea0d6e87b939e4cefb9971c6bbd4fbcaecc0ce (630 passed); accepted for Population A only by OD-QUALX-ACCEPT-2026-11-07 (operator-delegate); the self-adjudicated A1-A5 table is superseded by that audit (F-01).`
   - The later sentence on that row ("SKILL-X: predecessor satisfied on publication - not entered until independent acceptance of closure commit") is historical; reconcile it only per the SKILL-X acceptance record, not by these decisions.
5. Inventory doc `docs/project/maintainers/qualification/QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md`: add under the P0 header: `Independent verification: AUD-QUALXP0 PASS_SCOPED (see roadmap L479). Erratum: Q01/Q03 gates red at a70b68cf9 (F-00); chaos script non-functional (F-02); Q03 path is tests/unit/runtime/architecture/test_ebh_2f_r2_integration_plugin_contract_typing_gate.py (F-03).` Deferred doc fixes F-04 ("32 modules" -> 35 per the P5 manifest at d9ea0d6e8) and F-06 (stale "READY FOR AUDIT" body text) go in the same revision.
6. Do NOT change any FRZ-REG-01..10 status and do NOT mark any stage CLOSED through this publication.
~~~~~~

## 4. SIGN-2 (SKILL-X) - SIGNED_DECISIONS.md, verbatim

Source (not tracked; session dossier): `.tmp/session/cert-campaign/SIGNING/SIGN-2_SKILLX/SIGNED_DECISIONS.md` - sha256 `1b1f0d518e677385b23634e96c2741eee508d7760097fe58ffe2d309e51be204`. The block below is the file content byte-for-byte, unedited (fence is tildes because the source contains backtick fences).

~~~~~~markdown
# SIGNED_DECISIONS - SIGN-2_SKILLX (session 7eaa7ae1-e9e6-4d9c-a4fd-b1c214bf1a55)

## Decision OD-SKILLX-ACCEPT-2026-11-07 - outcome: SIGNED_WITH_CONDITIONS

**Exact text signed:** file `SIGNED_TEXT_OD-SKILLX-ACCEPT-2026-11-07.txt` in this dossier (verbatim, LF, bytes as stored).
**sha256 of that text:** `5b6ecb949a1d0dae54355986c4321ba6ddd6c44fdad71a792fb1fa297c694bf5`
The parent sentence is the verbatim text under "What the Operator would sign" in AUD-SKILLX/OPERATOR_SIGNATURE_PACKAGE.md. The four sub-decisions and the QUAL-X condition are appended in the same text.

### Sub-decisions (inside the same record)
| Item | Choice | Why |
|---|---|---|
| OD-SKILLX-A | ACCEPT (recommended option) | Roadmap 3.0.4 says "no prescribed package technology" for supply-chain/content (EC9), so catalog-witness without signing meets the wording. It mandates no full GOV-X2 replay (EC2). Negative tests for plan ceiling, forged metadata and HITL pass in the replay. |
| OD-SKILLX-B | ACCEPT (recommended option) | I found no SKILL-X review artefact: `git ls-tree` at 76a98cb has only the Batch D/E evidence and the reconciliation doc, and `git grep` of governance/ and docs/audit_results/ for "skill-x" has no hits. The only description of the review is the implementer-authored section 0 of the reconciliation doc. So AUD-SKILLX serves as the re-acceptance. |
| F-03 | ACCEPT as record debt (recommended option) | SKILL-X code and tests are unchanged since e00a0aab (git), and the wave suite was independently replayed at the baseline. The L2 gap is a missing record, not a defect. |
| F-04 | Roadmap scope note REQUIRED (recommended option) | I confirmed in `intergrax/skills/host_lifecycle.py` that there are no disable/remove/upgrade methods. Only install projection, `enable_installed_skill` and `bind` exist. |

### Evidence I verified myself (not taken from earlier sessions)
- Delegation record sha256 = 5264bca3...bdab (match).
- SHAs resolved with git, all commits, all ancestors of my HEAD b288ba756: bfb63b49 (Wave A), e00a0aab (Wave E), 552e3798 (acceptance baseline), 76a98cb3 (publication), 2f6664d5 (base). 76a98cb's parent = 552e3798 (confirmed).
- `git diff --name-status e00a0aab..552e3798`: 4 docs files (roadmap, freeze checklist, Batch E evidence, new reconciliation doc). Zero code/test delta.
- `git diff 552e3798..76a98cb3`: 3 docs files (roadmap, checklist, reconciliation doc).
- `git diff e00a0aab..HEAD -- intergrax/skills tests/unit/skills` is empty and `git log` for those paths is empty. Skills code and tests are unchanged through my HEAD.
- FRZ status column compared row by row (225 rows, parsed from the table's status column) between e00a0aab and 76a98cb3: identical (130 OPEN, 95 PASS). FRZ-BND-03 is still OPEN. The PASS rows touched (STA-06, REC-02, REC-04, CTL-09, TEN-04) only gained "supporting only" evidence text, and no status changed. New global FRZ PASS = 0.
- No imports from agents/ or applications/ in `intergrax/skills` (grep empty).
- Replay log `AUD-SKILLX/logs/pytest-552e.log`: "117 passed in 94.80s", no skip mention. I did not re-run pytest: skills code is unchanged and the log is at the exact baseline SHA, so no bounded run was needed.
- AUD-SKILLX COMPLETION.json: PASS_SCOPED, PROCEED, no blockers. Its 7-module inventory matches Batch E evidence section 4 as cited by the audit.
- Roadmap 3.0.4 EC9 text: "no prescribed package technology". The roadmap at 76a98cb states TOOL-LIFE-X is not CLOSED.

### Conditions and limits (stated, not hidden)
1. Conditional on the QUAL-X predecessor regularisation, decided by SIGN-5 (DS-5). I do not decide DS-5. This acceptance does not stand without it.
2. The original 2026-10-23 "independently accepted" label (F-01/F-02: self-published, 5m41s after "NOT CLOSED") is NOT validated by this signature. The acceptance stands on AUD-SKILLX (a different session) plus this session's independent verification.
3. Not verified by me and not covered: the original external review (does not exist in the repo), GOV-X2 replay, ME15-ME17 signed-artifact supply chain, EC4 determinism tests outside the 7-module inventory (F-05), runtime/host-factory edits since e00a0aab (F-07), replay at base 2f6664d53.
4. No FRZ-* promotion. PROD-Q Dimension B open. TOOL-LIFE-X not CLOSED. No EVOL-X decision is touched.
5. Assurance: process_enforced; a delegated record is not a cryptographic signature.
6. Sub-decision outcomes are part of the signed text above. I wrote no separate signature lines for OD-SKILLX-A, OD-SKILLX-B, F-03 or F-04.

### Signature lines (copy verbatim)
```
Approved OD-SKILLX-ACCEPT-2026-11-07: operator-delegate @ 2026-11-07T18:18:48+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```
~~~~~~

## 4a. SIGN-2 (SKILL-X) - exact signed text file, verbatim

Source (not tracked; session dossier): `.tmp/session/cert-campaign/SIGNING/SIGN-2_SKILLX/SIGNED_TEXT_OD-SKILLX-ACCEPT-2026-11-07.txt` - sha256 `5b6ecb949a1d0dae54355986c4321ba6ddd6c44fdad71a792fb1fa297c694bf5`. The block below is the file content byte-for-byte, unedited (fence is tildes because the source contains backtick fences).

~~~~~~markdown
OD-SKILLX-ACCEPT-2026-11-07
SKILL-X parent is CLOSED — INDEPENDENTLY ACCEPTED for roadmap §3.0.4 EC1–EC11, scoped to: catalog-witness supply-chain model (EC9), no per-certification full GOV-X2 replay (EC2), structural-only registry-less recovery, lifecycle install/bind/enable, no global FRZ promotion, PROD-Q Dimension B open — on the basis of the AUD-SKILLX audit of 552e379845a9512c275f3ebe520ac51767c97cf8 (publication 76a98cb3f3e4576ab694f03a5b3e270eb7694f9e), replacing the unverified 'external independent review' claim.
Decided inside this record:
- OD-SKILLX-A: ACCEPTED — EC9 reclassification (optional signing out of §3.0.4; catalog-witness model) and EC2 no-full-GOV-X2 scope.
- OD-SKILLX-B: ACCEPTED — no original external review artefact exists in the repository (verified at 76a98cb3f3e4576ab694f03a5b3e270eb7694f9e); the AUD-SKILLX audit of 552e379845a9512c275f3ebe520ac51767c97cf8 serves as the re-acceptance.
- F-03: ACCEPTED as record debt — ROADMAP-REPLAY-X L2 row 222 has no SKILL-X run/log; no targeted L2 entry ordered.
- F-04: roadmap scope note REQUIRED — SKILL-X lifecycle certification covers install/bind/enable only; disable/remove/upgrade are unsupported and untested.
Condition: this acceptance is conditional on the QUAL-X predecessor regularisation (decided by SIGN-5, DS-5). This record does not decide DS-5. If the QUAL-X regularisation is not completed, this acceptance does not stand.
Not granted: no FRZ-* promotion; TOOL-LIFE-X not CLOSED; PROD-Q Dimension B remains open.
~~~~~~

## 5. SIGN-5 (route and SSOT decisions) - SIGNED_DECISIONS.md, verbatim

Source (not tracked; session dossier): `.tmp/session/cert-campaign/SIGNING/SIGN-5_ROUTE/SIGNED_DECISIONS.md` - sha256 `3c8b94ccff3b89612d109a24425b2f109dabb0daf2b1306ba75671cfd65f796f`. The block below is the file content byte-for-byte, unedited (fence is tildes because the source contains backtick fences).

~~~~~~markdown
# SIGNED_DECISIONS - SIGNING/SIGN-5_ROUTE (agent sign_route)

Session id: 806b318c-6f0a-4b61-8041-d2894ebb5fc1 (taken from the session scratchpad path; Claude Code, model claude-sonnet-5-5). New session, dedicated solely to SIGN-5, not a continuation, not an implementation session (see PROGRESS.md entry 1).
Delegation: DELEG-CERTCAMPAIGN-SIGN-2026-11-07, sha256 5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab (verified with sha256sum before anything else; match). Expires 2026-11-07T23:59:59+01:00.
Worktree HEAD (origin/development at signing): b288ba756ef736f3d372832d8e684dbb19901ca4. Signed texts are the files under `texts/` (byte-exact, hash below is sha256 of the file bytes). Nothing was written to any tracked file; nothing was committed; I ran no pytest.

## Overall outcome
| Decision | Outcome |
|---|---|
| OD-ROUTE-DS0-2026-11-07 | SIGNED (wording tightened, not widened) |
| OD-ROUTE-DS1-2026-11-07 | SIGNED (ratifies the AUD-QUALX audit as performed) |
| OD-ROUTE-DS2-2026-11-07 | SIGNED (ratifies the AUD-SKILLX audit as performed) |
| OD-ROUTE-DS3-2026-11-07 | SIGNED (re-audit authorised after correctives; no waiver) |
| OD-ROUTE-DS4-2026-11-07 | SIGNED (WL-7 wording judged NOT to be a waiver; see reasoning) |
| OD-ROUTE-DS5-2026-11-07 | SIGNED_WITH_CONDITIONS (regularisation effective only once OD-QUALX-ACCEPT-2026-11-07 is recorded in the repository) |
| OD-SSOT-D1-2026-11-07 | SIGNED (8 rows as lineage only) |
| OD-SSOT-D2-2026-11-07 | SIGNED, option = exact-SHA audits for L395 and L404; absorption NOT adopted |
| OD-SSOT-D3-2026-11-07 | SIGNED (operator-scoped acceptance only; audit obligation stays attached) |
Not decided here (as instructed): DS-6, DS-9. Also not decided: DS-7, DS-8, D-4..D-6, anything on the exclusion list.

## Common verification (applies to every decision)
- Delegation hash verified; IIAC Amendment 01 read in full. Its effect on the premises: none. `git diff 2f6664d53..b288ba756` (7 commits) touches only the amendment, the signing-delegation template, T-IGX-021-A and EVOL-X records; no roadmap, FRZ checklist, IIAC protocol/verifier/activation file, QUAL-X/SKILL-X/TOOL-LIFE-X/DIST-SCALE-X/TRACE-X/PROD-Q document changed. IIAC scope is still EVOL-X and ENT-AUDIT-X (protocol Purpose line 8; OD-01 line 27; ECAP line 90). The only consequence of the amendment for this task: the `operator-delegate` form is process-enforced and no validator recognises it (amendment section 1 preface); DS-0 says so.
- Every SHA named in the signed texts was resolved with `git cat-file` and checked as an ancestor of HEAD (2f6664d53, d9ea0d6e8, 4c5faa89a, 0d52df6e0, 552e37984, 76a98cb3f, 1548640708, 35344557d, f5c933097, 1786a7fc7, df485a962, 5c9b66db6, 215f82f85, 5f348257e, fcfd59e37, 00460754, ddc87044, 8b3db6763, b7efe6b98, 5a361688e, 25df09093); ddc87044 and 00460754 are ancestors of 8b3db6763; 215f82f85 is before b7efe6b98 and fcfd59e37 before 25df09093.
- Roadmap rows read at HEAD: section 2.2/2.3 (CLOSED needs independent exact-SHA audit), L392-405, L466, L474, L479-483, L491 - all agree with the analysts' descriptions.

---

## OD-ROUTE-DS0-2026-11-07
Signed text file: `texts/OD-ROUTE-DS0-2026-11-07.txt` - sha256 `fdda3f285518629de35d70d5cf1827f7e4efcdb04ad5d13b52a74a170778567b`

Exact text signed:
```text
Decision OD-ROUTE-DS0-2026-11-07 (DS-0, route for non-IIAC stages).
For the stages QUAL-X, SKILL-X, TOOL-LIFE-X and DIST-SCALE-X, which are outside the scope of IIAC policy IIAC-2026-11-05.1 (EVOL-X and ENT-AUDIT-X only), independent acceptance under roadmap section 2.3 is to be obtained through an independent exact-SHA audit in the manner of Protocol 03: a separate auditor session that is not the implementer and not the original closer; an audit package; an auditor pass/findings record; and an operator closure-signature record. IIAC remains limited to EVOL-X and ENT-AUDIT-X. No change is made to the IIAC protocol, its verifiers, its activation gate, its trust anchor or its impact matrix.
This decision does not extend IIAC to any stage (DS-9 is not decided), does not accept or close any stage, and does not promote any FRZ-* row. Where an operator closure-signature record under this route is written by a designated signing session under IIAC Amendment 01, it uses the process-enforced `operator-delegate` form with its Delegation line; that form is not a cryptographic signature and no IIAC verifier recognises it.
```

Evidence verified independently: IIAC scope is EVOL-X and ENT-AUDIT-X only (protocol Purpose, OD-01 Scope, ECAP line 90); ECAP line 152 is the only text that reads as universal, which is the ambiguity DS-0 removes; the IIAC tooling being stage-agnostic (analysis S12) is a technical fact, not a policy extension, and this decision does not use it. Protocol 03-style records exist as precedent in the qualification directory (EBH-5/6/7, DUP-X, ROADMAP-REPLAY-X). The three audits already performed (AUD-QUALX, AUD-SKILLX, AUD-TOOLLIFE) follow exactly this shape (separate session, package, findings record, no operator signature in the auditor record).
Choice and why: adopted as the recommended option. I rejected the alternative of extending IIAC (DS-9) because it would edit a material governance path and break the in-force binding (ACT-N11) on the EVOL-X/ENT-AUDIT-X critical path; and DS-9 is not mine to decide. Tightening only: added that this decision accepts/closes/extends nothing and states the status of delegate lines.
Outcome: SIGNED.
Signature lines:
```text
Approved OD-ROUTE-DS0-2026-11-07: operator-delegate @ 2026-11-07T18:24:31+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

---

## OD-ROUTE-DS1-2026-11-07
Signed text file: `texts/OD-ROUTE-DS1-2026-11-07.txt` - sha256 `9a90f2a0162efe82b3e8d31fbc98ef48680e62ff395bbe5a328ef980b41780c4`

Exact text signed:
```text
Decision OD-ROUTE-DS1-2026-11-07 (DS-1(a), QUAL-X Population A audit).
The independent exact-SHA audit of the QUAL-X Population A formal closure (acceptance SHA d9ea0d6e87b939e4cefb9971c6bbd4fbcaecc0ce; conditional publication commit 4c5faa89aad381e293da2210ec3e83566c3a8473; formal closure commit 0d52df6e0da10bb1ec36bcb113b54a4866b9b76f), already performed on 2026-11-07 by auditor session aud_qualx (verdict PASS_SCOPED; report .tmp/session/cert-campaign/AUD-QUALX/AUDIT_REPORT.md), is ratified as the audit contemplated by DS-1(a). Its single bounded pytest replay at d9ea0d6e8 (41 modules, 630 passed, one invocation) is authorised retroactively as performed, and no further replay is authorised by this decision.
Scope is Population A only; FRZ-REG-01..10 and PROD-Q Dimension B remain OPEN regardless of the verdict.
This ratification concerns the audit route only. It is not the acceptance of QUAL-X (decided separately as OD-QUALX-ACCEPT-2026-11-07), it does not cover QUAL-X-P0 (decided separately), and it does not close audit findings F-01..F-07.
```

Evidence verified independently: AUD-QUALX report and COMPLETION.json (PASS_SCOPED, 630 passed, audited SHA d9ea0d6e8); the log `AUD-QUALX/logs/pytest-d9ea.log` ends "630 passed in 1080.39s"; auditor PROGRESS shows a fresh session, a scratch worktree at the exact SHA, one pytest run and the mutex; chain d9ea0d6e8 -> 4c5faa89a -> 0d52df6e0 confirmed (I re-ran `git diff --name-status d9ea0d6e8 0d52df6e0`: three .md files only - roadmap, FRZ checklist, QUAL-X inventory). The original A1-A5 adjudication is inside the closure commit (self-published), so this audit is the first independent one. The audit also found that the 535-test attestation did not cover d9ea0d6e8 and replayed at the exact SHA, which is why the replay is the ratified element.
Choice and why: DS-1(a) ratify as recommended; retroactive authorisation limited to the replay as performed. The draft's "heavy replay authorised only as listed" is satisfied by naming that one run. Not widened: acceptance (SIGN-1 decided it separately), QUAL-X-P0 and findings F-01..F-07 are expressly left alone.
Outcome: SIGNED.
Signature lines:
```text
Approved OD-ROUTE-DS1-2026-11-07: operator-delegate @ 2026-11-07T18:24:31+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

---

## OD-ROUTE-DS2-2026-11-07
Signed text file: `texts/OD-ROUTE-DS2-2026-11-07.txt` - sha256 `f50eb8443c6c145ffeda0e28398aa51df0a1337bb1789e6a629b667bfef6820b`

Exact text signed:
```text
Decision OD-ROUTE-DS2-2026-11-07 (DS-2(a), SKILL-X audit).
No artefact of the external independent review on which SKILL-X acceptance baseline 552e379845a9512c275f3ebe520ac51767c97cf8 is claimed to rest has been supplied, and none is to be reconstructed or paraphrased; if the operator supplies it later it is archived verbatim and does not alter this decision.
In its place, the independent exact-SHA audit already performed on 2026-11-07 by auditor session aud_skillx at that baseline (verdict PASS_SCOPED; 117 passed in one invocation; publication commit 76a98cb3f3e4576ab694f03a5b3e270eb7694f9e compared with the baseline as documentation-only with no FRZ promotion; report .tmp/session/cert-campaign/AUD-SKILLX/AUDIT_REPORT.md) is ratified as the independent audit for SKILL-X. Its single bounded pytest replay is authorised retroactively as performed, and no further replay is authorised by this decision.
This ratification concerns the audit route only. It is not the acceptance of SKILL-X (decided separately as OD-SKILLX-ACCEPT-2026-11-07) and it does not validate the original 'independently accepted' label of 2026-10-23, which remains unevidenced.
```

Evidence verified independently: AUD-SKILLX COMPLETION.json (PASS_SCOPED, 117 passed at 552e379845a9); `pytest-552e.log` ends "117 passed in 94.80s"; 552e379845a -> 76a98cb3f is the pair named; the auditor found no review artefact in the repository and I found the same in the reconciliation text as quoted by two independent analysts and by SIGN-2; commit times 16:46:35 (552e379) and 16:51:17 (76a98cb) confirm the six-minute flip from "not CLOSED" to "independently accepted". No review artefact was supplied and none is invented.
Choice and why: the mission says ratify the audit at the baseline since no external review artefact was supplied, so DS-2(a)'s first limb (operator supplies the review) is recorded as "not supplied; archive verbatim if it appears later". Not widened: the acceptance of SKILL-X was decided separately in SIGN-2 (SIGNED_WITH_CONDITIONS, depends on DS-5); I did not touch it.
Outcome: SIGNED.
Signature lines:
```text
Approved OD-ROUTE-DS2-2026-11-07: operator-delegate @ 2026-11-07T18:24:31+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

---

## OD-ROUTE-DS3-2026-11-07
Signed text file: `texts/OD-ROUTE-DS3-2026-11-07.txt` - sha256 `7b3546b01eee470424e9b2e573a7bdab220440df7db06a36721c6c83ccec02b9`

Exact text signed:
```text
Decision OD-ROUTE-DS3-2026-11-07 (DS-3, TOOL-LIFE-X re-audit).
TOOL-LIFE-X is neither accepted nor waived. The independent exact-SHA audit by auditor session aud_toollife at baseline 1548640708c850b3fbfca3f2db0e5ecfee4274dd returned CONDITIONAL / chain gate STOP with conditions C-1 (four red tests in tests/unit/runtime/tools/test_tools_side_effect_safety.py) and C-2 (test_f3_streamable_http_parent_cancel_during_catalog_child fails in module order).
A new independent exact-SHA re-audit of TOOL-LIFE-X is authorised, to start only after the corrective tickets for C-1 and C-2 have landed and an exact post-correction SHA is pinned, by an auditor session that is not the implementer of those corrections and not the original closer. The re-audit covers at least conditions C-1 and C-2 and the F8 inventory at the new SHA, and confirms the classification of the citation of 76a98cb3f3e4576ab694f03a5b3e270eb7694f9e as the TOOL-LIFE-X P0 SHA (audit finding F-05: documentation slip). Any pytest replay is one invocation at a time under the heavy-test mutex.
No part of C-1, C-2, the absence of an external review artefact (finding F-03) or the narrowing of exit criteria 7/8 (finding F-04) is waived by this decision. No roadmap label is changed by it; how the existing 'CLOSED - INDEPENDENTLY ACCEPTED' label is treated while the audit verdict is CONDITIONAL is not decided here.
```

Evidence verified independently: AUD-TOOLLIFE report and COMPLETION.json: CONDITIONAL / STOP at 1548640708; the log of the combined run ends "4 failed, 189 passed" and the bisect logs exist for six commits; the F8 inventory 67/67 reproduced. The failing tests are in `tests/unit/runtime/tools/test_tools_side_effect_safety.py` (C-1) and the F3 cancel test (C-2). I did not rerun them (the claim is supported by the logs and nothing I read contradicts it). The citation anomaly: 76a98cb3f is the SKILL-X publication commit and f5c933097 is the TOOL-LIFE-X P0 record (16:56:22), exactly as the auditor says, so F-05 is a documentation slip.
Choice and why: the mission recommends authorising a re-audit after the correctives land, with no waiver. A waiver or acceptance would be unsupported by two red mandatory tests, so I decided only the re-audit and said explicitly that nothing is waived and no label is changed. Added "at least C-1, C-2 and the F8 inventory" as tightening. SIGN-4 handles the corrective plan; I did not decide it.
Outcome: SIGNED.
Signature lines:
```text
Approved OD-ROUTE-DS3-2026-11-07: operator-delegate @ 2026-11-07T18:24:31+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

---

## OD-ROUTE-DS4-2026-11-07
Signed text file: `texts/OD-ROUTE-DS4-2026-11-07.txt` - sha256 `79b18a6fc405c21c2b645de87ee86359c2fdc4065a3a64fe4e95de5f130713f7`

Exact text signed:
```text
Decision OD-ROUTE-DS4-2026-11-07 (DS-4(a), DIST-SCALE-X audit; WL-7).
An independent exact-SHA audit of DIST-SCALE-X per DIST_SCALE_X_L9_AUDITOR_CHECKLIST.md is authorised at AUDIT_BASELINE_SHA 1786a7fc762360dfa90c9ac7eddf2ae30c0c4327 (operator scoped acceptance baseline df485a962688c4fdc4c4e95fea4005d614578c2d), by an auditor session that is not the implementer and not the session of the original operator-scoped closure.
WL-7: the 14400 s soak is not executed under this decision (the Docker engine is unavailable on the designated machine and the C10/C15 operator freeze applies). This decision grants no C10/C15 exception, no Docker use, and no pytest beyond one bounded invocation at a time under the heavy-test mutex. It does not waive the 14400 s requirement. It does not create, re-adopt, ratify or extend the existing records DIST_SCALE_X_WL7_ENVELOPE_AMENDMENT_1H_RECORD.md and DIST_SCALE_X_WL7_4H_DEFERRED_NON_BLOCKING_RECORD.md: they stay exactly as they are, and 14400 s stays NOT PROVEN. The auditor rules independently on whether the recorded 3600 s substitute (run wl07-soak-c8e86ebef60e) satisfies L9 rows 11 and 16 and records any finding, including a finding that it does not; what that finding means for acceptance is reserved to the acceptance decision. Without a proven 14400 s run no verdict stronger than AUDIT PASS (scoped) is possible.
This decision does not accept DIST-SCALE-X, does not decide DS-6, and promotes no FRZ-* row.
```

Evidence verified independently: AUDIT_BASELINE 1786a7fc7 and operator scoped baseline df485a962 resolve and are ancestors; roadmap L483 and `DIST_SCALE_X_G06_G07_OPERATOR_GATE.md` say G-06 is operator scoped acceptance with no external audit in the repository and that upgrading needs a recorded external audit, fabricating a transcript being forbidden. The existing records: `DIST_SCALE_X_WL7_4H_DEFERRED_NON_BLOCKING_RECORD.md` (operator decision 2026-10-28: 14400 s deliberately deferred, NOT PROVEN, 3600 s substitute authoritative for the scoped acceptance, "full" only when 14400 s is proven) and `DIST_SCALE_X_WL7_ENVELOPE_AMENDMENT_1H_RECORD.md`. Docker engine pipe `\\.\pipe\dockerDesktopLinuxEngine` is absent on this machine (Test-Path = False), consistent with DIST-REHEARSAL; the C10/C15 freeze forbids Docker migration and broad diagnostic pytest without separate authorisation.
Is the wording itself a waiver? I judged: no. A waiver relieves a requirement or fixes its outcome. This text (a) does not relieve the 14400 s requirement and states it stays NOT PROVEN, (b) does not create, re-adopt, ratify or extend the earlier deferral/substitute records, so it adds no operator authority to them, (c) leaves the auditor free to find that the 3600 s substitute does not satisfy L9 rows 11/16, (d) reserves what any such finding means for acceptance to the acceptance decision, and (e) says that without a proven 14400 s run nothing stronger than AUDIT PASS (scoped) is possible. Not executing the soak is a statement of the present impossibility under the freeze, not a relief. Had the wording said "retain the 3600 s as authoritative" or "the auditor shall treat the deferral as non-blocking", it would have been a waiver and I would have refused. Residual honest gap: the earlier 2026-10-28 operator records have no verbatim signed text in the repository (analyst gap G-3); I neither cure nor rely on that.
Outcome: SIGNED.
Signature lines:
```text
Approved OD-ROUTE-DS4-2026-11-07: operator-delegate @ 2026-11-07T18:24:31+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

---

## OD-ROUTE-DS5-2026-11-07
Signed text file: `texts/OD-ROUTE-DS5-2026-11-07.txt` - sha256 `01b3103634a0edb0697e91bd984a379245f74f0aa34ace00c9346a6b19128355`

Exact text signed:
```text
Decision OD-ROUTE-DS5-2026-11-07 (DS-5(a), QUAL-X -> SKILL-X predecessor deviation).
The roadmap L480 condition 'SKILL-X not entered until independent acceptance of the QUAL-X formal closure commit' was not met when SKILL-X Batch E, acceptance baseline 552e379845a9512c275f3ebe520ac51767c97cf8 and publication 76a98cb3f3e4576ab694f03a5b3e270eb7694f9e were produced on 2026-10-23 (QUAL-X formal closure 0d52df6e0da10bb1ec36bcb113b54a4866b9b76f was committed at 10:52, the SKILL-X baseline and publication later that day at 16:46 and 16:51, with no independent QUAL-X acceptance at that time). The deviation is recorded as such and is not waived.
It is regularised by the QUAL-X audit ratified in OD-ROUTE-DS1-2026-11-07 (PASS_SCOPED), with effect only from the recording in the repository of the QUAL-X Population A acceptance OD-QUALX-ACCEPT-2026-11-07. Until that acceptance is recorded, and if it is later withdrawn or the QUAL-X audit is reopened, the independent acceptance of SKILL-X, TOOL-LIFE-X and DIST-SCALE-X is conditional on it.
This decision rewrites no historical record or verdict, promotes no FRZ-* row and closes no stage.
```

Evidence verified independently: roadmap L480 states the "SKILL-X not entered until independent acceptance" condition; git committer times on 2026-10-23: 0d52df6e0 10:52:56, 552e37984 16:46:35, 76a98cb3f 16:51:17, f5c933097 16:56:22; AUD-SKILLX and AUD-TOOLLIFE both note the ordering; QUAL-X audit PASS_SCOPED (DS-1). SIGN-1 (sign_qualx, session ec974642-7d45-42d7-a69c-28374140189a) reports OD-QUALX-ACCEPT-2026-11-07 as SIGNED; I read its report and did not re-verify its substance; the signed line is not yet in the repository.
Choice and why: DS-5(a) regularise by audit, as recommended, not DS-5(b) waiver. Condition added (tightening): effect starts only when the QUAL-X acceptance is recorded in the repository, and the three downstream stages stay conditional if it is withdrawn or the audit is reopened. Also noted for the record: SIGN-1 states the QUAL-X acceptance is bound to d9ea0d6e8 and is not carried to product state after it (214 files changed d9ea..HEAD), so regularisation does not certify the current tree.
Outcome: SIGNED_WITH_CONDITIONS.
Signature lines:
```text
Approved OD-ROUTE-DS5-2026-11-07: operator-delegate @ 2026-11-07T18:24:31+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

---

## OD-SSOT-D1-2026-11-07
Signed text file: `texts/OD-SSOT-D1-2026-11-07.txt` - sha256 `dff6a49dc815c09201edaa54eced5977db8b066ff7e1acd53ff56bdb0faf1ed9`

Exact text signed:
```text
Decision OD-SSOT-D1-2026-11-07 (D-1, TRACE-X supersession).
The supersession of eight stale TRACE-X child rows of the roadmap is adopted as 'SUPERSEDED - LINEAGE ONLY' (historical rows; not CLOSED; not accepted on their own): TRACE-X-P5-R2-P3-R1-R1-R1-R1 (L392), TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1 (L393), TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1 (L394), TRACE-X-P5-R2-P4-R1 (L399), TRACE-X-P5-R2-P4-R1-R1 (L400), TRACE-X-P5-R2-P4-R1-R1-R1 (L401), TRACE-X-P5-R2-P4-R1-R1-R1-R1 (L402), TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1 (L403).
Basis: the parents TRACE-X-P5-R2-P3 (P3-R2 b7efe6b980ba010572f9acc68f8d3db4493733e8 with correction 5a361688e78928d23b6e8ffa161bcaf41b4c6ad3) and TRACE-X-P5-R2-P4 / P4-R2 (25df09093bf4d7103f0bf66780ae794d52a7f929) are recorded as closed and independently accepted with unresolved blockers 12-23 and 24-35 equal to 0, and their qualification documents declare the rejected child rows lineage evidence only. The wording is that of SSOT patch 07 as corrected by STALE-REVIEW P-1 and P-3. The two READY FOR AUDIT rows L395 and L404 are excluded (see OD-SSOT-D2-2026-11-07).
Adoption is a label decision only: no audit verdict is created, no parent status is changed and no FRZ-* row is promoted. Publication is by the authorised integration owner, reviewed by a session other than the author of the patch. The statement that the parents were independently accepted rests on repository text; the underlying GitHub audit artefacts are not in the repository.
```

Evidence verified independently: the eight rows at L392-394 and L399-403 read at HEAD (all `[ ] BLOCKED`, none CLOSED); parents L396/L397 (P3-R2, P3-R2-R1) and L398/L405 (P4, P4-R2) are `[x] CLOSED / independently accepted`; the P3 parent reconciliation line 17 "Unresolved blockers (parent scope 12-23): 0" and line 153 "Historical rejected/superseded P3 child rows remain lineage evidence only - they do not block the parent"; the P4 document line 25 designates the R1 chain "historical design lineage"; the P4-R2 document line 11 and 35 "unresolved (24-35) = 0". Ancestry of b7efe6b98, 5a361688e, 25df09093 confirmed. Patch 07 content read: it changes only the status cell to `[x] SUPERSEDED - LINEAGE ONLY` and keeps the description column; the STALE-REVIEW found the original proposal needed four corrections (P-1..P-4), and patch 07 carries P-1 and P-3.
Choice and why: signed. The roadmap section 2.2 vocabulary has no "SUPERSEDED" status and `[x]` is reserved for audited CLOSED, so the text says plainly that this is a label for lineage rows, not a closure. The roadmap already uses `[x] REJECTED` for lineage rows (L471-473), so the form has precedent. Limit stated: the parents' independent acceptance rests on repository text only.
Outcome: SIGNED.
Signature lines:
```text
Approved OD-SSOT-D1-2026-11-07: operator-delegate @ 2026-11-07T18:24:31+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

---

## OD-SSOT-D2-2026-11-07
Signed text file: `texts/OD-SSOT-D2-2026-11-07.txt` - sha256 `ecb0dcbb4da4a689ebbd85f66f882d988d7bb7a867195219bf3f6d18b8268943`

Exact text signed:
```text
Decision OD-SSOT-D2-2026-11-07 (D-2, the two READY FOR AUDIT TRACE-X rows).
Closure-by-absorption is not adopted as a closure mode. Roadmap sections 2.2 and 2.3 define CLOSED only through independent exact-SHA audit; no record defines or accepts an 'absorbed' closure; the repository holds no standalone audit verdict for these two design rows and does not show which audit accepted design locks 24-31.
Independent exact-SHA audits are ordered instead, for exactly two rows: (1) TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1-R1 (roadmap L395; design commit 215f82f855bd6ed5316c1c23f8d35ff076be3526; the roadmap row cites 5f348257e7ff506f57a2b8f381c131ee0e62599f, the START_HEAD of that design); (2) TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1-R1 (roadmap L404; fcfd59e3727163a9e67d61498a293e6002d41159). The auditor pins the exact SHA to be audited. Each audit is by a session that is not the implementer of the TRACE-X P3/P4 chain, makes no product edits, and may be limited to the row's qualification document and its named gate tests (both rows record production delta 0), with at most one bounded pytest invocation at a time under the heavy-test mutex.
Until each audit passes, the row stays '[ ] READY FOR AUDIT', SSOT patch 10 must not be applied, and nothing about their closure is adopted. SSOT patch 07 (OD-SSOT-D1-2026-11-07) does not depend on patch 10.
```

Evidence verified independently: L395 and L404 are `[ ] READY FOR AUDIT`; no qualification document or roadmap text defines a closure mode called "absorbed"; the P3-R2 qualification document line 11 shows START_HEAD 215f82f855bd = the L395 design commit; fcfd59e37 is an ancestor of 25df09093; STALE-REVIEW's own caveat says the repository does not show which audit accepted design locks 24-31 (relevant to the lock-31 / reconcile-before-pin content of L404). Patch 10 would introduce the new closure mode `ABSORBED - CLOSED BY ACCEPTED IMPLEMENTATION` for these two rows.
Choice and why: the evidence favours audits. Closure by absorption would be a new closure mode that is not in roadmap sections 2.2/2.3, would be defined by this signing session on the operator's behalf, and rests on an attribution the repository cannot show; two documentation-only audits (production delta 0) are cheap. Rows needing audits: (1) TRACE-X-P5-R2-P3-R1-R1-R1-R1-R1-R1-R1 (L395; design commit 215f82f855bd6ed5316c1c23f8d35ff076be3526 / roadmap cites 5f348257e7ff506f57a2b8f381c131ee0e62599f); (2) TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1-R1 (L404; fcfd59e3727163a9e67d61498a293e6002d41159). Consequence: patch 10 is not to be applied; those two rows remain OPEN items for the ARCH-FREEZE counter until audited.
Outcome: SIGNED (option: exact-SHA audits).
Signature lines:
```text
Approved OD-SSOT-D2-2026-11-07: operator-delegate @ 2026-11-07T18:24:32+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

---

## OD-SSOT-D3-2026-11-07
Signed text file: `texts/OD-SSOT-D3-2026-11-07.txt` - sha256 `1adc3b1f05ab79855186e4d8796ee913573fd8be02b3fe68efea4ee29defea0c`

Exact text signed:
```text
Decision OD-SSOT-D3-2026-11-07 (D-3, PROD-Q-P0-R1-R1-R1-R1-R1-R1-R1-R1, roadmap L474).
Recorded as an operator-scoped acceptance only: the mechanical composition-ownership and authoritative-gate lock 0046075484cd85da9a220f6f537c48ad94dd8078, which is the implementation evidence of this row, is covered by the operator B6 scoped closure of PROD-Q-P0 (PROD-Q-P0-B6-FORMAL-CLOSURE, 2026-10-21, execution SHA 8b3db6763e4b5ee1d6b3815c4b7f601152169afe; replay 438 collected, 436 passed, 2 skipped). The PROD-Q qualification ledger already maps the lock to this row as accepted under that B6, and the operator closure of 2026-11-02 reaffirms PROD-Q-P0. The row may therefore be labelled '[x] CLOSED - OPERATOR ACCEPTED (scoped)' with the wording of SSOT patch 09.
This is not an independent audit and must not be presented as an audit PASS or as independently accepted, and it is not counted as independently audited evidence for any FRZ-* promotion. The independent exact-SHA audit obligation stays open and visibly attached to: the PROD-Q-P0 closure record (roadmap L466); the lock 0046075484cd85da9a220f6f537c48ad94dd8078; the post-lock corrective ddc8704457192aafffdd32ca8d65e258db2679e0 (READY FOR AUDIT, not superseded; the B6 replay ran on a tree that includes it and its log records no git SHA); and L474 itself.
Not decided here: the disposition of ddc87044 (D-6), DS-7, the other unchecked REJECTED rows (D-5), and any FRZ-PRD-* promotion.
```

Evidence verified independently: L466 reads `[x] CLOSED - OPERATOR ACCEPTED / SCOPED PROD-Q-P0` (B6 2026-10-21, execution 8b3db6763, replay 438/436/0/2) with "Closure-record commit READY FOR INDEPENDENT AUDIT on GitHub"; the PROD-Q qualification document line 24 maps lock 0046075484 to this row ("Accepted under OPERATOR B6") and line 25 records ddc87044 as READY FOR AUDIT, not superseding the lock; the 2026-11-02 operator closure line 16 reaffirms PROD-Q-P0; both 0046075484 and ddc87044 are ancestors of 8b3db6763, so the B6 replay ran on a tree containing the unaudited corrective; L474 itself still says "independent GitHub SHA audit pending". Patch 09 content read: label `[x] CLOSED - OPERATOR ACCEPTED (scoped, ...)` with an explicit "independent exact-SHA audit not performed" clause naming L466, the lock and ddc87044.
Choice and why: signed as a scoped operator acceptance only, because the operator already accepted exactly this evidence under B6 and the label follows the L466 precedent; I did not treat it as an audit result, kept the audit obligation attached at four places, and left D-6 (ddc87044), DS-7, D-5 and any FRZ promotion undecided. This is the only decision in this set that creates an operator acceptance rather than a route or a label, so I limited it to what the operator's own earlier record already says.
Outcome: SIGNED.
Signature lines:
```text
Approved OD-SSOT-D3-2026-11-07: operator-delegate @ 2026-11-07T18:24:32+01:00
Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab
```

---

## Publication note (for the authorised implementer or integration owner)
Copy each signature block verbatim together with its text, with no edits. SSOT patches affected: 07 (needs D1), 09 (needs D3), 10 (must NOT be applied: D2 chose audits). Patches 08, 11, 12, 13 are not covered by any decision here. DS-0..DS-5 are not roadmap edits; they need a decision record file chosen by the integration owner. Delegate lines are not recognised by any existing validator (amendment 01, section 1 preface).
~~~~~~
