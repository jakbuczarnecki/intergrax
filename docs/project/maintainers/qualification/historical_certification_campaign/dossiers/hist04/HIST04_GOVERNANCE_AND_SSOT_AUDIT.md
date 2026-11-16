# HIST04 - governance and historical SSOT audit (inventory + reconciliation)

Baseline `ba07ebaa34d04633bd60510f68c18346b1ef160d`, branch hist-cert/hist04-governance, worktree clean (no product/doc edits made). HIST-04 is implementer-of-nothing here: this is a candidate inventory, not an audit PASS, not a certification. No test was run (pure read; one hash recompute set, git ancestry checks, regex/source reading).
Companions: HIST04_EARLY_AUTHORITY_NOTE.md, HIST04_IIAC_AUTHORITY_RECONCILIATION.md, HIST04_EVOLX_CLOSURE_RECONCILIATION.md, HIST04_HISTORICAL_SUPERSESSION_MATRIX.json, HIST04_CANONICAL_PATCH_PROPOSALS.md (+ PATCH_H4_01 diff), HIST04_FINDINGS_AND_RESIDUALS.json (16 findings).

## 0. Baseline checks
- FRZ checklist recomputed from the table: 225 rows = 120 OPEN + 105 PASS; historical-85 families REG10 BND6 OWN5 CTR6 TYP6 PLG8 RPL4 OBS7 SEC10 REL11 PRD12 match the brief; excluded AUD10 DOC6 DEBT6 FRZ13 match. **No divergence.** HIST-04 owns no FRZ row; `frz_progress` is empty by design.
- Hashes recomputed: DELEGATION_2026-11-07 sha256 5264bca3...bdab (matches all citations); OD-EVOL-12 and OD-EVOLX-08 pre-signature text digests match their delegation files.

## 1. Mission items -> result
| # | Item | Result |
|---|---|---|
| 1 | latest formal EVOL-X closure SHA | be77b3a5fb97f10a3bd2df6ee038cb92aad77eca (phase 1; phase 2 9f4042110); B6 1dd769331; handoff 7a4f279fd - EVOLX doc s1 |
| 2 | B6, OD-EVOL-12, OD-EVOLX-08 binding | content/digest binding verified; authority binding not verifiable (F-H4-01) |
| 3 | stale EVOL-X PENDING text | L489 (body + status cell), L2065, ladder L81 - PATCH_H4_01 |
| 4 | EVOL-X -> ENT-AUDIT-X handoff | published, consistent, untouched |
| 5-7 | delegated model, Amendment 01, declared vs implemented, adverse cases | IIAC reconciliation doc; F-H4-01..04 |
| 8 | effectivity SIGN-1/2/5 | effective, process_enforced, within DELEG-2026-11-07 (expiry not exceeded); not validator input |
| 9 | QUAL-X predecessor + SKILL-X | see s2 |
| 10 | DS-5 | prerequisite met at baseline (s2) |
| 11 | TOOL-LIFE-X audit dependency | s3 |
| 12 | eight TRACE-X SUPERSEDED | published lineage-only; basis = repo text of parents; GitHub artefacts absent (retained limitation) |
| 13-14 | two READY FOR AUDIT children | audited PASS_SCOPED, closure signed, publication pending; scoped design-lock rows, not open obligations (s4) |
| 15 | DIST-SCALE-X WL-7 4 h | s5 |
| 16 | scoped historical acceptance authority | operator-scoped records (PROD-Q, DIST-SCALE-X) are operator decisions of 2026-10-2x/11-02 whose verbatim signed text is not in the repo for DIST (analyst G-3); delegated 11-07/11-08 decisions are process_enforced operator-delegate |
| 17 | PROD-Q supersession and scoped closure | s6 |
| 18 | misleading CLOSED/PASS claims | s7 |
| 19 | canonical patches | HIST04_CANONICAL_PATCH_PROPOSALS.md |
| 20 | audit/activation work | proposals doc section D |

## 2. QUAL-X predecessor, SKILL-X, DS-5
- QUAL-X: `CLOSED - POPULATION A ONLY`, independently accepted by AUD-QUALX PASS_SCOPED @ d9ea0d6e8 (630 passed, single replay), OD-QUALX-ACCEPT (delegated). Limits: not carried past d9ea0d6e8; FRZ-REG-01..10 OPEN; QUAL-X-P0 verified separately with erratum F-00 (3 gates red at the P0 tree), F-02 (chaos script non-functional), F-03; F-01 (audit-before-P1 gate bypass) acknowledged, not waived; doc fixes F-04/F-06 still deferred.
- SKILL-X: 2026-10-23 ordering deviation (QUAL-X formal closure 10:52:56, SKILL-X baseline 16:46:35, publication 16:51:17, "independently accepted" label 5 min after "not CLOSED") recorded and not waived (DS-5). Re-acceptance on AUD-SKILLX PASS_SCOPED @ 552e37984 (117 passed). Condition: effective from repo recording of OD-QUALX-ACCEPT - **met** (9d0fc1ca1 / b0286f769 ancestors of baseline). Falls if QUAL-X audit is reopened/withdrawn.
- DS-5 downstream: TOOL-LIFE-X and DIST-SCALE-X independent acceptance are likewise conditional on it. HIST-01 prerequisites: QUAL-X Pop. A stays as accepted; do not widen; carry F-04 lifecycle note.
- Classification: SKILL-X = VALID_SCOPED_PASS_CANDIDATE (conditional); QUAL-X Pop. A = ACCEPTED_LIMITATION (scoped).

## 3. TOOL-LIFE-X historical audit dependency (feeds HIST-01)
AUD-TOOLLIFE @ 1548640708: CONDITIONAL / STOP (C-1: four red tests in test_tools_side_effect_safety.py; C-2: F3 streamable-http parent-cancel test fails in module order; F8 inventory 67/67 reproduced). AUD-TOOLLIFE2 @ 2d14bd76f (local, unpublished): CONDITIONAL (C-3: durable record left RUNNING after post-close pool.submit failure - product fix `impl-c0` c3fe07590 implemented, tests green, NOT audited; C-4: disclosure addendum + inventory entry). T-IGX-023-A/B (test fixes for C-1/C-2) audited PASS but unpublished (ticket id collision). Roadmap L482 still reads CLOSED - INDEPENDENTLY ACCEPTED: **misleading** (F-H4-08). OD-ROUTE-DS3 authorises re-audit only after correctives and waives nothing. TOOL-LIFE-X is therefore not closed on evidence; HIST-01 must treat it as AUDIT_REQUIRED.

## 4. TRACE-X
Eight children (L392-394, L399-403) published SUPERSEDED - LINEAGE ONLY (e7eabef81): label decision only, no audit verdict created. L395 and L404: AUD-TRACE-A (215f82f855bd, 30/30 gates, conditions C-1 two broken links, C-2 name the submission SHA - the row cites the START_HEAD 5f348257e not the submitted 215f82f8) and AUD-TRACE-B (fcfd59e3727, 7 passed/7 skipped placeholders, 13 passed at HEAD, Redis ambiguous-write test not reproduced) both PASS_SCOPED; delegated closure decisions signed 2026-11-07T21:32:22+01:00; roadmap not yet updated. They are scoped **design-lock** closures (production delta 0), not implementation acceptance; FRZ-TRC-11 stays OPEN; absorption (patch 10) was rejected by OD-SSOT-D2 and must stay unapplied.

## 5. DIST-SCALE-X
Operator scoped acceptance 2026-10-28 (G-06 satisfied by operator scoped acceptance only, no external audit transcript; G-07). WL-7: 14400 s never proven (aborted at ~3135 s; failed resume at ~13974 s/97 %; build session cancelled); 3600 s substitute run wl07-soak-c8e86ebef60e @ 9a062322 PASS is the authoritative M evidence per operator records; L9 rows 11/16 are "PASS (SCOPED)" by operator records. OD-ROUTE-DS4 authorises an independent audit at 1786a7fc7 and expressly grants no waiver, no C10/C15 exception, no Docker; the audit was not performed (DIST-REHEARSAL: 9 gaps G-1..G-9). Retained limitation: 4 h soak; open: independent audit. Classification INDEPENDENT_AUDIT_GAP (primary), ACCEPTED_LIMITATION (4 h) in the evidence record.

## 6. PROD-Q supersession and scoped closure
PROD-Q-P0 (L466), P2 (L475), P3 (L476), PROD-Q (L478): operator scoped closures, live qualification NOT QUALIFIED (Path B, owned by HIST-03). L474 (P0-R1^8) now `CLOSED - OPERATOR ACCEPTED (scoped)` by OD-SSOT-D3 - explicitly not an audit and not FRZ evidence; independent audit obligation stays attached at L466, lock 0046075484, ddc87044 (READY FOR AUDIT, "retained as open audit item", D-6 signed, unpublished) and L474. L467-470 are unchecked "REJECTED / superseded" boxes (patch 08, signed with conditions, unpublished); L471-473 are `[x] REJECTED`. D-5 (16 REJECTED rows COMPAT-X/TENANT-X) signed with conditions, unpublished.

## 7. Misleading current-HEAD CLOSED/PASS claims (item 18)
| # | Where | Issue | Severity |
|---|---|---|---|
| 1 | roadmap L482 TOOL-LIFE-X | CLOSED - INDEPENDENTLY ACCEPTED vs CONDITIONAL/STOP audits | HIGH |
| 2 | roadmap L489 EVOL-X | PENDING/OPEN vs published closure (inverse: understated) | MEDIUM |
| 3 | handoff/ladder/roadmap statements "operator signatures delegated" for EVOL-X | does not mention the Amendment's non-retroactivity/EVOL-X exclusion; authority gap | HIGH |
| 4 | Amendment 01 header "IN FORCE" | contradicted by its own body; not mechanically enforced | HIGH |
| 5 | roadmap L481 SKILL-X headline | conditional qualifier not in the headline | LOW |
| 6 | roadmap L483 DIST-SCALE-X | "OPERATOR SCOPED" fine, but 4 h NOT PROVEN / no audit not explicit | LOW-MED |
| 7 | roadmap L395 / L404 | still READY FOR AUDIT although audited and signed (understated) | LOW |
| 8 | roadmap L2034, L2065, ladder L81 | stale PENDING wording | LOW |
| 9 | IIAC protocol L4/L91 NOT_IN_FORCE | stale; material path, retained | LOW |
| 10 | "independently accepted" TRACE-X parents L396-398/L405 | rest on repository text; GitHub artefacts absent | LOW |
Scoped certificates cited as platform-wide proof: none found in the canonical roadmap rows I read; QUAL-X and SKILL-X texts carry their scope.

## 8. Certification SHA chains (spot-verified)
EVOL-X chain (s1) all ancestors of baseline and monotone in time. IIAC activation f8c29fb8d -> a5fb6cb51 (evidence/certification publication) -> ... recorded in the activation JSON; not re-walked (no need; `verify_iiac_policy_activation` not executed). QUAL-X d9ea0d6e8 -> 4c5faa89a -> 0d52df6e0 (3 doc files delta, no code). SKILL-X 552e37984 -> 76a98cb3f (docs only). DS-5/TRACE/PROD-Q SHAs were verified by the signers; I re-verified only those cited in this report.

## 9. Counts
Findings 16: AUTHORITY_GAP 2, IMPLEMENTATION_GAP 1, EVIDENCE_GAP 3, SSOT_DRIFT 4, ACCEPTED_LIMITATION 3, INDEPENDENT_AUDIT_GAP 2, VALID_SCOPED_PASS_CANDIDATE 1. Supersession matrix: 20 rows: NO_REMAINING_OBLIGATION 8, VALID_SCOPED_PASS_CANDIDATE 3, INDEPENDENT_AUDIT_GAP 3, SSOT_DRIFT 3, ACCEPTED_LIMITATION 2, PATH_B_LIVE_DEFERRED 1.

## 10. Closure status of HIST-04
Every contradiction is resolved by verified evidence (EVOL-X SHAs, DS-5, FRZ baseline), by a ready patch (PATCH_H4_01), by a justified retained limitation (protocol header, QUAL-X/DIST/TRACE-parent limits), or by an exact authority blocker (operator ratification of EVOL-X delegated lines; Amendment 01 status). Not done by design: any policy/validator change, any signing, any FRZ change. Not executed: adverse-case test run (needs heavy lock and MASTER go-ahead); full-row reread of L483/L2034 before patching B3/B5.
