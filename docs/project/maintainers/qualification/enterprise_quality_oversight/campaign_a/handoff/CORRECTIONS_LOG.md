<!-- Campaign: EQO-A | BASELINE_SHA: 258bbf6abb6fdbb92ef0185d058527fbb1e47f04 | generated_utc: 2026-11-15T18:54:31Z | source_registry_sha256: 557e1a8981e8d37f2431249e363afd45844edc06d651527689871da08f0bb264 -->

# EQO-A corrections log

## Loop 1 (2026-11-15) - response to the independent audit verdict CONDITIONAL

New MANIFEST.json sha256: `7f8d533f7d56aec100586b113ab347803e228cb782ecd1fd96c657ba27007773`. Checker `--wave W5`: C01 PASS; C02 PASS; C03 PASS; C04 PASS; C05 PASS; C06 PASS; C07 PASS; C08 PASS; C09 PASS; C10 PASS; C11 PASS; C12 PASS; C13 PASS; C14 PASS; C15 PASS; NOT-ASSESSED PASS.

Only data was corrected; no verdict the auditor did not contest was changed, with the consequences listed below (re-labelled rows are named). Every change is reproducible from `data/a2_row_overrides.json`, `data/a5_row_overrides.json`, `findings/overrides.json`, `tools/consolidate_a3.py` (relabel rule) and `tools/render.py`.

| Item | Correction | Files / rows changed |
|---|---|---|
| AUD-01 / K1 | Rule added in `tools/consolidate_a3.py`: CV-VERIFIED requires evidence level E3 or higher. 62 FRZ rows proposed as CV-VERIFIED at E1/E2 were re-labelled: 21 HIST-REQUAL (the row's own text records later changes or `pass_applies_to_current_code` = NO) and 41 EV-INCOMPLETE (existence check only). Original proposal kept in `frz_verdict_executor`, reason in `relabel_reason`. FRZ-REC-02 and FRZ-EXE-05 -> HIST-REQUAL; FRZ-TRC-01, FRZ-STA-01, FRZ-CMP-02 -> EV-INCOMPLETE. D03 now carries a reading guide that defines `PASS applies to current code` as an existence check only, and an evidence-level histogram next to the verdict roll-up. | `A3/frz_rows.jsonl`, D03, D09 roll-ups, `A9/requalifications.jsonl`, D10 |
| AUD-02 / K2 | EQOA-A5-007 extended: second non-zero counter `vendor_neutrality_replay_failures = 1`, the named unscoped paths (as read by the auditor), and the note that sequential assertions in the test mask the second counter. The executor re-executed `all_counters()` directly (run `A5-007-counters`, no pytest): counters 2 and 1. Severity stays HIGH. | `findings/overrides.json` (A5-3 f1), `logs/A5-007-counters*.log`, D05, FINDINGS_REGISTER |
| AUD-03 / K3 | Drift row TOOL-LIFE-X: DRIFT-BENIGN/E3 -> DRIFT-UNREQUALIFIED/E2 with the omitted commit 5df074fcb (2026-10-30, ancestor of the baseline) recorded. | `data/a5_row_overrides.json`, `A5/drift_rows.jsonl`, D05, D09 |
| AUD-04 / K3 | D05 shows `n/a - surface derived by reasoning` instead of 0 where the mechanical surface has fewer than two product paths (30 rows). | `tools/render.py`, D05 |
| AUD-05 / K5 | CONF-001 gets the third (canon) source `PLATFORM_FOUNDATION.md:304` and is narrowed to 'stub silent, canon forbids'; EQOA-A4-003 impact and evidence extended. Severity unchanged. | `data/conflicts.jsonl`, EQOA-A4-003 |
| AUD-06 / K4 | Severity rule for repo-owned gate failures stated in D09 section 2 and appended to EQOA-A4-004/-005/-006 with the reason why A4-005 and A4-006 stay MEDIUM. | D09, `findings/parent_findings.jsonl` |
| AUD-07 / K4 | A2 row PRE-ENT W6-G4 settlement: CV-SCOPED -> OPEN_NOT_CERTIFIED (CV-SCOPED 20 -> 19, OPEN_NOT_CERTIFIED 10 -> 11). | `data/a2_row_overrides.json`, `A2/cert_rows.jsonl`, D02, D09 |
| AUD-08 / K6 | D02 header documents the verdict rule when Q04/Q05 are unsupported; EBH-6 Q05 and independence basis aligned with EBH-5 (PARTIAL / SELF_CLAIMED); EBH-6 verdict stays EV-INCOMPLETE. | `data/a2_row_overrides.json`, D02 |
| AUD-09 / K5 | Wording of the STATE-X/TRACE-X override narrowed to 'two named nodes pass; the suites were not executed'; row evidence note aligned. | `findings/overrides.json`, `data/a5_row_overrides.json`, D05 |
| AUD-10 / K6 | FINDINGS_REGISTER headline distinguishes all recorded from open findings; DEV-06 explains the ID gaps (removed seed placeholders); seed S-12 `finding_id` is now null with `removed_placeholder_finding_id`; EQOA-A1-017 `arch_freeze_impact` aligned with EQOA-A2-001 (BLOCKS); the `:1` line-pointer limitation is stated in FINDINGS_REGISTER and D09. | `data/seeds.jsonl`, `data/deviations.jsonl`, FINDINGS_REGISTER, `findings/overrides.json` |
| AUD-11 / K5 | CONF-002 added: activation record `IN_FORCE` vs verifier FAIL (EQOA-A0-007). | `data/conflicts.jsonl`, AUTHORITY_CONFLICT_REGISTER |

Not changed: all BLOCKER/HIGH verdicts (upheld by the auditor), the A2 verdicts of rows outside AUD-07/AUD-08, and the Q01-Q12 / concern substitutions (limitation: the spec list of the 12 questions is not available to either session).

Notes: (1) the executor's runner also gained `EQOA_*` environment overrides before the audit; unchanged. (2) `origin/development` moved after the baseline (the auditor saw `cc57311d2`); no verdict uses it.

## Loop 2 (2026-11-15) - response to the re-audit residual conditions R-1..R-3

New MANIFEST.json sha256: `d3e1879112e01b5a7fe05de9a2c2f20949b71e83d3b52c82c768f832af2c22a9`. Checker `--wave W5`: C01 PASS; C02 PASS; C03 PASS; C04 PASS; C05 PASS; C06 PASS; C07 PASS; C08 PASS; C09 PASS; C10 PASS; C11 PASS; C12 PASS; C13 PASS; C14 PASS; C15 PASS; NOT-ASSESSED PASS.

| Item | Correction | Files / rows changed |
|---|---|---|
| R-1 | The relabel rule in `tools/consolidate_a3.py` now also applies to E3 rows: a CV-VERIFIED row whose own text records later changes to the certified surface becomes HIST-REQUAL regardless of evidence level. FRZ-REC-05 and FRZ-REC-08 -> HIST-REQUAL. Totals: CV-VERIFIED 2 (FRZ-TRC-02, FRZ-TRC-12), HIST-REQUAL 37, EV-INCOMPLETE 59, CV-SCOPED 14, EV-INCONSISTENT 4, OPEN_NOT_CERTIFIED 109. Re-scan for other rows: no other CV-VERIFIED row carries such a note. Three CV-SCOPED rows (FRZ-FRZ-11, FRZ-AUD-06, FRZ-AUD-10) carry the note 'enterprise_audit code changed after TES prime (finding f2)'; they stay CV-SCOPED because the scoped verdict is the certifier's and the A5 row for ENT-AUDIT-X is DRIFT-REQUALIFIED (FULL 85/85 run in the PDC audit); this is disclosed here for the auditor to contest. | `A3/frz_rows.jsonl`, D03, D09, D10, `A9/requalifications.jsonl` |
| R-2 | TOOL-LIFE-X drift `rationale` (not only `rationale_parent`) now states DRIFT-UNREQUALIFIED at E2 and names `5df074fcb`; D05 rendering updated. | `data/a5_row_overrides.json`, `A5/drift_rows.jsonl`, D05 |
| R-3 | `audit_replay/` copy replaced with the auditor's version 2 of the report, manifest and handoff record; handoff regenerated with the new manifest digest. | `audit_replay/*`, `handoff/EQOA_INDEPENDENT_AUDIT_HANDOFF.md` |
