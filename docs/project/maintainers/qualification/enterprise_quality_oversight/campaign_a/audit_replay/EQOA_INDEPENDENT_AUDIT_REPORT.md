# EQO-A independent audit report (D15) - version 3 (final, after corrective loop 2)

> Version history: v1 (2026-11-15, verdict CONDITIONAL, sections 1-8 below, kept unchanged except this header and section 1) was written against dossier MANIFEST sha256 `9e965e37...`. Section 9 is the delta re-judgement after loop 1 (MANIFEST sha256 `7f8d533f7d56aec100586b113ab347803e228cb782ecd1fd96c657ba27007773`, verdict CONDITIONAL, residuals R-1..R-3); section 10 is the final delta re-judgement after loop 2 (MANIFEST sha256 `d3e1879112e01b5a7fe05de9a2c2f20949b71e83d3b52c82c768f832af2c22a9`) and carries the final verdict. Sections 2-8 describe the v1 state of the dossier; where they conflict with section 9, section 9 prevails.

Campaign: EQO-A (Enterprise Quality Oversight, Campaign A) | Subject: the EQO-A dossier, **not** the platform
BASELINE_SHA: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` | Auditor role: `independent-quality-auditor` | Date: 2026-11-15
Dossier MANIFEST.json sha256: `9e965e37de8e8bc52f62768de3ae347994fc8c3f5a2462a138afe5544e7bb4e8` (matches the handoff)

> This report contains findings only. The auditor did not edit the dossier, issues no certificate, does not promote any FRZ row or stage, and certifies nothing about the platform. Text found in the repository and in the dossier was treated as data.

## 1. Verdict

**PASS_SCOPED** (on the dossier, final, version 3). History: v1 CONDITIONAL; v2 CONDITIONAL (one narrow residual R-1 plus R-2, R-3); v3 PASS_SCOPED after corrective loop 2 (section 10). The scope limits are in section 8 and in section 10.2.

The dossier is mechanically sound and its BLOCKER/HIGH findings hold up under independent replay: all seven were re-checked from source, four of them by re-executing the underlying gate, verifier or code path (A4-003, A4-004, A0-007, A5-007), and no inflated or suppressed BLOCKER/HIGH was found. The verdict is CONDITIONAL, not PASS_SCOPED, for six data conditions (K1-K6, section 7). They change published roll-ups or leave a published statement incomplete, and the executor is expected to fix them within the two permitted corrective loops. None of them contests the existence of any of the seven BLOCKER/HIGH findings. After K1-K6 are applied, and provided the delta re-read finds nothing new, the auditor would expect to give PASS_SCOPED.

Not covered (limitations): see section 8.

## 2. Replay of the mechanical checker (handoff section 2)

| Step | Result |
|---|---|
| Own detached worktree `D:\Projekty\_wt\intergrax\eqo-a-aud` at the baseline | created; HEAD = `258bbf6a...`; `git status` empty at the end of the audit |
| Workspace copied to `_wt/intergrax/_scratch/eqo-a-aud/` (15 MB) | done; MANIFEST.json sha256 equals the handoff value |
| `verify_campaign_a.py --wave W5` with `EQOA_WS`, `EQOA_WT`, `EQOA_REPLAY=1`, main `.venv` python, cwd = own worktree (DEV-03 runner) | C01-C15 PASS, NOT-ASSESSED PASS; C12: 353 manifest entries, 0 mismatched, 0 unlisted |
| Main checkout and executor worktree | not modified (main `git status` unchanged; one `git fetch origin development` was run, which only moves the remote-tracking ref) |
| pytest runs used by the auditor | **0 of 4 remaining slots**: every pytest claim was replayed through the underlying script or Python call instead (section 3) |

Observation: C07 and C09 check that evidence references resolve and that BLOCKER/HIGH rows are not below E3. They do not read the cited lines. Nine findings cite only `:1` line pointers (file-level evidence), including EQOA-A0-007 (acceptable there: the ws log carries the content).

## 3. 100% of BLOCKER/HIGH (7) - independent re-check

| Finding | Auditor result | How |
|---|---|---|
| EQOA-A2-002 BLOCKER (L499 entry condition not met) | **Upheld** | Roadmap L487/L488/L489/L499 read at the baseline. L488 is labelled "CLOSED - INDEPENDENTLY ACCEPTED" but its own annotation says AUD-TOOLLIFE = CONDITIONAL/STOP and "must not be cited as current-HEAD proof"; L489 DIST-SCALE-X is "OPERATOR SCOPED ACCEPTANCE" with "no external Protocol 03 audit in repo" (L2155); L487 SKILL-X rests on AUD-SKILLX PASS_SCOPED, conditional (OD-ROUTE-DS5), artefacts not in the repo. L499 demands "independent exact-SHA evidence" for all three. BLOCKER for the ARCH-FREEZE entry condition is defensible. |
| EQOA-A2-001 HIGH (TOOL-LIFE-X label, mis-cited P0 SHA) | **Upheld** | L488 text as above. `76a98cb3f` resolves to "publish SKILL-X parent independent acceptance"; its child `f5c933097` is the TOOL-LIFE-X P0 commit, so the ledger value (L2154) is the P0 start head, as the finding states. |
| EQOA-A0-007 HIGH (IIAC activation not in force) | **Upheld** | `python -m scripts.governance.verify_iiac_policy_activation --require-in-force` in the auditor worktree: exit 1, `IIAC policy activation: FAIL`, 71 `ACT-N11 product path changed after the audited SHA` lines (the finding says 71). The record itself still says `status: IN_FORCE`. Context: the re-pin is a planned PDC-window action; the finding's recommendation (do not edit outside that process) is right. |
| EQOA-A4-003 BLOCKER (Tier-0 imports applications) | **Upheld; contested label is weaker than stated** | `check_intergrax_no_applications_imports.py` replayed: exit 1, the same 12 files (11 under `intergrax/runtime/execution`, 1 under `intergrax/runtime/architecture`). The regex gate is unchanged since 2026-06-27 and its unit test is in `tests/unit/runtime/architecture/test_faudit_remediation.py:39`. The executor's CONF-001 says the authority reading is contested (AGENTS.md names top-level `applications/`). But `docs/project/architecture/PLATFORM_FOUNDATION.md:304`, the canon cited by SYS-INV-01, states that lower layers "MUST NOT import `intergrax.applications` or `applications`". That resolves the reading in favour of the scanner (see AUD-05). BLOCKER stands. |
| EQOA-A4-004 HIGH (registry bypass gate) | **Upheld** | `check_agent_registry_bypass.py` replayed: exit 1, the same 10 files; the step is in `.github/workflows/unit-tests.yml:363`. The CI result for the baseline SHA was not checked (as in the dossier). |
| EQOA-A3-012 HIGH (TENANT-X missed TASK memory namespace) | **Upheld** | `5ccb72fc` and `de384336d` resolve, and the former is an ancestor of the latter. At `5ccb72fc` `_remember_task` writes with a namespace taken from the scope, with no tenant qualification or guard; at the baseline it uses `task_namespace_for_tenant(identity.tenant_id, namespace)` plus a governance check. No hit for `memory_control` / `TaskMemoryCapability` in `tests/qualification/tenant_x`. |
| EQOA-A5-007 HIGH (ROADMAP-REPLAY-X counter non-zero) | **Upheld, but incomplete** | Re-executed `all_counters()` in the auditor worktree (Python call, not pytest): `historical_evidence_invalidated_by_later_changes = 2`; the violations are `unscoped_prod_delta:intergrax/applications/contracts/environment_profile/sub_profiles.py` and `unscoped_prod_delta:applications/legal_application/mcp/server.py`. **A second counter is also non-zero at the baseline: `vendor_neutrality_replay_failures = 1`** (`unexpected_prod_delta_for_vbn:applications/dispute_sim_application/mcp/server.py`). The test asserts the counters in sequence, so the first failure hides the second; the dossier mentions neither the second counter nor the named paths (grep of the workspace: 0 hits). See AUD-02. |

No BLOCKER/HIGH was found inflated. Suppression candidates are listed in AUD-06 (calibration consistency), not as missed HIGHs.

## 4. Sampled rows

### 4.1 A2 Tier-1 rows (12 of 60; chosen by the auditor, seed `EQOA-AUD-2026-11-15`, stratified by verdict)

| Row | Executor verdict | Auditor result |
|---|---|---|
| EBH-2H | EV-INCOMPLETE | Upheld (no acceptance record for 86fafaa2 in docs/governance; only roadmap, replay matrix, PDC inventories and the gate test hit). |
| EBH-6 | EV-INCOMPLETE | Verdict upheld. Contested detail: Q05 = NO, "independence basis not stated", although the pass record's own "Auditor identity" cell reads "Independent auditor - Protocol 03 ECAP B1-B6"; EBH-5 with the same style of record is rated SELF_CLAIMED. Classify both the same way (AUD-08). |
| AW-7C | EV-INCONSISTENT | Upheld: the qualification record ends "READY FOR AUDIT - not CLOSED" while the roadmap says closed. |
| INT-CONFIG-REAL-X | EV-INCONSISTENT | Upheld: the sole cited architecture record says "P0: BLOCKED" and "do not self-declare CLOSED". |
| EBH-4 | EV-INCONSISTENT | Upheld: the record's last status line is "EBH-4 = READY FOR AUDIT"; acceptance is in the ledger only. |
| S24-GAP-04-P1-R1 | CV-SCOPED | Content claims hold (service file unchanged since 03232729; the certification test dir unchanged since 234d3c03). Contested only on label consistency (AUD-08). |
| S24-GAP-04-CERT | CV-SCOPED | Same as above. |
| QUAL-X | CV-SCOPED | Upheld (honest scoped closure; audit artefacts untracked, recorded as limitation). |
| PRE-ENT / T-IGX-034-A (W6-G4 settlement) | CV-SCOPED | **Contested.** The row's own note says "Valid as a scoped campaign settlement; not a certification. Treat as OPEN_NOT_CERTIFIED for stage purposes", and its two sibling PRE-ENT rows carry OPEN_NOT_CERTIFIED. CV-* codes mean "certificate valid". See AUD-07. |
| MCP-SERVER-X | HIST-REQUAL | Upheld. The 7 MCP-RC commits (a45d0df1b..b0e19bf61, 2026-11-13) postdate cert `919e34b8f`; the A5 row calls it DRIFT-REQUALIFIED on the later RC re-audit. The two labels answer different questions (original certificate vs the RC delta) and are both shown side by side in D04. |
| PDC-X | OPEN_NOT_CERTIFIED | Upheld; the roadmap L498 says "Product implementation **not started**" while P6/P7/P8 records exist (verified). |
| ENTERPRISE-FREEZE-CONTROL-01 | CLAIM-UNSUPPORTED | Upheld (ledger L1939 says "Independently audited"; the only other hits for the id are the replay matrix and the checklist). |

Q01-Q12 substitution (DEV-02, DEV-01): the auditor does **not hold the spec question list either** (the plan only says "all 12 spec questions" and names Q1, 2, 4, 5, 11 as the mechanical ones, which matches the executor's Q01/Q02/Q04/Q05/Q11). A verdict-level comparison is therefore not possible; this is a limitation (section 8). The A4 concern mapping (`A4/spec_concern_mapping.json`) is internally complete: 20 spec concerns each map to at least one executor concern, C25-C28 were added with a dedicated batch; C27 and C28 are only at E2 while the other 26 are at E3.

### 4.2 FRZ rows (14 of 225 = 6.2%; seed `EQOA-AUD-FRZ-2026-11-15`)

FRZ-REC-02, EXE-05, TRC-01, STA-01, CMP-02 (all CV-VERIFIED); TEN-01, TEN-07 (HIST-REQUAL); AUD-04, AUD-08 (EV-INCOMPLETE); HRN-08 (EV-INCONSISTENT); EVOL-01, EVOL-10 (CV-SCOPED); PRD-03, PRD-04 (OPEN_NOT_CERTIFIED).

- Upheld: TEN-01, TEN-07, HRN-08, EVOL-01, EVOL-10, PRD-03, PRD-04, FRZ-AUD-04/FRZ-AUD-08 (verdict EV-INCOMPLETE is defensible; EV-INCONSISTENT would also be defensible because the row says plain PASS while the certifier says PASS_SCOPED).
- **Contested: all five CV-VERIFIED rows.** All five are evidence level E2, and each is marked `pass_applies_to_current_code` YES (an existence check only) or UNKNOWN. By the plan's own scale E2 = "artifact exists", E3 = "content checked against the claim", and CV-VERIFIED = "evidence verified". Of 66 CV-VERIFIED FRZ rows, 62 are E2 (and 34 of those have `pass_applies_to_current_code = UNKNOWN`). Specific defects: FRZ-REC-02 is CV-VERIFIED while its own note says "Static check only (tests not run); 21 durable/recovery product files changed after bd54941d; not covered by this row" (auditor count of recover/checkpoint/continuation/durable/resume paths changed since `bd54941d`: 25), which the A3 brief itself classes as HIST-REQUAL; FRZ-TRC-01 is YES although the two cited implementation files have 3 later commits; FRZ-CMP-02's test directory has 30 later commits. See AUD-01.

### 4.3 Drift rows (4 of 50: 1 UNREQUALIFIED, 1 REQUALIFIED, 2 further)

- DIST-SCALE-X (DRIFT-UNREQUALIFIED, E3): upheld; the rationale itself explains that the 0/0 counts are an artefact.
- MCP-SERVER-X (DRIFT-REQUALIFIED, E2): upheld; counts 0/0 and "surface 1" are wrong for the real MCP surface (7 later MCP commits exist), see AUD-04.
- EBH-4 (DRIFT-REQUALIFIED, E3): class upheld as "intact at the baseline"; the requalification is the unchanged gate plus the executor's own equivalent AST scan, which is not independent requalification evidence. Wording only.
- **TOOL-LIFE-X (DRIFT-BENIGN, E3): contested.** The surface was derived by the sub-agent (input had none). The listed changes omit `5df074fcb` (2026-10-30, "EBH-7 gap-closure W1 ... invoker tenant guard"), a second post-cert commit to the same `invoker.py`; the file has 187 insertions / 212 deletions since `1548640708` and one of the four listed "touches" entries is `UNKNOWN` (provider/MCP bound edits). BENIGN at E3 is too strong. See AUD-03.

### 4.4 Conflict register (73 entries) and seeds (20)

- Register: all 73 entries screened mechanically (sources named; deciding authority and procedure present; about 40 entries are per-row repetitions of two patterns, CONF-A039..A072, which is acceptable). Content-verified against the baseline text: 7 entries (CONF-001, A001, A014, A017, A018, A024, A061). All verified; the other 66 were not content-checked. Gaps: CONF-001 omits the canonical source (AUD-05); the conflict between the activation record (`IN_FORCE`) and its verifier (FAIL) is in no register entry (AUD-11).
- Seeds: all 20 resolutions read; 9 re-verified from source: S-03, S-04 (A2-002), S-11 (EBH-5 "51 rows" is historical wording at the baseline; verified on the L490 text), **S-12 refutation upheld** (`product_manifest_registry.py` is 36 lines with static imports only; commit `01b3e1961` is an ancestor), S-13 (`is_durable` present), S-14 (the four SHAs resolve and are NOT ancestors of the baseline), S-15 (the env-gated skips exist), S-16 (`^Approved:\s*.+$`), S-17 (OD-EVOL-11 and OD-ARCH-FREEZE-01 are OPEN in the register). S-18 and S-19 are supported by the sampled A2 rows (MCP-SERVER-X, PDC-X) but the ledger wording at L2161 was not read. The other seeds were not re-verified. No seed resolution is contested; note that S-12's `finding_id` points to EQOA-A4-001, which does not exist in FINDINGS (S-12 is refuted), which may confuse a reader.

## 5. Handoff section 5 questions

- **Q01-Q12 substitution (DEV-02) / concern list (DEV-01):** see 4.1; not decidable without the spec; mapping complete.
- **Sub-agent judgement:** the sample finds the content claims reliable (A2 sample 8/12 fully upheld, 4 label/consistency contests; FRZ 9/14 upheld; 8 of 8 random E3 MEDIUM/LOW findings re-read at their cited lines were accurate: A7-032, A6-008, A4-036, A1-020, A3-015, A1-017, A3-004, A8-004). The weaknesses are label strength (CV-VERIFIED at E2, BENIGN at E3) rather than facts.
- **Severity calibration:** BLOCKER/HIGH not inflated. A2-002 as BLOCKER for the ARCH-FREEZE entry wording is defensible; A2-001 (HIGH) and A1-017 (MEDIUM, CONDITIONS) describe the same TOOL-LIFE-X label and carry different `arch_freeze_impact` (BLOCKS vs CONDITIONS). Consistency of gate-failure severities is AUD-06. EQOA-A4-044 (MEDIUM; the host-task router built without `production_mode`) was re-read at `nexus_host_execution.py:57` and `agent_router.py:49` and is accurate; MEDIUM is acceptable while reachability is unexecuted.
- **E4 evidence and `-x`:** T02 and T04 were single-node runs, so `-x` limited nothing there; but T04's single failing test **masks the second counter** by assertion order (AUD-02). T03 ran a whole file with `-x` and stopped after the first failure ("1 failed in 0.67s"); other tests of `test_dependency_governance.py` were never evaluated, so EQOA-A4-005 proves one red test, not the file. T06 passed on two named nodes only; the A5-2 f2 override ("the invariants these gates encode hold at the baseline") generalises from two nodes (AUD-09).
- **Process:** DEV-04 (stray file) - the auditor worktree was clean at replay start and end, and the checker's C11 is consistent. DEV-05: 2 of 10 slots wasted, disclosed. DEV-06: FINDINGS.jsonl has 240 unique IDs; no collision remains; unexplained gaps in the sequence: EQOA-A2-003, EQOA-A2-005, EQOA-A4-001 (AUD-10).
- **Post-baseline delta:** the delta was not used for any verdict. At audit time `origin/development` was already `cc57311d25a658a635c5044ea12ae946ac136a35` (2 further docs commits after the executor's `4b4a199e8`). The activation record at that tip still reads `IN_FORCE`; A0-007 is therefore not superseded by any record change visible to the auditor. The auditor did not run the verifier at that tip.

## 6. Contested items (data corrections requested)

| ID | Item | Requested correction | Severity (for the dossier) |
|---|---|---|---|
| **AUD-01** | A3: 62 of 66 CV-VERIFIED FRZ rows are only E2; 34 have `pass_applies_to_current_code = UNKNOWN`; FRZ-REC-02 and FRZ-TRC-01 are verified despite recorded later changes; "YES" means only "cited paths still exist". | Do not use CV-VERIFIED below E3: re-label the rows (HIST-REQUAL where the row's own note records later changes to the certified surface, otherwise a non-VERIFIED label such as CV-STATIC or EV-INCOMPLETE, executor's choice), re-render D03 roll-ups, define `pass_applies_to_current_code` explicitly in the matrix header (existence check only vs "unchanged since pin"), and show the E-level histogram next to the verdict roll-up. | MEDIUM-HIGH (headline roll-up) |
| **AUD-02** | EQOA-A5-007 is incomplete and the T04 record understates the failure. | Add to A5-007 (or file a linked finding): the two named unscoped paths; the second non-zero counter `vendor_neutrality_replay_failures = 1` (`applications/dispute_sim_application/mcp/server.py`); and that the test's sequential assertions mask later counters. Keep severity HIGH (E4 on the first counter, E3 on the second via direct call by this auditor). | MEDIUM |
| **AUD-03** | Drift row TOOL-LIFE-X is BENIGN at E3. | Mention `5df074fcb`; resolve or flag the `UNKNOWN` touches entry; downgrade to DRIFT-UNREQUALIFIED/inconclusive or E2 with an explicit gap. | MEDIUM |
| **AUD-04** | 30 of 50 drift rows show `0` / `n/a` file and commit counts that are artefacts of an empty input surface (e.g. MCP-SERVER-X shows 0/0 although 7 MCP commits postdate its cert; DIST-SCALE-X 0/0 next to UNREQUALIFIED). | Render "n/a - surface derived by reasoning" instead of 0, or fill the counts from the derived surface. | LOW-MEDIUM |
| **AUD-05** | CONF-001 treats Tier-0 -> `intergrax.applications` as contested but omits the canonical statement in PLATFORM_FOUNDATION.md:304. | Add it as a third source; narrow "contested" to "AGENTS.md stub is silent on `intergrax.applications`; canon forbids it". Does not change A4-003's severity. | LOW |
| **AUD-06** | Gate-failure severities are not governed by one rule: A4-003 BLOCKER (E4), A4-004 HIGH (E3), A4-005 MEDIUM (E4, red unit test), A4-006 MEDIUM (a tenant-propagation gate that cannot run). All four fall under the ARCH-FREEZE condition "failing mandatory architecture gate = 0". | State the rule used, or align A4-005 and A4-006 with A4-004 (HIGH/CONDITIONS), with a one-line reason where they differ. | MEDIUM |
| **AUD-07** | A2 row "PRE-ENT / T-IGX-034-A (W6-G4 settlement)": verdict CV-SCOPED contradicts its own note and its siblings. | Change to OPEN_NOT_CERTIFIED (CV-SCOPED roll-up 20 -> 19, OPEN_NOT_CERTIFIED 10 -> 11). | LOW |
| **AUD-08** | A2 verdict rule: rows with the identical profile Q04 = PARTIAL, Q05 = NO, independence NONE_STATED (17 rows) got four different verdicts (6 CV-SCOPED, 8 EV-INCONSISTENT, 2 EV-INCOMPLETE, 1 HIST-REQUAL); EBH-5 (SELF_CLAIMED) and EBH-6 (NONE_STATED) are classified differently on comparable records. | Document in D02's header what drives the verdict when Q04/Q05 are unsupported (the independence-basis column is already shown), and align EBH-6's Q05 with EBH-5 (PARTIAL / SELF_CLAIMED). | LOW |
| **AUD-09** | A5-2 f2 override says the invariants of the STATE-X and TRACE-X closed-world gates "hold at the baseline" on the basis of two nodes. | Reword to "two named nodes pass; the suites were not executed", keep LOW/E4 for those two nodes only. | LOW |
| **AUD-10** | Housekeeping: `FINDINGS_REGISTER.md` headline "BLOCKER 3" vs D01 "BLOCKER 2" (the register counts the merged A2-025; say so); gaps in IDs A2-003, A2-005, A4-001 unexplained (note in DEV-06 or `id_map.json`); seeds.finding_id of S-12 points to a non-existent finding; 9 findings cite only `:1` line pointers; A2-001 vs A1-017 carry different `arch_freeze_impact` for the same label; task brief mentions 237 findings vs 240 in the dossier (240 recorded / 239 open is internally consistent). | Fix or footnote. | LOW |
| **AUD-11** | Conflict register lacks the activation-record vs verifier contradiction (record `IN_FORCE`, verifier FAIL, 71 paths). | Add a CONF entry linking EQOA-A0-007. | LOW |

## 7. Conditions for PASS_SCOPED (K1-K6)

K1 = AUD-01. K2 = AUD-02. K3 = AUD-03 and AUD-04 (D04). K4 = AUD-06 and AUD-07 (severity rule stated; the row relabelled). K5 = AUD-05, AUD-09 and AUD-11. K6 = AUD-08 and AUD-10 (documentation/housekeeping). The auditor will re-read only the delta (CORRECTIONS_LOG.md, the changed rows and re-rendered deliverables) and re-run the checker. No verdict of an uncontested row may be changed in the corrective loops without the auditor being told.

## 8. Limitations of this audit

- The Campaign A spec (the 12 questions and the 20 concerns, original text) was not available to the auditor; the planning plan file only references them. Verdict-level comparison of Q01-Q12 against the spec is **not done**; only the mapping's internal completeness was checked.
- Sampling, not exhaustive: 12/60 A2 rows, 14/225 FRZ rows, 4/50 drift rows, 8 random MEDIUM/LOW findings, 7 of 73 conflict entries content-verified (all 73 screened mechanically), 9 of 20 seeds re-verified from source. Rows outside the sample are not endorsed.
- Test-based claims were replayed through the underlying scripts or Python calls (A4-003, A4-004, A0-007, A5-007), not through pytest; E4 claims T03 (A4-005) and T06 (STATE-X/TRACE-X nodes) were not re-executed.
- CI results for the baseline SHA were not checked (EQOA-A4-004 and the `-x` gaps remain as stated by the executor).
- The platform itself was not audited; no platform verdict is implied. A0-007, A4-003, A4-004 and A5-007 were replayed to confirm that the dossier reports them correctly.
- `origin/development` was fetched once (remote-tracking ref only) to read the post-baseline tip; no verdict uses it.
- `implementation_session_id` and `auditor_session_id` in the manifest are the identifiers visible to the auditor session, not host-issued attestations (assurance level `process_enforced`).

## 9. Re-audit after corrective loop 1 (delta only)

Input: `handoff/CORRECTIONS_LOG.md` (loop 1) and the dossier with MANIFEST.json sha256 `7f8d533f7d56aec100586b113ab347803e228cb782ecd1fd96c657ba27007773` (the value stated in the log; the refreshed scratch copy matches it). Replay: `verify_campaign_a.py --wave W5` with the same `EQOA_*` settings in the auditor worktree (HEAD = baseline, porcelain empty): C01-C15 PASS, NOT-ASSESSED PASS; C12: 356 manifest entries, 0 mismatched, 0 unlisted; C08: 535 roll-ups recomputed. No pytest used. Only changed rows, findings and re-rendered deliverables were re-read; no row outside the log was changed in a way the auditor could detect (FRZ rows without `relabel_reason` still equal the executor proposal: 0 differences).

| Item | Re-judgement | Evidence |
|---|---|---|
| AUD-01 / K1 | **Mostly resolved; residual R-1** | `A3/frz_rows.jsonl`: 62 rows have `relabel_reason` (21 HIST-REQUAL, 41 EV-INCOMPLETE, all previously CV-VERIFIED at E2); 4 CV-VERIFIED remain, all E3 (FRZ-TRC-02, TRC-12, REC-05, REC-08); verdict totals CV-VERIFIED 4, CV-SCOPED 14, EV-INCOMPLETE 59, EV-INCONSISTENT 4, HIST-REQUAL 35, OPEN_NOT_CERTIFIED 109 = 225. The five sampled rows: REC-02 and EXE-05 -> HIST-REQUAL; TRC-01, STA-01, CMP-02 -> EV-INCOMPLETE (acceptable: the rule applies the row's own text; for FRZ-TRC-01 the later commits to the cited files were found by the auditor, not stated in the row, so HIST-REQUAL would have been equally defensible). D03 carries the reading guide (existence-only meaning of `PASS applies to current code`) and the evidence-level histogram, with CV-VERIFIED by level E3 = 4. **R-1:** FRZ-REC-05 and FRZ-REC-08 stay CV-VERIFIED although their notes read, word for word like FRZ-REC-02's: "Static check only (tests not run). 21 durable/recovery product files changed after bd54941d ... not covered by this row". The rule the executor stated (HIST-REQUAL where the row's own text records later changes) was applied only to rows below E3. Half of the remaining "verified" rows therefore contradict the stated rule. |
| AUD-02 / K2 | Resolved | EQOA-A5-007 now names both counters (2 and 1), the named paths, the masking by sequential assertions; evidence `logs/A5-007-counters*.log` (counters 2 and 1, matching the auditor's own call); severity HIGH. Note: the finding cites `ws:audit_replay/EQOA_INDEPENDENT_AUDIT_REPORT.md`, which is the v1 report; the executor should replace that copy with this v2 (digest in the manifest changes). |
| AUD-03 / K3 | Resolved with a wording residual (R-2) | `A5/drift_rows.jsonl` TOOL-LIFE-X: DRIFT-UNREQUALIFIED, E2; the omitted commit `5df074fcb` is recorded in `data/a5_row_overrides.json`. The row's own `rationale` (rendered in D05) still reads "Core enforcement files are unchanged since cert; the only behavioral change (30f67c94a)", which now contradicts the class; `5df074fcb` does not appear in the rendered D05 line. |
| AUD-04 / K3 | Resolved | D05 shows "n/a - surface derived by reasoning" for 29 rows (the log says 30; the auditor counts 29 rows with fewer than two product paths in the data, which equals the rendered 29 x 2 occurrences). Trivial. |
| AUD-05 / K5 | Resolved | CONF-001 has the third source `PLATFORM_FOUNDATION.md:304` and is narrowed; EQOA-A4-003 impact extended; BLOCKER unchanged. |
| AUD-06 / K4 | Resolved | Rule stated in D01 (and on EQOA-A4-004/-005/-006): BLOCKER = hard tier-boundary gate with red unit test (E4); HIGH = other mandatory architecture gate at E3 on a tier/ownership boundary; MEDIUM = dependency governance or a gate that cannot run. A4-005 and A4-006 keep MEDIUM with the stated reasons; accepted (the contest was about the absence of a rule). |
| AUD-07 / K4 | Resolved | PRE-ENT W6-G4 settlement: OPEN_NOT_CERTIFIED; A2 histogram CV-SCOPED 19, OPEN_NOT_CERTIFIED 11 (matches D02). |
| AUD-08 / K6 | Resolved | D02 header states the verdict rule when Q04/Q05 are unsupported; EBH-6 Q05 PARTIAL, SELF_CLAIMED, aligned with EBH-5; EBH-6 verdict EV-INCOMPLETE unchanged. |
| AUD-09 / K5 | Resolved | EQOA-A5-012 (STATE-X/TRACE-X nodes): "Only these two named nodes were executed; the STATE-X and TRACE-X suites were not." |
| AUD-10 / K6 | Resolved | 240 unique finding IDs; DEV-06 explains the gaps; S-12 `finding_id` null with `removed_placeholder_finding_id`; A1-017 `arch_freeze_impact` BLOCKS; register headline and the `:1` limitation stated. |
| AUD-11 / K5 | Resolved | CONF-002 (record IN_FORCE vs verifier FAIL, 71 paths, linked to EQOA-A0-007); register now 74 entries. |

BLOCKER/HIGH set is unchanged (7 findings; A5-007 extended). The numbers in the dossier are now internally consistent for the changed rows. Only data was changed; nothing outside the log was touched.

### 9.1 Verdict after loop 1

**CONDITIONAL**, narrow: eleven of the eleven contested items are corrected except for the one residual that falls under the auditor's own K1 wording and the executor's own stated rule:

- **R-1 (MEDIUM, blocks PASS_SCOPED):** relabel FRZ-REC-05 and FRZ-REC-08 from CV-VERIFIED to HIST-REQUAL (or remove the later-change note with evidence), re-render D03/D09/D10 roll-ups (CV-VERIFIED 4 -> 2, HIST-REQUAL 35 -> 37), and re-check that no other E3 CV-VERIFIED or CV-SCOPED row carries a "changed after ... not covered" note (the auditor's scan found only these two).
- **R-2 (LOW, with R-1):** align the TOOL-LIFE-X `rationale` text (and its D05 rendering) with DRIFT-UNREQUALIFIED and name `5df074fcb` there.
- **R-3 (LOW, with R-1):** replace the cited `audit_replay/` copy of the auditor report with version 2 and update the dependent digest references.

With R-1 to R-3 applied, and the delta re-read finding nothing else, the auditor will issue PASS_SCOPED. This is within the second (last) corrective loop.


## 10. Final re-audit after corrective loop 2 (delta only)

Input: `handoff/CORRECTIONS_LOG.md`, section "Loop 2"; dossier MANIFEST.json sha256 `d3e1879112e01b5a7fe05de9a2c2f20949b71e83d3b52c82c768f832af2c22a9` (the refreshed scratch copy matches the log). Replay in the auditor worktree (HEAD = baseline, porcelain empty): `verify_campaign_a.py --wave W5` C01-C15 PASS, NOT-ASSESSED PASS (356 manifest entries, 0 mismatched, 0 unlisted; 535 roll-ups recomputed). No pytest used.

| Item | Re-judgement | Evidence |
|---|---|---|
| R-1 | **Resolved** | `A3/frz_rows.jsonl`: FRZ-REC-05 and FRZ-REC-08 are HIST-REQUAL (`relabel_reason`: the row's own text records later changes). Totals: CV-VERIFIED 2 (FRZ-TRC-02, FRZ-TRC-12, both E3), HIST-REQUAL 37, EV-INCOMPLETE 59, CV-SCOPED 14, EV-INCONSISTENT 4, OPEN_NOT_CERTIFIED 109 = 225. Rows without `relabel_reason` still equal the executor proposal (0 differences). The auditor's re-scan for CV-VERIFIED rows carrying a later-change note finds none. |
| R-2 | **Resolved** | TOOL-LIFE-X: `rationale` itself now says DRIFT-UNREQUALIFIED at E2, names `5df074fcb` and the UNKNOWN touches entry; the D05 line contains `5df074fcb`. |
| R-3 | **Resolved, with a note** | The `audit_replay/` copies in the dossier are byte-identical to the auditor's version 2 files. They are superseded by this version 3; the executor copies the three final files at publication (a publication step, not a data correction; finding EQOA-A5-007 refers to the path only). |
| Disclosed point: FRZ-FRZ-11, FRZ-AUD-06, FRZ-AUD-10 stay CV-SCOPED | **Not contested** | The three rows are scoped verdicts (E3) bound to the certifier's PASS_SCOPED; two have `pass_applies_to_current_code` UNKNOWN and one YES (existence only); their limitations list OPEN-BP-01, F-W3-001 and Path B. CV-SCOPED means "valid within the declared scope, limitations listed"; it makes no currency claim. The later change to the enterprise_audit code is carried by finding EQOA-A3-030 and by the A5 row for ENT-AUDIT-X (DRIFT-REQUALIFIED at E3, on the later FULL 85/85 run at `cba32d24`, with 14 later paths named as a residual). Suggestion only (no condition): add the later-change fact to these rows' `scope_limits`. |

### 10.1 Statement of the final verdict

- All 11 contested items of v1 (AUD-01..AUD-11) and the three residuals of v2 (R-1..R-3) are resolved in the dossier data; the replay passes; no uncontested row verdict was changed; the BLOCKER/HIGH set (7 findings) was upheld in v1 and is unchanged.
- The dossier is accepted as an evidence-graded baseline inventory **within the scope of section 10.2**. It does not state that the platform is defect-free, is not a certificate, and promotes nothing.

### 10.2 Scope limits that remain (they are why the verdict is PASS_SCOPED, not PASS)

1. The Campaign A spec (12 certification questions and 20 concerns, original text) was not available to the auditor; Q01-Q12 are executor-defined (DEV-02) and could not be compared with the spec.
2. Sampling: 12/60 A2 rows, 14/225 FRZ rows, 4/50 drift rows, 7/73 conflict entries content-checked (all screened), 9/20 seeds re-verified, 8 random MEDIUM/LOW findings re-read. Rows outside the sample are not endorsed by the auditor; the loop-1/2 changes were re-read in full.
3. Test-based claims were replayed through gates, verifiers and direct Python calls, not pytest; E4 claims T03 (EQOA-A4-005) and T06 were not re-executed; T03 ran a whole file with `-x`; CI results for the baseline SHA were not checked.
4. The post-baseline delta was not used for any verdict; `origin/development` at audit time was `cc57311d25a658a635c5044ea12ae946ac136a35`.
5. Evidence levels in the dossier are mostly E2/E3; only four findings are E4. 62 FRZ rows are now EV-INCOMPLETE or HIST-REQUAL for lack of E3 evidence; readers must not treat any FRZ PASS status as current-code proof on the strength of this dossier.
6. Session identifiers in the manifest are those visible to the auditor session (assurance level `process_enforced`); `evidence_publication_sha` and `certification_publication_sha` are all-zero placeholders until the publication PR exists.
