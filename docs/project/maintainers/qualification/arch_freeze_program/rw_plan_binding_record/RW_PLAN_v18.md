# T-AF2-RW-PLAN — RW product train plan v18 (ARCH-FREEZE, MEP-AF2 v2.1 W2)

## Changelog v17 -> v18

v18 re-extracts the five EQO-A tickets of lanes 4 to 8 from `.tmp/session/arch-freeze/w3-packet-4/` and applies the orchestrator rulings Q-F-2 and Q-F-3. The MEP drafts in `w3-drafts/` are unchanged (all nine plan digests equal the ones recorded in v14 and v17), so no MEP block is refreshed; RW-RUNTIME (051-D) and RW-RECOVERY (051-E) stay at their signed digests. v17 is kept in `work/v17/`.

| Node | Plan | Digest | v17 | v18 | Difference from A1 |
| --- | --- | --- | --- | --- | --- |
| PT-FC3 | `w3-packet-4/043-C/` | `5d4f0ac3...7506` | 62 / 77 / 0 | 62 / 77 / 76 | path set of the plan equal to A1; +76 cond: the census test files of plan section 7 (52 + 33 bounded census, `work/fc3_census.json`, F-80) outside the write set, census-confirmed opt-in only, by the RWD14-01 rule (Q-C-01, Operator: any other file and every `applications/**` path is a stop and an amendment) |
| PT-FC3-F | `w3-packet-4/043-F/` | `903e830d...d202` | 0 / 2 / 0 | 0 / 2 / 0 | none (12 negative tests NT-FC3-CX1..12) |
| PT-WF | `w3-packet-4/043-J/` | `c78454ba...8ab8` | 41 / 51 / 9 | 41 / 51 / 10 | +1 cond: `tests/qualification/ebh_6/_contracts_typing_qualification.py` (scan hit S-1, Q-WF-17) |
| PT-WE | `w3-packet-4/043-K/` | `57c9dc5f...581a` | 0 / 0 / 2 | withdrawn (DORMANT, v18 addendum) | none |
| PT-T01 | `w3-packet-4/043-L/` | `df3d2276...7471` | 0 / 5 / 0 | 0 / 5 / 0 | none (NT-07, NT-04b HELD, Q-T01-01) |

**Q-F-3 (Operator ruling option 3a):** NT-ROW-049 (FRZ-CFG-08) restated: only a vendor-slug default on the eight FC-3 paths fails the CONFIG-X gate; hits outside them and ambient environment reads outside the inventoried surfaces are recorded (printed list, one EDER row per real hit), not failing; `NT-CFG-08-1` withdrawn (no plan defines it). The A4-025 row (NT-ROW-155) is basis N (no failing test is delivered) and A4-025 is an open Operator item.

**Q-F-2 / A4-019:** T-IGX-043-F does not close A4-019 (embedding default, vector-store fallback, guardrail stub are outside the narrowed scan). A4-019 moves from PT-FC3 to the explicit open owner `OPEN-OPERATOR` (no owner yet, flagged for the Operator, lane 4 to 7 packet); PT-FC3 and PT-FC3-F are named as contributors; its row keeps the PT-FC3 tests as contributed. `coverage_check.py` reports open findings apart (A4-019, A4-025) and does not count them as RW-owned.

Section h: the open questions of the five drafts with one owner and a default each (Q-C-01..05, Q-F-1/4/5, Q-WF-01, Q-WF-04, Q-WF-03..18, Q-WF-14b, Q-WE-01..08, Q-T01-01, Q-T01-02..10), plus Q-A4-019 and Q-A4-025 (Operator).

**v18 addendum (orchestrator rulings, `FOLLOW_UPS.md` 2026-11-17 ORCHESTRATOR RULINGS items 1 and 2), applied to v18 before it was handed over:**

- **PT-WE (T-IGX-043-K) withdrawn, DORMANT:** withdrawn from the RW train (orchestrator ruling, FOLLOW_UPS.md 2026-11-17 ORCHESTRATOR RULINGS item 1): the A1 condition 'only if the audit reproduces one' cannot be met inside RW, because RA-1 (PT-Q08) runs after the wave; a reproduction after the wave goes to RB-3 or a second wave by Operator decision (plan section O, item 15). The draft (w3-packet-4/043-K, digest 57c9dc5f...581a) stays DORMANT, not merged; no section d row; the Operator is told in the lane 4 to 7 packet. Removed from the ticket inventory, the lanes, the overlap matrix and the packet scope; its write set if revived: intergrax/runtime/nexus/tools/invoker.py (cond); tests/unit/runtime/nexus/tools/test_we_invoker_residual_hardening.py (cond, new). It closed no finding, so no disposition moves.
- **PT-T01:** NT-07 (A2-049) and NT-04b (detector scope) have no fixer in the train and are held out of the merge set; open Operator item Q-T01-01 names the two test-support files A1 does not contain. The edge RW-TENANT -> PT-T01 now reads: the NT-01..16 nodes with a train fixer must be green when merged.
- **PT-WF (043-J, final INPUT, digest `c78454ba...e8ab8`, equal to the one folded above):** 41 firm product (JA 20, JB 8, JC 4, JG-1..5a 9) and 6 cond product (JG-5b) equal A1; 10 new firm tests; firm total 51 = A1; cond 10 = A1 9 + `tests/qualification/ebh_6/_contracts_typing_qualification.py`. v18 decision on tests/qualification/ebh_6/_contracts_typing_qualification.py (plan :273-277, scan hit S-1 at plan :452): it belongs in the write set as cond. The EBH-6 typing debt registry pins harness_auth.py line 137 and the gate compares violation.line == entry.line, so a JG-5a edit above that line turns the gate red unless the registry line moves with it. The plan's default keeps line 137 unmoved (no edit); the registry edit is needed only if the line moves. A path the ticket may have to edit is listed, as cond, instead of being an unlisted edit at implementation time. Not in this ticket: Part JF (RW-RUNTIME), Part JD (out of the train), A4-005 (DEPS-BATCH), the seven RW-GATES rows, TYP-05 and the ratchet (RW-TYP). Q-WF-01 accepted (one ticket). Q-WF-04 is Operator-only and blocks the merge of Part JB (8 product paths); JA, JC, JG may proceed. Q-WF-03, Q-WF-05 keep their defaults.
- **Operator-only items** (not covered by the standing rules OD-AF-23/S3), marked in section h: Q-WF-04 (blocks PT-WF Part JB), Q-A4-019 and Q-A4-025 (open findings without an owner), Q-T01-01 (NT-07 / NT-04b: fixer ticket or accepted residual), Q-WE-DORMANT (PT-WE withdrawn, for the Operator's information in the lane 4 to 7 packet).

Overlaps: 23 pairs (v17: 23); widened by the FC-3 census: PT-FC1 x PT-FC3 (+6 test files) and PT-FC3 x RW-TRUST (+6 test files under `tests/unit/rag/*` and `tests/integration/rag/*`). PT-WE had no pair. Lanes: PT-WE leaves lane 4 (now 4 tickets: T-AF2-A4007-CALLERS, PT-FC3, RW-SUPPLY, RW-GATES); no other ticket moves; merge order unchanged.

## Changelog v16 -> v17

v17 applies RWD16-01 and RWD16-02 of `.tmp/session/arch-freeze/rw-plan-audit/AUDIT_DELTA_V16.md` through the corrected T-IGX-051-I draft and nothing else. v16 is kept in `work/v16/`.

| Finding | Change in v17 |
| --- | --- |
| RWD16-02 (LOW) | the T-IGX-051-I block now cites the corrected draft: digest `2add59746348470889b69598bda2498f8e022cd7e78fe97b1f8b099c7ad5a244` (checked with `plan_digest.py`; v16 read `1724042a...c48a`); the Source column gives the draft plan lines (table 98-107, conditional line 109) instead of `T-AF2-RW-PLAN`; the draft's count line now reads 'product 4 edited + 1 new, scripts 2, tests 3 new', equal to the block (5 / 10 / 2). Path set, kinds and classes unchanged. |
| RWD16-01 (MINOR) | section d gains the named exception RF-EXC-1 to the red-first base (a ticket that needs merged train code runs red-first at the train head at its start, records the SHA, and shows the defect unchanged from the audit baseline; rules 1-4 and the log path unchanged; T-IGX-051-I is the only named instance). NT-ROW-139 (A4-007): base = the train head at the start of T-IGX-051-I with PT-WB and PT-FC2 merged; the b1/b2 defect is stated unchanged from `a7f023615`; log path `.tmp/session/arch-freeze/T-IGX-051-I/red-first/` (the draft plan step 1 uses the same base). The audit's observation that other later-lane tickets may need the exception is not acted on: each needs its own named entry. |

Overlaps: 23 pairs (v16: 23); lanes and merge order unchanged.

## Changelog v15 -> v16

v16 adds the train ticket **T-IGX-051-I (node T-AF2-A4007-CALLERS)** from the draft `.tmp/session/arch-freeze/w3-drafts/a4007-callers/` (plan digest `1724042abde1792fca0b630db90ab9dcfa6b47b2b90c61b73d81c52ea865c48a`, checked with `plan_digest.py`), following the orchestrator ruling option (b): the 043-B digest `14e4de95...` stays unchanged and the A4-007 callers go to this ticket. v15 is kept in `work/v15/`.

| Item | Change in v16 |
| --- | --- |
| A1 block | new block from draft plan section 7: firm 10 (5 product: `import_policy.py` new, `agent_resolution.py`, `wiring.py`, `environment_conformance.py`, `roster_agent_contract_authority.py`; 2 scripts; 3 new tests), cond 2 (`package_wiring.py` on Q-A7C-02, `environment_diff_wiring.py` on Q-A7C-04). Every path checked at `origin/arch-freeze/rw-train` `6b94d202` (F-79: 7 exist, 3 new with an existing parent) and at the facts SHA. |
| A4-007 ownership | disposition owner PT-WB -> T-AF2-A4007-CALLERS, contributor PT-WB (loaders, `import_allowlist.py`, b4; merged); dedupe D-03 owner updated; NT row of A4-007 restated: NT-A7C-01..10 (08 cond) in the two new test files, controls CTL-A7C-01..04, unit tests UT-A7C-01..05; the NT-WB-18..23, 30 tests are named as contributed and do not count for b1/b2. A4-007 appears once in the dispositions. |
| PT-WB residual | PT-WB note records b1/b2 as its residual, owned by the new node; its three unedited cond Part E callers (`environment_conformance.py`, `environment_diff_wiring.py`, `roster_agent_contract_authority.py`) moved to the new ticket (drop_paths); PT-WB now 19 / 33 / 17 (v15: 19 / 33 / 20). |
| Regression (section e) | 22 of the draft's 23 V2 files as paths; `tests/unit/applications/test_wb_import_allowlist.py` exists only on the train (created by PT-WB) and is named in the targeted text; V3 migration script exit 0; `test_check_application_migrations.py` excluded (no_ci, wrong script path). |
| Section h | Q-A7C-01..08 with one owner each (Operator: Q-A7C-06; the rest orchestrator) and a default; Q-RW-24 recorded as resolved (ruling option b). |
| Lane | precedence PT-FC2 -> T-AF2-A4007-CALLERS (imports `wiring_posture.wiring_fails_closed`). Dependency-earliest lane 3 is full (5), so the node is in **lane 4**, which is now full (5: with PT-FC3, RW-SUPPLY, RW-GATES and the conditional PT-WE). Serialization reasons: after PT-WA: plan G order step 1: PT-WA alone first (largest blast radius); after PT-WB: plan G order step 2: PT-WB second; after PT-FC2: T-IGX-051-I imports wiring_posture.wiring_fails_closed, created by PT-FC2 (draft plan section 9); after PT-WB: file overlap (2 shared: intergrax/applications/_shared/agent_resolution.py, intergrax/applications/_shared/wiring.py); lane 3 full (max 5 parallel tickets); placed in lane 4. |

Overlaps: 23 pairs (v15: 22). New: PT-WB x T-AF2-A4007-CALLERS (`agent_resolution.py`, `wiring.py`; PT-WB merged, lane 1 first). No other ticket shares a write path (`test_package_wiring.py` of PT-FC2 is regression only here). No other ticket changed lane and the merge order is unchanged; lane 4 has no free slot left.

## Changelog v14 -> v15

v15 fixes RWD14-01..03 of `.tmp/session/arch-freeze/rw-plan-audit/AUDIT_DELTA_V14.md` and nothing else. v14 is kept in `work/v14/`.

| Finding | Change in v15 |
| --- | --- |
| RWD14-01 (MINOR) | PT-FC2: the 22 test files of the plan's static census (P intersect C of `w3-packet-2/043-B/work/_P.txt` and `_C.txt`, 32 files; 6 already in scope, 4 non-test) are enumerated as `cond`, census-confirmed opt-in only (`work/fc2_census.json`, FACT_CHECK F-77); PT-FC2 now 10 / 23 / 29 (v14: 10 / 23 / 7). Q-B-02 restated (any other file and every `applications/**` path, `applications/*/tests/` included, is a stop and an amendment) and owned by the Operator. |
| RWD14-02 (LOW) | RW-TRUST: the 12 `[context:` test modules (draft F-21; FACT_CHECK F-78, `work/trust_context_census.json`) are `cond`, expectation-only edit, census-confirmed; RW-TRUST now 7 / 14 / 12 (v14: 7 / 14 / 0). Q-TR-02 corrected: only planner and engine tests remain for step 0, each a stop and an amendment. |
| RWD14-03 (LOW) | single owners: Q-B-02, Q-TEN-02, Q-TEN-08 -> Operator; Q-GAT-01..09 split into Q-GAT-01, 02 (Operator, = GM-1), 02b, 03..08 (orchestrator), 09 (Operator) and Q-GAT-REQ (A1 requests R-1..R-7); GM-5 split from Q-GAT-GM as Q-GAT-GM5 (orchestrator). Defaults added: Q-RW-16 'Part G excluded from the PT-WB packet until ruled', Q-RW-17 'orchestrator text edit; no scope change'. |

Overlaps: 23 pairs (v14: 21). New: PT-FC2 x T-AF2-RW-GATES (`tests/qualification/plug_03/test_plug_03_gates.py`). Widened: PT-FC2 x T-AF2-RW-TENANT to 4 shared paths (+ `tests/unit/applications/test_memory_wiring.py`, `test_mem_audit5a_production_provider_admission.py`). Lanes and merge order unchanged (PT-FC2 lane 2 before RW-GATES lane 4 and RW-TENANT lane 7); the 12 TRUST modules are in no other scope.

## Changelog v13 -> v14

v14 reconciles appendix A1 with the eight MEP train ticket drafts (`.tmp/session/arch-freeze/w3-drafts/<slug>/`) and the EQO-A lane 2/3 drafts (`.tmp/session/arch-freeze/w3-packet-2/`), and applies the orchestrator rulings Q-TR-01, Q-TEN-01 and the NT-FC1 file-column ruling. v13 is kept in `work/v13/`. Counts are firm product / firm total / conditional.

**EQO-A tickets re-extracted from their amended plans** (digest checked with `python -I scripts/governance/plan_digest.py`; A1 `Source` now cites the w3-packet-2 plan lines):

| Node | Plan | Digest | v13 | v14 | Change |
| --- | --- | --- | --- | --- | --- |
| PT-FC1 | `w3-packet-2/043-A/` | `f9c0b109...1147` | 30 / 143 / 14 | 35 / 148 / 24 | the plan enumerates the whole write set (v11 hand list retired; its 100 sweep files equal the census). +5 firm product: Part C1 (4 files) firm by the draft default of Q-FC1-01, `intergrax/agents/defaults.py` replaces the non-existent `agents/defaults.py`, the Tier-3 spec builder `host_orchestration_backend_spec_builder.py` (PT-WA split); +17 conditional product (2 forwarders, `tool_wiring.py`, `scenario_runtime_baseline.py`, 13 `build_harness_host_runtime` call-site files); 10 notebooks listed by name instead of globs. Plan label lines mark the conditional blocks, so v14 classes them by line range (`FC_EXTRACT_COND`). |
| PT-FC2 | `w3-packet-2/043-B/` | `14e4de95...daff` | 10 / 24 / 6 | 10 / 23 / 29 | KR-19 now in the plan (no path change); `tests/unit/docs/test_platform_configuration_reference.py` firm -> cond as the plan states (Q-B-01 draft default, reversible). |
| PT-FC4 | `w3-packet-2/043-D/` | `2dfe0ad9...dd30` | 6 / 12 / 0 | 6 / 12 / 0 | none (docs and governance lines stay outside A1, as in v1). |
| PT-FC2-E | `w3-packet-2/043-E/` | `b641f93f...1f53` | 0 / 2 / 0 | 0 / 2 / 0 | none. |
| PT-WD | `w3-packet-2/043-I/` | `79b760a7...abc7` | 9 / 30 / 8 | 9 / 30 / 8 | none (Source lines re-derived; the plan's 32 counts the two governance files). |
| PT-E05 | `w3-packet-2/044-F/` | `531ef8e1...fb0e` | 0 / 6 / 0 | 0 / 6 / 0 | none (the plan is new; same 6 paths). |
| PT-D08 | `w3-packet-2/046-D/` | `ddc1b231...f424` | 2 / 2 / 0 | 2 / 2 / 0 | none. |

The `21` of the 043-A message is a **scope ceiling**, not a parallel-ticket cap: Operator transcription item 2 (`OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md`, D-FC1-03 and the write-set ceiling of A) accepts *up to 21 conditional product paths* for PT-FC1. The draft has 17 (inside it); keeping A1's class for Part C1 would give exactly 21. The ceiling says nothing about firm paths (30 -> 35 is not covered by that record: Q-FC1-01). The parallel cap is unchanged: MAX_PARALLEL = 5.

**NT-FC1 rows (section d), file column per the 043-A plan section 5:** NT-ROW-019 (01..04 routing file; 12 in the construction-gate file), NT-ROW-020 (07..11 router file), NT-ROW-021 (13..15, 23 persistence file; 20..22, 26 harness file), NT-ROW-031 (related ids with their files), NT-ROW-140 (test path now the router file for 05, 06; 12 gate file; 24 harness file), NT-ROW-142 (25 persistence file), NT-ROW-143 (19 harness file; 16, 17, control 18 persistence file). Six files, unchanged; only the file column moved.

**MEP drafts against A1:**

| Node | Draft digest | Write-set difference | Change in v14 | Recorded in section h |
| --- | --- | --- | --- | --- |
| T-AF2-RW-TENANT | `77e5d3db...32d2` | none in A1, but plan sections 4.1 (GAP-01..08) and 4.2 (pinned tests) name paths the code needs | Q-TEN-01 applied: +10 firm (GAP-06 3 MCP call sites, GAP-07 2 reindex forwarders, GAP-03 CTRL-X test, 4 storage-tool tests), +42 cond (GAP-01/02/04/05 option-dependent paths, the CTRL-X catalog note, 5 reindex and 2 memory tests, 25 `install_agent(` test files, F-75); 14 / 22 / 0 -> 19 / 32 / 42. GAP-05 backends and GAP-08 name no path (stop rule) | Q-TEN-02..12 (Operator: 02, 04, 07, 08, 10), Q-TEN-REG |
| T-AF2-RW-SUPPLY | `23fe5994...` | none (33 = 33); gaps G-01..G-13 are requests | none | Q-SUP-01..10 (Operator: 03, 05, 09), Q-SUP-REG |
| T-AF2-RW-TRUST | `133286ec...` | none, except `intergrax/context/formatter.py` (role chosen at :47-53) | Q-TR-01 applied: formatter.py firm (F-76); 6 / 13 / 0 -> 7 / 14 / 12 | Q-TR-02..08 (Operator: 03, 05, 06) |
| T-AF2-RW-RUNTIME | `5297254e...` | none (17 = 17) | none | Q-RUN-01..11 (Operator: 02) |
| T-AF2-RW-RECOVERY | `40040bf5...` | none (17 = 17) | none | Q-RC-01..10 (Q-RC-01 'add' would create a pair with PT-WB) |
| T-AF2-RW-GATES | `3ad7140d...` | none (35 = 35); requests R-1..R-7 not scope | none | Q-GAT-GM (Operator: GM-1..4, 6), Q-GAT-01..09 |
| T-AF2-RW-TYP | `0434bb52...` | none (6 = 6); re-measurement at the train head equals the HEAD baseline | none | Q-TYP-01..06 |
| T-AF2-DEPS-BATCH | `62053d90...` | none (4 = 4); the four bot PRs are closed unmerged | none | Q-DEP-01..06 (Operator: 04) |

NT-row premises that the drafts correct but that wait for a ruling are left unchanged in section d and named in section h: NT-ROW-065/066/068/074 (TENANT), 088/093 (SUPPLY), 097..102 (TRUST), 106 (RUNTIME), 115/116 (RECOVERY), 057/117/125/164/165 (GATES), and the PT-WD tests without a row (Q-WD-NT). Regression additions proposed by the drafts are not applied to section e until the orchestrator rules (Q-TEN-REG, Q-SUP-REG, Q-RUN-10, Q-RC-07, Q-TR-07, Q-TYP-04).

**Overlaps: 23 pairs (v13: 19).** New: PT-FC1 x PT-FC3 (8 host factories, `host_runtime_composition.py`, 2 scaffold files: FC-1 conditional call sites, FC-3 firm) and PT-FC1 x T-AF2-RW-GATES (`scenario_runtime_baseline.py`). Widened: PT-WA x PT-FC1 (+ the Tier-3 spec builder, `scenario_runtime_baseline.py`), PT-FC1 x PT-WF (+5 host/scaffold files), PT-FC1 x RW-TENANT and PT-WF x RW-TENANT (+ `nexus_loop.py`, TENANT GAP-04 cond), RW-TENANT x RW-SUPPLY (+ `online_evaluation_registry.py`, cond), RW-TENANT x RW-TRUST (+2 reindex tests under TRUST's `tests/unit/rag/*` glob). **Lanes and merge order unchanged**: every new or widened pair already runs in lane order (PT-FC1 lane 3 before PT-FC3 and RW-GATES lane 4; RW-SUPPLY lane 4 and RW-TRUST lane 5 before RW-TENANT lane 7).

## Changelog v12 -> v13

v13 applies the amended W-A plan (Amendment 1 v2) and nothing else. Input: plan `.tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md`, digest `5b7d2827185ae3f06562d5e8d76a4cdec067f26b5283929c0e1ebcc4333ac8d1` (checked with `python -I scripts/governance/plan_digest.py`); independent review `.tmp/session/arch-freeze/w3-packet-0/REVIEW_AMENDMENT_1.md` (REVIEW_PASS on v2); the Operator's approval of that digest and of variant P2 for KR-18 is recorded in `w3-packet-0/PACKET_0.md` ('Amendment record') and `RW_TRAIN_DELEGATION_RECORD_2.md`. v12 is kept in `work/v12/`.

| Item | Change in v13 |
| --- | --- |
| A1 PT-WA | re-extracted from the amended plan; exclusion ranges now plan lines 172-181 (baseline table and design prose), 234 (the `host/wiring.py` 'not listed' sentence) and 279-285 (docs, forbidden); Source line numbers refreshed. Own count: product firm 45, product cond 1, tests and support firm 19, cond 3, firm total 64, cond total 4; equal to the review's count (45 / 1 / 64 / 4). Added (DR-3): `intergrax/tools/contracts/tool_availability.py` (new, firm), `intergrax/skills/registry/tool_requirements.py` (firm), `applications/local_workspace_application/manifest.py` (firm; it was not in any A1 block before, not in Part C either), `applications/local_workspace_application/host/environment_profile.py` (cond, condition in the ticket row), `tests/unit/applications/test_wa_lkw_roster_search_provider.py` (new, firm). Nothing removed. |
| negative tests | the plan now has 25 tests in 8 files (R 4, S 12, P 1, C 8). NT-ROW-156 (KR-18) restated for P2: KR18-1 (full register-node rule) and the new KR18-3 (lazy catalog import pinned) are S, KR18-2/4 controls; red-first adds F-73. New row NT-ROW-169 (A4-004, S2d consequence): NT-WA-23, 25 (S) and NT-WA-24 (C) in `test_wa_lkw_roster_search_provider.py`, red-first F-74; its regression is S3d. |
| KR-18 / Q-WA-09 | variant P2 recorded on the KR-18 disposition; the PACKET_0 Q-WA-09 default (inline helper in `root.py`, the former A1 shape) is superseded (DR-2); the EDER residual (lazy import invisible to the EBH-2A extraction) is named in the disposition note; Q-RW-22 added to the resolved list. |
| decision records | DR-1..DR-4 of plan section 7c are carried in the PT-WA ticket note; DR-4 also in the RWD7-02 contract-editor row (no `Any` in `rag_retrieval_source_scope.py`, no baseline exception; `tool_availability.py` outside the ratchet scope, 0 `Any`/`object`/reflection checked in S6). |
| regression (section e) | PT-WA gains the register node, the EBH-2A gate, the two KR-18 caller tests and the five S3d LKW host files, with the plan's stated limit (red at base and head in the development environment, no pass claimed). |

Overlaps: 23 pairs (v12: 19). None of the five added paths is in any other ticket's scope (exact paths and globs checked), so no pair is added and lanes are unchanged. The S3d regression file `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py` is a firm write path of PT-WB; it is regression-only for PT-WA, so it is not a write overlap, and the order PT-WA -> PT-WB already holds. The other four S3d files and `host/wiring.py` are in no ticket's scope.

## Changelog v11 -> v12

v12 applies the corrections of `.tmp/session/arch-freeze/w3-packet-1/INPUT_043-H.md` (section 'Differences between Appendix A1 and the plan') and nothing else. v11 is kept in `work/v11/`.

| Input row | Change in v12 |
| --- | --- |
| 11, 13 | PT-WB re-extracted from the final W-B plan, digest `f5f0d6304c8b8f270dd7b74981027c6d829d6112a3788328021ce0d00dc5ccc9` (checked with `scripts/governance/plan_digest.py`); A1 Source line numbers refreshed; exclusion range now plan lines 261-295. Own count: product firm 19, product cond 12, tests firm 14, tests cond 5, firm total 33, cond total 17; equal to the EQO-A count (19 / 15 / 14 / 5 / 33 / 20). |
| 7 | NT-ROW-139 (A4-007): `NT-WB-18..23, 30` plus control 24; red-first maps b1 -> NT-WB-20, b2 -> NT-WB-21, b4 -> NT-WB-30. |
| 8 | NT-ROW-147 (X-A8-003): `NT-WB-08..13, 29`; red-first names NT-WB-08/09. |
| 9 | NT-ROW-034 (FRZ-OWN-03): `NT-WB-01..07` (06, 07 controls). |
| 10 | NT-ROW-055 (FRZ-PRD-05) and NT-ROW-153 (A4-040): NT-WB-14..17, 25 firm in `test_wb_serving_bind_defaults.py`; NT-WB-26 is `cond` with Part G, in `test_wb_gca_hosted_strict_serving.py`; red-first restated for the firm part. |
| 3 | the v8 -> v9 changelog bullet that called Part G `firm` is annotated as superseded (Part G is `cond` since v10, RWD9-01). |
| open items | Q-WB-09, Q-WB-13, Q-WB-14 added to section h with the EQO-A draft defaults. |

Overlaps: 23 pairs; lanes unchanged (the W-B path set is the same as in v11; only line numbers and the digest moved).

## Changelog v10 -> v11

v11 fixes RWD10-01 of `.tmp/session/arch-freeze/rw-plan-audit/AUDIT_DELTA_V10.md` and nothing else. v10 is kept in `work/v10/`.

- **RWD10-01:** the PT-FC1 `tests/` sweep is enumerated in scope from a recorded census (`work/fc1_sweep_census.py`, output `work/fc1_sweep_census.json`, FACT_CHECK F-72). Command: `git grep -l -e "NexusLoop(" -e "AgentRouter(" a7f02361540752426f0015b628f8a40b6cbdbb2a -- tests` -> 107 files. Rule applied: keep a file only if its AST has a call whose callee is exactly the name NexusLoop or AgentRouter
(ast.Name id, or ast.Attribute attr, e.g. module.NexusLoop(...)); substring matches such as FooNexusLoop( or matches
in comments/strings/docstrings are non-call matches and are removed. A file that does not parse is kept as cond. Result: 100 call-site files are `firm` (95 with `NexusLoop(`, 5 with `AgentRouter(`; 214 + 7 call sites; the plan's 92 + 5 files / 197 + 7 sites were counted on an earlier tree); the 7 files that match only in strings, comments or docstrings are `cond`, condition: census-confirmed at step 0 that a source-token assertion on `NexusLoop(`/`AgentRouter(` is changed by the FC-1 edit. PT-FC1 now 35 / 148 / 24 (v10: 30 / 48 / 7).
- **Overlaps:** 23 pairs (v10: 18). New pair PT-FC1 x PT-FC2 on `tests/unit/applications/architecture/test_diag_platform_adoption_gate.py` (a cond sweep file for FC-1, a cond census candidate for FC-2), already ordered by the precedence PT-FC2 -> PT-FC1; PT-WA x PT-FC1 widens by the two diag gate tests (PT-WA runs alone first). Lanes unchanged.

## Changelog v9 -> v10

v10 fixes RWD9-01 and RWD9-02 of `.tmp/session/arch-freeze/rw-plan-audit/AUDIT_DELTA_V9.md` and nothing else. v9 is kept in `work/v9/`.

| Finding | Change in v10 | Where |
| --- | --- | --- |
| RWD9-01 | PT-WB Part G (8 product paths: the strict serving command, 4 new `hosting/` files, the compose file, the RUNBOOK, the env example; and 4 Part G tests) is `cond`, condition: B-2 A4-040 ruled in scope (Q-RW-16 b; PROD-Q node runs Q-RW-16 d). No ruling is cited because none is recorded. The PT-WB wave packet excludes Part G until the ruling (ticket condition in section a). The PT-WB / RW-SUPPLY serialization stays. PT-WB now 19 / 33 / 17 (v9: 27 / 45 / 8) | sections a, b; appendix A1 PT-WB |
| RWD9-02 | PT-FC2: the 2 named edited tests (`test_ecp_depth_gate.py`, `test_ecp_control_plane_governance.py`) are `firm` and the 5 census candidates are `cond` (found by the census step); the FC-2 exclusion of plan line 100 is removed. PT-FC3: the 9 inverted or updated tests are `firm`, with their full paths at HEAD. PT-FC2 now 10 / 23 / 29; PT-FC3 62 / 77 / 76 | appendix A1 PT-FC2, PT-FC3 |
| RWD9-02 (re-check) | the other FC plans' write-set prose was re-checked for edited-test lists named by file name only: FC-1 plan line 104 names the 5 `AgentRouter(` test files of its tests/ sweep; they are added as `firm` with their full paths (the rest of the 97-file sweep is not enumerated in the plan). FC-4, 043-E, 043-F and W-F have no such list (FC-4 names its edited tests with full paths). PT-FC1 now 35 / 148 / 24 | appendix A1 PT-FC1 |

Overlaps: 23 pairs (v9: 17). New pair PT-WA x PT-FC2 on `tests/unit/applications/architecture/test_diag_platform_adoption_gate.py` (W-A edits it; FC-2 census candidate), serialized by the PT-WA-alone rule; PT-FC3 x RW-TRUST shares 3 more existing RAG tests (already serialized). Lanes unchanged.

## Changelog v8 -> v9

v9 closes Q-RW-21: PT-WB and PT-WD are re-extracted from their final plans; nothing else changes. v8 is kept in `work/v8/`. Plan digests as read (`python -I scripts/governance/plan_digest.py`):

| Ticket | Plan | Digest | Firm product / firm total / cond |
| --- | --- | --- | --- |
| PT-WB (T-IGX-043-H) | `.tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md` | `e37ee1ec48d739f7426c99db07240b316d54a52010a6b4acdfb115a24dc9b2b6` | 19 / 33 / 17 (v8: 19 / 33 / 15) |
| PT-WD (T-IGX-043-I) | `.tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md` | `8f142c58baccc98a54d88d6e53c903200b1172e643388e6846ab9909fa96d4a5` | 9 / 30 / 8 (v8: 8 / 22 / 0) |

- **PT-WB:** Part E carries A4-007 (DP-21). Part G is inside the ticket and now `firm` (plan heading: inside this ticket, serialized with T-AF2-RW-SUPPLY on the strict serving command): 8 product paths (4 modified, 4 new under `hosting/`) and 4 Part G tests move from `cond` to `firm`. Added as `cond`: the 4 candidate files of a Part C ruling (B-2 A4-020, Q-WB-13) and `tests/unit/applications/test_wb_provider_roots.py`. The A4-040 scope ruling stays open (Q-RW-16 b). Exclusion range: plan lines 254-288 (documents owned elsewhere, part view, forbidden). *(Superseded in v10 by RWD9-01: Part G is `cond`, condition B-2 A4-040 ruled in scope.)*
- **PT-WD:** Q-RW-04 option B as the plan's firm write set: 9 product paths (adds `intergrax/platform/enterprise_audit/evidence.py`), 3 scripts, 18 tests (5 new incl. `test_wd_mapped_family_checks.py`, 13 pinned); `ENTERPRISE_AUDIT_BASELINE.json` stays out (forbidden in the plan). X-A6-003 stays owned by PT-WD (Part B). New `cond` set (Q-WD-11..13): 6 acceptance wrappers, `scripts/enterprise_audit/qual_wave_lib.py`, `test_evidence_publication_strict.py`. The v2 rule that promoted every conditional W-D path to firm is removed, because the final plan's conditional set is a different, decided-later set. Exclusion ranges: prose lines 129, 149, 167-169, 193, the governance-file block 171-178 and the forbidden block 195-216.
- **Other EQO-A plans:** FC-1..FC-4 (043-A..D), 043-E, 043-F and W-F (043-J) re-extracted for comparison only: identical path sets to the v1 extraction and unchanged files, so they are not re-extracted into the plan. W-A was re-extracted in v7 and is unchanged (digest `0c0cd523...e714`).
- **Non-finding improvement (optional note of AUDIT_DELTA_V8.md):** the wave-packet `contract_edit_check` now also runs the TYP-05 detector (`.tmp/session/arch-freeze/rw-plan/work/typ05/measure_typ05.py`, the script that produced the v8 measurement) at each contract-editing ticket head; the regression scope of PT-WA, PT-WB, PT-WF and T-AF2-RW-SUPPLY states that it must report no new string-keyed lookup table (still exactly the 5 tables of TYP05_REDUCTION). Previously a new table would have been caught only when RW-TYP merges.
- **Overlaps and lanes:** unchanged: 23 pairs with the same shared paths (W-B x RW-SUPPLY still shares only the strict serving command; none of the new W-B/W-D paths is in another scope); lanes as in v8.

## Changelog v7 -> v8

v8 fixes RWD7-01..03 of `.tmp/session/arch-freeze/rw-plan-audit/AUDIT_DELTA_V7.md` (all on T-AF2-RW-TYP) and nothing else. v7 is kept in `work/v7/`.

| Finding | Change in v8 | Where |
| --- | --- | --- |
| RWD7-01 | TYP-05 detector defined in NT-ROW-040: a module-level mapping with str-literal / str-constant / UPPER_CASE-constant (or tuple) keys used as a lookup (`[]`, `.get`, `in`) in its module. Re-measured at HEAD over 781 files (`work/typ05/measure_typ05.py`): 5 tables in 2 files (the three `_*` tables of `governed_execution_result.py:57-78` all count, plus two in `external_operation_redrive_trusted_effect_contracts.py`). `typ05_reduction` set to that set. The 3 payload files (`usage_event.py`, `runtime_execution_context.py`, `sanctioned_external_work_operation.py`) are removed from the RW-TYP scope; `external_operation_redrive_trusted_effect_contracts.py` is added | sections a, d; appendix A1 RW-TYP; JSON typ05_reduction |
| RWD7-02 | one baseline: the HEAD measurement (138 Any, 357 object, plus the reflection metrics). The RW-TYP note, NT-ROW-040, NT-ROW-167, NT-ROW-168 and both precedence reasons say so; PT-WA, PT-WB, PT-WF and RW-SUPPLY carry the no-increase check in their regression scope and wave packet; exceptions only as entries of `typ_ratchet_baseline.v1.json` with ticket and reason | sections c, e; JSON ratchet_baseline, contract_edit_check |
| RWD7-03 | option 1: the ratchet covers getattr_dynamic, hasattr, setattr, importlib, __import__ with HEAD baselines 1 / 3 / 0 / 0 / 0; NT-ROW-168 seeds a `getattr(obj, name)` dispatch in a temporary copy of a real contract file | section d |

Overlaps (23 pairs) and lanes are unchanged: the two TYP-05 files are in no other scope.

## Changelog v6 -> v7

v7 re-extracts PT-WA only, from the final W-A plan `.tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md` (plan digest `0c0cd5236637c50b7eb5460d09694b1d9d0f29ca19c6dcd729c6e311ed45e714`, `scripts/governance/plan_digest.py`; input `.tmp/session/arch-freeze/w3-packet-0/INPUT_043-G.md`), and re-runs the overlap matrix and lanes. v6 is kept in `work/v6/`.

- **PT-WA counts:** 45 firm product paths; 64 firm paths in total (19 test/support); 4 conditional. These match the EQO-A count (42; 60 firm / 3 cond). A1 v6 read 41 and 59 / 3.
- **`intergrax/contracts/application_graph_topology.py` belongs to the write set.** The final plan lists it first in the code block of section `Product paths (ACT-N11), new` (plan line 153), the 6th new contract next to the five others; `intergrax/applications/contracts/graph_spec.py` (edited, line 183) is rebased on it and NT-WA-09 pins it. Cause of the omission since v1: the extractor kept only the first mention of a path, which is in the design prose just above the block (line 149 of the final plan, line 155 of the earlier draft), and that prose line lies inside the excluded baseline range; the code-block occurrence was dropped as a duplicate. v7 keeps every occurrence and dedupes after the exclusion ranges (`work/extract_ws.py`, `rw_plan_data.py` FC_EXTRACT_EXCLUDE PT-WA = 140-150, 237-244).
- **Part E:** the two Part E paths are `firm` in the final plan heading (`Product paths, Part E (firm; ...)`, line 203); the registry facade under `intergrax/integrations/registry/` stays out (UNVERIFIED in the plan; a third path would be an amendment).
- **Stale entries corrected (INPUT_043-G items 6, 1):** appendix A1 now has a header that names the version, HEAD and the plan digest of PT-WA; the PT-WA `Source` column cites the final plan's line numbers; the v1 -> v2 changelog line on Part E is annotated as superseded (Part E firm since v6); the Limits paragraph no longer lists W-A Part E or RW-TYP contracts as conditional.
- **Overlaps and lanes:** 23 overlapping pairs, all serialized; lanes unchanged from v6 (the new path is a new file that no other scope names or globs).
- **FACT_CHECK S-19** now records that the amended W-D plan dropped its option-B heading (the citation no longer resolves).
- **New open item Q-RW-21:** the W-B and W-D plans were also amended after the extraction, with changed path sets; not re-extracted in v7 (PT-WA only, as instructed).

## Changelog v5 -> v6

v6 records the Operator decisions of packet 3 (`.tmp/session/arch-freeze/OPERATOR_PACKET_3.md`, 2026-11-17, an English rendering of option choices) and nothing else. v5 is kept in `work/v5/`.

| Decision | Change in v6 | Where |
| --- | --- | --- |
| Q-WA-05 / Q-RW-03 (option A) | W-A Part E (two LKW modules, NT-RW-WA-01) is `firm`; the fallback to PT-G01 is dropped | appendix A1 PT-WA; section g A2-009 |
| OD-AF-11 / Q-RW-14 | T-AF2-RW-TYP firm scope: ratchet gate (no new Any/object in intergrax/contracts/**) + TYP-05 reduction target 4 -> 0 dispatch-table candidates in 4 named files; the conditional `intergrax/contracts/**` scope is removed; FRZ-TYP-05 row becomes basis S (red-first: 4 at HEAD) with the seeded-Any ratchet test as sensitivity; FRZ-TYP-02/04 recorded with RW-TYP (basis M). Overlap matrix and lanes re-run | sections a, b, c, d, g; appendix A1 RW-TYP |
| Q-RW-19 (DP-30 exception) | FRZ-PLG-08 UPGRADE part: DP-30 exception recorded with audit condition 3; supersedes OD-PDC-SCOPE-01 A for UPGRADE only | section g |
| new | Q-RW-20: the W-A plan was amended by EQO-A after the scope extraction; FACT_CHECK S-11 updated (its section 7a no longer exists) | section h; FACT_CHECK |
| OD-AF-22 | approval route recorded: one packet per wave approved by the Operator, Approved lines by the signing delegate with a Delegation pointer; FCS stays the Operator's | section a; JSON approval_route |

## Changelog v4 -> v5

- RWD4-01 (`rw-plan-audit/AUDIT_DELTA_V4.md` section 5): the FRZ-PLG-08 NT-row assertion now covers RETIRE and skill lifecycle only; the UPGRADE part is out of the row (Q-RW-19 / DP-30); `m_conditions` (1)-(3) unchanged. Q-RW-19 in section h carries the 4 conditions of the audit ruling (section 4). Nothing else changed; v4 is kept in `work/v4/`.

## Changelog v3 -> v4

v4 fixes the findings of the delta re-audit of v3 (`.tmp/session/arch-freeze/rw-plan-audit/AUDIT_DELTA_V3.md`, verdict quoted: RW_PLAN_AUDIT_FINDINGS, 0 BLOCKER, 0 MAJOR, 4 MINOR, 1 LOW) and nothing else. v3 is kept in `work/v3/`. Facts stay pinned at `a7f023615`; the product tree equals the audited `e304598b3` (FACT_CHECK F-57).

| Finding | Change in v4 | Where |
| --- | --- | --- |
| RWD-01 | FRZ-SEC-06 retargeted at the remaining gap (security-defense loader with provenance=None on a strict/product root), basis C; `defense_plugin_loader.py` (firm) and `core/security_bootstrap.py` (cond) added to RW-SUPPLY; FRZ-PLG-05 red-first text corrected; code locations cited (F-58..F-60b) | section d; appendix A1 RW-SUPPLY |
| RWD-02 | FRZ-RPL-04 assertion widened to every production isinstance against the concrete adapter (4 sites); `production_lifecycle.py` and `strict_production_serving_launcher.py` added to PT-WF; overlap check re-run | sections b, d; appendix A1 PT-WF |
| RWD-03 | Q-RW-18 resolved: KR-03, KR-04, KR-07 owned by T-AF2-RW-GATES, basis C with a step-0 cause measurement, cond product scope; test-side cause goes to a T-AF2-RED-CLASSES ruling | sections d, g; appendix A0 |
| RWD-04 | FRZ-PLG-08: basis M kept only for RETIRE and skill lifecycle; the ABSENT UPGRADE part routed to the Operator (new Q-RW-19) with a DP-30 freeze-document exception (contributors T-AF2-MANIFESTS, PT-D06) | sections g, h |
| RWD-05 | FRZ-FRZ-10 relabelled basis B; wave aligned: T-B7-PFP-ACTIVATE W5 (plan section J), T-B7-PFP-VERIFY W8 | sections d, g |

## Changelog v2 -> v3

v3 fixes every finding of the independent audit of v2 (`.tmp/session/arch-freeze/rw-plan-audit/AUDIT.md`, verdict quoted: RW_PLAN_AUDIT_FINDINGS, 0 BLOCKER, 3 MAJOR, 4 MINOR, 4 LOW) and applies its Q-RW-07 ruling. Regenerated with `build_rw_plan.py`; v2 is kept in `work/v2/`. No finding is claimed closed here; the audit decides.

| RWA | Change in v3 | Where |
| --- | --- | --- |
| RWA-01 | A4-005 assertion restated: in-process `check_project()` reports no CORE_OWNERSHIP_VIOLATION; red-first at HEAD = `jsonschema`, `tzdata` missing from CORE_ALLOWLIST (F-49, F-50a/b); `scripts/maintenance/check_dependency_ownership.py` added to the DEPS-BATCH firm scope; `tests/unit/test_dependency_governance.py` added to the regression scope; FRZ-DEBT-02/03 follow | section d rows A4-005, FRZ-DEBT-02/03; appendix A1 DEPS-BATCH |
| RWA-02 | matrix appendix C2: 11 rows get a disposition; the 5 PRODUCT rows get an NT row (FRZ-SEC-06 RW-SUPPLY, FRZ-PLG-08 RW-GATES NC gate, FRZ-TYP-01/03 and FRZ-RPL-04 PT-WF); `coverage_check.py` gains a C2 universe; NONPRODUCT C2 rows owned outside the train | sections d, g |
| RWA-03 | expected-red split: table 1 lists only the 6 entries of `expected_red_A01.json`; table 2 lists the planned reds, each with job/step, owner and registration moment; BASELINE:MANIFEST_DIGEST needs a section I exception ruling before PT-WD merges | section c |
| RWA-04 | specific regression paths for all 18 phase-3 findings (checked to exist); regression rows for the non-train owners PT-G01, PT-G02, PT-E02, PT-E03; coverage requires >= 1 existing path per finding | section e |
| RWA-05 | audit-flag owner binding recorded in the JSON (`audit_flags`) and checked; A3-018/PT-G02 bound to PT-Q02 by the WP-12 PT-G02 row; T-IGX-051-H/-D ticket files must carry the audit line | sections d, a; JSON |
| RWA-06 | A4-004 and X-A4-004 red-first restated for the new assertion: 4 files at HEAD; the 10-file count kept as context for the old gate | section d |
| RWA-07 | binding start rule: at most 5 train tickets in flight; capacity-placed tickets start only when a slot frees | section c |
| RWA-08 | PT-T01 wildcard paths use the dedicated kind `new_glob` and are part of the overlap expansion | appendix A1; build |
| RWA-09 | path corrected to `.tmp/session/arch-freeze/W0-ci-probe/expected_red_A01.json` | sections c, e |
| RWA-10 | FRZ-FRZ-10 names its planned test file `tests/unit/governance/test_pfp_post_freeze_policy_gate.py` (T-B7-PFP-ACTIVATE, W5) | section d |
| RWA-11 | external precedence recorded: T-AF2-MANIFESTS final after PT-WA (FRZ-FRZ-06) and PT-FC1 (FRZ-FRZ-08), and after RW-GATES for the NOT_COMPOSED marks | section c |
| Q-RW-07 | audit ruling ACCEPT WITH CONDITIONS applied: conditions attached to each of the 15 basis-M rows; M rows reported apart from the red-first count | section d; JSON `m_conditions` |
| new | Q-RW-18: KR-03/KR-04/KR-07 have no disposition in this plan although the register names RW-GATES (KR-03/04) | section h |

## Changelog v1 -> v2

v2 applies the answers in `ORCHESTRATOR_ANSWERS_RW.md` (ARCH-FREEZE orchestrator) and `EQOA_ANSWERS_RW.md` (EQO-A orchestrator), relayed by the orchestrator; regenerated with `build_rw_plan.py`. The v1 outputs are kept in `work/v1/`.

- Ticket letters: PT-WE = T-IGX-043-K, PT-T01 = T-IGX-043-L, PT-D08 = T-IGX-046-D, PT-E05 = T-IGX-044-F.
- PT-WA: Part F removed (A4-007 -> PT-WB Part E); Part E and its test NT-RW-WA-01 are `cond` with the fallback to PT-G01 (Operator Q-WA-05 open); KR-18 added (`environment_profile/root.py` + new test). *(Superseded in v6: Part E is firm, OPERATOR_PACKET_3.)*
- PT-WD: the Q-WD-07 option B set is binding (`firm`); `ENTERPRISE_AUDIT_BASELINE.json` removed from the write set; `BASELINE:MANIFEST_DIGEST` registered as expected-red with owner PT-WD; the W-D Approved line is the Operator's (gate-meaning).
- Reassignments D-04, D-05, D-07, D-11, D-12 accepted; W-F Part JD stays outside the train.
- New: re-disposition rule for basis-C rows (section d); expected-red table with owners (section c); DEPS-BATCH re-listed at the train cut; RW-TYP gated on OD-AF-11.
- Known-red FIX rows assigned with negative-test rows: KR-18 -> PT-WA, KR-19 -> PT-FC2 (`observability_otlp.py`, `export_boundary.py` cond), KR-17 -> T-AF2-RW-GATES (MongoDB `client.py`, `opens.py` cond).
- W-B Part G stays inside W-B, serialized with RW-SUPPLY on `package_b_strict_production_serving_command.py` (the overlap is in the matrix); A4-040 lands in that file (W-B Part G or RW-SUPPLY).
- Section h keeps only Q-RW-03, Q-RW-07, Q-RW-14 and the new Q-RW-16 (Operator items), Q-RW-17 (orchestrator text amendments); the other answers are in appendix A0.

> **Status: planning record, not a verdict.** Written by the planning node T-AF2-RW-PLAN (read-only on tracked files; no worktree, branch or PR). It closes nothing, promotes no FRZ status and claims no PASS. The independent node T-AF2-RW-PLAN-AUDIT reviews it; phase check 3 stays as WP-12 recorded it until that audit runs.

| Item | Value |
| --- | --- |
| Facts read at | `origin/development` = `a7f02361540752426f0015b628f8a40b6cbdbb2a` (fetched 2026-11-16). Product tree identical to `AUDIT_BASELINE_2` `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (FACT_CHECK F-50). |
| Campaign B baseline | `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`; only 3 product files changed B -> HEAD, none in any train scope (F-51; per-ticket drift column in rw_plan.v1.json is empty for all 22 tickets) |
| Train branch | `arch-freeze/rw-train` (plan section G; DP-11 one product wave) |
| Generator | `build_rw_plan.py` (data `rw_plan_data.py`, `rw_plan_negtests.py`, `regression.json`, `dedupe.json`, `open_questions.json`) -> `rw_plan.v1.json`; `coverage_check.py` -> `coverage_check.out.txt`; `render_md.py` -> this file and `FACT_CHECK.md` |
| Inputs | plan sections 0, E, G, I, J, K and appendix Z; WP-12 `TICKETS_DRAFT.md`, `PHASE_CHECKS.md`, `finding_node_map.json`, `master_rows.jsonl`; EQO-A plans `PLAN_T-IGX-043-A..J`; repo registers at HEAD (FRZ matrix appendix C, DS-EXC package tables A/B/C, known-red register, gap register); closed-world `RESULT.md`; B-1/B-2 alpha results; REV-MASTER RM-02/RM-03 |

**Counts.** 22 train tickets; 23 overlapping ticket pairs (all serialized); 9 lanes; 169 negative-test rows; 192 dispositions; coverage gaps reported by `coverage_check.py`: 0; 117 open questions.

**Scope rules.** Each scope entry is one of `file` (exists at HEAD), `new` (absent at HEAD, parent directory exists) or `glob` (matches >= 1 tracked file). Class `firm` is binding; `cond` is binding only under the stated condition. EQO-A tickets take their scope from the write set of their plan (overlap tables, forbidden lists and documents owned elsewhere excluded); the MEP-AF2 tickets have no prior plan and get their scope here. The full per-path list with its source line is appendix A1 and `rw_plan.v1.json` (`tickets[].scope`).

## (a) Ticket inventory

`Tnn` is the short id used in the overlap matrix. Closes = IDs this ticket owns (exactly-once ownership, section g). Contributes = IDs owned elsewhere that this ticket helps close.

| Id | Node | Ticket | Sens | Lane | Firm product / firm total / cond paths | Closes (owned) | Contributes to | Notes |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| T01 | PT-WA | T-IGX-043-G | PRODUCT | 0 | 45 / 64 / 4 | FRZ: FRZ-BND-04; finding: A2-009, A4-003, A4-004, X-A4-004; other: KR-18 | FRZ-EXE-02, FRZ-PLG-05 | W-A boundary inversion (12 runtime modules) and Tier-2 agent seams. applications/local_workspace_application/host/environment_profile.py is cond: only if step 0 finds that apply_roster_agent_governance does not leave integration_profile unchanged (the review of Amendment 1 read agent_certification_wiring.py:71-78 at b6738c73a and found that it does, so the path is not expected to be needed); the three conditional test paths: only if their source-token or call-shape assertions see the change; v6: Part E (S8b, two LKW production modules through facades, DP-20) is firm, a separable last commit group (OPERATOR_PACKET_3, Q-WA-05 option A). v13 (Amendment 1 v2, digest 5b7d2827...c8d1, REVIEW_PASS in w3-packet-0/REVIEW_AMENDMENT_1.md; the Operator approved the digest and variant P2, PACKET_0.md 'Amendment record', RW_TRAIN_DELEGATION_RECORD_2.md): decision records of plan section 7c: DR-1 BLOCKER-1 option (a), the web-search binding's enabled flag is derived from the search_provider slot through one function (local_workspace_profile_has_search_provider in host/agent_factories.py) and one profile instance (manifest.py, host/environment_profile.py cond); DR-2 KR-18 / Q-WA-09 variant P2 (new intergrax/tools/contracts/tool_availability.py, re-export in skills/registry/tool_requirements.py, inline admission builder in root.py) supersedes the PACKET_0 default (inline helper in root.py, the former A1 shape); DR-3 the five Amendment 1 paths are in this A1 block; DR-4 the contracts Any ratchet does not rise (rag_retrieval_source_scope.py typed with JsonValue, no baseline exception entry); Part F (A4-007, NT-WA-18..22) moves to PT-WB Part E (dedupe D-03, accepted v2) |
| T02 | PT-WB | T-IGX-043-H | PRODUCT | 1 | 19 / 33 / 17 | FRZ: FRZ-OWN-03, FRZ-OWN-05, FRZ-PRD-05; finding: A4-020, A4-040, X-A4-016, X-A8-003 | A4-007, FRZ-BND-05, FRZ-CFG-02, FRZ-CFG-06, FRZ-CTR-01, FRZ-CTR-02, FRZ-CTR-06, FRZ-SEC-04 | W-B one LLM resolver, profile projections, provider roots, serving, importlib allowlist. Part G (8 product paths, 4 tests): B-2 A4-040 ruled in scope (Q-RW-16 b; PROD-Q node runs Q-RW-16 d). The PT-WB wave packet excludes Part G until that ruling is recorded; the PT-WB / RW-SUPPLY serialization on the strict serving command stays; v16: A4-007 seams b1/b2 (the product callers) are a residual of PT-WB (ticket checkpoint) and are owned by T-AF2-A4007-CALLERS (T-IGX-051-I); the three cond Part E callers environment_conformance.py, environment_diff_wiring.py and roster_agent_contract_authority.py, which PT-WB did not edit, moved there (drop_paths). PT-WB keeps the loaders, import_allowlist.py and b4. v9: re-extracted from the final plan; Part E carries A4-007 (DP-21). v10 (RWD9-01): Part G is inside the ticket but lands only if B-2 A4-040 is ruled in scope (plan :281, :351, :413), so its 8 product paths and 4 tests are cond |
| T03 | PT-FC2 | T-IGX-043-B | PRODUCT | 2 | 10 / 23 / 29 | FRZ: FRZ-PRD-03, FRZ-REL-07, FRZ-REL-09, FRZ-REL-10; finding: A4-030, A4-032, A4-035, A4-036; other: KR-19 | - | FC-2 production wiring: capacity, queue, memory, observability, sandbox. cond tests: the 5 named census candidates and test_platform_configuration_reference.py as the plan states; the 22 census files of RWD14-01 only as an explicit typed opt-in confirmed by the step-7 census and listed in the PR body; export_boundary.py only if section 4.7 shows observability_otlp.py alone cannot break the KR-19 cycle; v15 (RWD14-01): the 22 test files of the plan's static census (P intersect C of w3-packet-2/043-B/work/_P.txt and _C.txt, 32 files, section 8a; work/fc2_census.json, FACT_CHECK F-77) that lie outside the v14 scope are cond, census-confirmed opt-in only; the 4 non-test census files, any other file and every applications/** path (applications/*/tests/ included) are a stop and an amendment (Q-B-02, Operator). v14: re-extracted from the w3-packet-2 draft plan (digest 14e4de95...daff, INPUT_043-B.md): KR-19 is in the plan (observability_otlp.py firm, export_boundary.py cond, the KR-19 test firm); tests/unit/docs/test_platform_configuration_reference.py is cond as the plan states (Q-B-01 draft default, orchestrator may reverse to firm); counts 10 / 23 / 7. v10 (RWD9-02): the 2 named edited tests (plan :100) are firm; the 5 census candidates are cond, found by the census step (plus the PRODUCT host suites under applications/*/tests/, size UNVERIFIED in the plan). KR-19 (import cycle observability_otlp -> export_boundary) added in v2 (Q-RW-10) |
| T04 | PT-FC4 | T-IGX-043-D | PRODUCT | 2 | 6 / 12 / 0 | finding: A4-002 | - | FC-4 skill-pin durability and KV durability attestation. v14: re-extracted from the w3-packet-2 draft plan (digest 2dfe0ad9...dd30); path set and classes equal to v13. |
| T05 | T-AF2-DEPS-BATCH | T-IGX-051-H | NONPRODUCT | 2 | 0 / 4 / 0 | FRZ: FRZ-DEBT-02, FRZ-DEBT-03; finding: A4-005 | - | Held dependency-bot / uv.lock batch; dependency-ownership gate (A4-005). v2 (Q-RW-09): batch = the dependency-bot PRs open when arch-freeze/rw-train is cut (at drafting: #631, aiohttp 3.13.3 -> 3.14.4) plus the A4-005 ownership declarations; re-listed at the train cut |
| T06 | PT-E05 | T-IGX-044-F | MATERIAL | 2 | 0 / 6 / 0 | finding: A2-038 | - | T-ENF-5 IIAC validator hardening (material governance paths). v14: re-extracted from the w3-packet-2 draft plan (digest 531ef8e1...fb0e); path set and classes equal to v13. |
| T07 | PT-D08 | T-IGX-046-D | PRODUCT | 2 | 2 / 2 / 0 | finding: A2-012, A4-041 | - | DB-H product-path docs (2 files counted as product paths). v14: re-extracted from the w3-packet-2 draft plan (digest ddc1b231...f424); path set and classes equal to v13. |
| T08 | T-AF2-RW-RUNTIME | T-IGX-051-D | PRODUCT | 3 | 9 / 17 / 0 | EX: EX-D3-01, EX-D3-03, EX-L-07, EX-L-08; PARTIAL: PDC-D3-R005, PDC-D3-R019; FRZ: FRZ-CTR-01, FRZ-CTR-02, FRZ-CTR-06; finding: A4-018 | FRZ-CFG-02, FRZ-CFG-06, FRZ-SEC-04, PDC-D3-R018 | LLM runtime: stream-open errors through breaker/retry, timeout, typed error contract v1, served model, refusal matrix, SDK boundary |
| T09 | PT-WD | T-IGX-043-I | PRODUCT | 3 | 9 / 30 / 8 | finding: A6-003, A6-005, A6-007, A6-008, X-A6-003 | - | W-D ENT-AUDIT-X engine fixes (CHANGED-SINCE default-deny, FRZ attribution, FAST disclosure). v14: re-extracted from the w3-packet-2 draft plan (digest 79b760a7...abc7); path set and classes equal to v13. v2 (Q-RW-04): the Q-WD-07 option B set is binding (PT-C01 mapped ENT checks before G2); tests/qualification/ent_audit_x/ENTERPRISE_AUDIT_BASELINE.json is NOT in the write set (the accepted baseline is T-AF2-ENT-BASELINE-PREFREEZE, W5); BASELINE:MANIFEST_DIGEST is expected-red with owner PT-WD. W-D changes what the audit checks (gate-meaning); approval route per OD-AF-22 (wave packet approved by the Operator, Approved line by the signing delegate with a Delegation pointer). |
| T10 | PT-FC1 | T-IGX-043-A | PRODUCT | 3 | 35 / 148 / 24 | FRZ: FRZ-BND-05, FRZ-EXE-02, FRZ-PLG-05; finding: A4-009, A4-015, A4-031, A4-037, A4-044, X-A4-024 | FRZ-OWN-03, FRZ-OWN-05, FRZ-PRD-05 | FC-1 execution routing and Nexus construction. 17 conditional product paths: the two B6 forwarders, tool_wiring.py (B4), scenario_runtime_baseline.py (B9, Q-FC1-05) and the 13 build_harness_host_runtime call-site files, each edited only if step 0 shows the call needs the new keywords; 7 conditional tests: only if a source-token or call-shape assertion sees the FC-1 edit; Part C1 (4 firm paths) is dropped if the step-0 replay of V2 at the train head is negative; v14: re-extracted from the w3-packet-2 draft plan (digest f9c0b109...1147, INPUT_043-A.md); the plan now enumerates the whole write set: 35 firm product paths (Part C1 firm by the draft default of Q-FC1-01; intergrax/agents/defaults.py replaces the non-existent agents/defaults.py; the Tier-3 spec builder intergrax/applications/_shared/host_orchestration_backend_spec_builder.py added after PT-WA's split), 17 conditional product paths (inside the Operator's ceiling of up to 21 conditional product paths, transcription item 2), 109 firm tests and support files (the 100 sweep files equal the v11 census work/fc1_sweep_census.json exactly, so the v11 hand list is retired), 7 conditional tests, 4 docs. Approval of the old digest b2708519...1922 does not cover the new one |
| T11 | PT-FC2-E | T-IGX-043-E | NONPRODUCT | 3 | 0 / 2 / 0 | - | FRZ-PRD-03, FRZ-REL-07, FRZ-REL-09, FRZ-REL-10 | FC-2 gate scripts (capacity adapters, sandbox policy wiring); gate-meaning change, Operator personal. v14: re-extracted from the w3-packet-2 draft plan (digest b641f93f...1f53); path set and classes equal to v13. |
| T12 | T-AF2-RW-RECOVERY | T-IGX-051-E | PRODUCT | 3 | 12 / 17 / 0 | EX: EX-D1-07, EX-D4-01, EX-D7-01, EX-D7-03, EX-L-03, EX-L-06; PARTIAL: PDC-D7-R015; other: G-20 | - | Background and recovery: drain controller, wait_for_idle, CAS recovery, durable CAS, UNCERTAIN reconciler, capped retry, producer identity |
| T13 | T-AF2-A4007-CALLERS | T-IGX-051-I | PRODUCT | 4 | 5 / 10 / 2 | finding: A4-007 | - | A4-007 seams b1/b2: product callers of the import allowlist and fail-closed parameters (after PT-FC2 wiring_posture.py). intergrax/applications/_shared/package_wiring.py only if Q-A7C-02 is ruled in (then NT-A7C-08 is firm); intergrax/applications/_shared/environment_diff_wiring.py only if Q-A7C-04 is ruled for uniform threading (draft: not edited, control CTL-A7C-03); v16: new train ticket (orchestrator ruling option b: the 043-B digest 14e4de95... stays unchanged and the callers go here). Draft: .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md, digest 2add59746348470889b69598bda2498f8e022cd7e78fe97b1f8b099c7ad5a244 (v17, corrected after RWD16-01/02; v16 read 1724042a...c48a; INPUT_T-IGX-051-I.md, FACT_CHECK F-01..F-42); write set from plan section 7 (table lines 98-107, conditional line 109; count line :94 'product 4 edited + 1 new, scripts 2, tests 3 new'), every path checked at origin/arch-freeze/rw-train 6b94d202fb79a0611404f8d83d14505c6d7cb0c5 (FACT_CHECK F-79): 7 EXISTS, 3 new with an existing parent. Firm 10 = 5 product (import_policy.py new; agent_resolution.py, wiring.py, environment_conformance.py, roster_agent_contract_authority.py) + 2 scripts + 3 new tests; cond 2. Owns A4-007 (b1/b2 callers); PT-WB contributed the loaders, the allowlist module and b4 (merged). The two script edits are not gate-meaning by draft default (Q-A7C-06, Operator) |
| T14 | PT-FC3 | T-IGX-043-C | PRODUCT | 4 | 62 / 77 / 76 | FRZ: FRZ-CFG-02, FRZ-CFG-06, FRZ-SEC-04; finding: A4-017, A4-026 | A4-019, A4-032, FRZ-PRD-03, FRZ-REL-07, FRZ-REL-09, FRZ-REL-10 | FC-3 vendor/provider selector fail-closed. v18: re-extracted from the w3-packet-4 draft plan (digest 5d4f0ac3...7506, revision 5, INPUT_043-C.md): path set equal to v17 (62 / 77 / 0); the 76 census test files of plan section 7 outside the write set (52 + 33 bounded census, work/fc3_census.json, F-80) are cond, census-confirmed opt-in only (RWD14-01 rule); any other file is a stop and an amendment (Q-C-01, Operator). v10 (RWD9-02): the 9 'inverted or updated' tests of plan :182 are firm (full paths at HEAD); the remaining default-dependent tests are found by the census in plan section 8 step 3 |
| T15 | T-AF2-RW-SUPPLY | T-IGX-051-B | PRODUCT | 4 | 20 / 31 / 2 | EX: EX-D1-01, EX-D1-02, EX-D1-04, EX-D6-01, EX-D6-02, EX-D6-03, EX-D8-01; PARTIAL: PDC-D1-R017, PDC-D3-R018, PDC-D6-R007, PDC-D6-R010; FRZ: FRZ-SEC-06; other: A-01, F-D3-12, F-D5-12, F-D6-02, F-D6-03, F-D6-04, F-D6-09, F-D6-10, F-D6-11, F-D6-14, F-D8-14 | A4-040, FRZ-EXE-02, FRZ-PLG-05 | Supply chain and security: attestation, revocation, trust hook order, digests, fail-open set, strict validator, secrets, evidence sink, retention |
| T16 | T-AF2-RW-GATES | T-IGX-051-F | PRODUCT | 4 | 8 / 28 / 7 | EX: EX-D1-03, EX-D1-05, EX-D1-06, EX-D2-03, EX-D3-02, EX-D3-04, EX-D4-05, EX-D5-02, EX-D5-03, EX-D7-02, EX-D8-02, EX-L-01, EX-L-04, EX-L-10, EX-L-11, EX-L-13, EX-L-14; PARTIAL: PDC-D1-R015, PDC-D5-R002, PDC-D6-R015; FRZ: FRZ-BND-06, FRZ-DEBT-04, FRZ-OWN-02, FRZ-PLG-01, FRZ-PLG-02, FRZ-PLG-06, FRZ-PLG-07, FRZ-PLG-08, FRZ-PRD-08, FRZ-REG-09; other: KR-03, KR-04, KR-07, KR-17 | FRZ-OWN-03, FRZ-OWN-05, FRZ-PRD-05 | Gates and contracts: promotion gate binding, canonical get_steps, event schema, eval digest, NOT_COMPOSED gates, inventory gates |
| T17 | T-AF2-RW-TRUST | T-IGX-051-C | PRODUCT | 5 | 7 / 14 / 12 | EX: EX-D2-01, EX-D2-02, EX-D2-05, EX-D2-07, EX-L-05; PARTIAL: PDC-D2-R004 | - | Trust boundary: trust classes, no untrusted role=system, egress default, orphan tool_call_id, planner ledger, RAG reds. the 12 '[context:' test modules: expectation-only edit (the pinned role or prefix), census-confirmed at step 0, no assertion weakened; v14 Q-TR-01: intergrax/context/formatter.py firm. v15 (RWD14-02): the 12 test modules that contain '[context:' (TRUST draft FACT_CHECK F-21; git grep at the facts SHA, F-78, work/trust_context_census.json) are cond; the planner and engine tests that may pin the changed behaviour are not enumerated: each one found at step 0 is a stop and an amendment |
| T18 | PT-FC3-F | T-IGX-043-F | NONPRODUCT | 5 | 0 / 2 / 0 | FRZ: FRZ-CFG-08 | A4-019, A4-025 | CONFIG-X vendor-slug default scan extension (gate-meaning change, Operator personal). v18: re-extracted from the w3-packet-4 draft plan (digest 903e830d...d202, revision 4): 0 / 2 / 0, 12 negative tests NT-FC3-CX1..12; per the Operator ruling (option 3a) only hits on the eight FC-3 paths fail the gate, hits elsewhere are recorded (Q-F-3); T-IGX-043-F does not close A4-019 (Q-F-2) and does not deliver A4-025 (both open Operator items). |
| T19 | PT-WF | T-IGX-043-J | PRODUCT | 6 | 41 / 51 / 10 | FRZ: FRZ-RPL-04, FRZ-TEN-01, FRZ-TEN-10, FRZ-TYP-01, FRZ-TYP-03; finding: A4-027, A8-020, X-A5-011, X-ESC-E1, X-ESC-E2, X-ESC-E4, X-ESC-E5 | EX-L-09 | W-F tenant G-09 fail-closed, typed seams, B-1 confirmed items. Part JB (8 product paths, firm after Q-WF-03/Q-WF-04) includes the inversion of the two EBH-6 Q2-D2 gate files, a gate-meaning change that is the Operator's own decision (Q-WF-04, not covered by the standing rules OD-AF-23/S3): Part JB does not merge without that decision; Parts JA, JC and JG can proceed. Q-WF-01 accepted: one ticket as in A1, parts separately cuttable. v18 decision on tests/qualification/ebh_6/_contracts_typing_qualification.py (plan :273-277, scan hit S-1 at plan :452): it belongs in the write set as cond. The EBH-6 typing debt registry pins harness_auth.py line 137 and the gate compares violation.line == entry.line, so a JG-5a edit above that line turns the gate red unless the registry line moves with it. The plan's default keeps line 137 unmoved (no edit); the registry edit is needed only if the line moves. A path the ticket may have to edit is listed, as cond, instead of being an unlisted edit at implementation time. v18: re-extracted from the w3-packet-4 draft plan (digest c78454ba...8ab8): 41 / 51 / 10; one cond path added against A1, tests/qualification/ebh_6/_contracts_typing_qualification.py (scan hit S-1, a one-line data edit only if harness_auth.py line 137 moves, Q-WF-17).; Part JF (A4-018) moves to T-AF2-RW-RUNTIME (dedupe D-05); Part JD stays held and is not scope (D-09) |
| T20 | T-AF2-RW-TENANT | T-IGX-051-A | PRODUCT | 7 | 19 / 32 / 42 | EX: EX-D2-06, EX-D4-04, EX-D5-05, EX-D8-03, EX-D8-04, EX-L-02, EX-L-09; other: F-D2-11, G-12, R-P8-01 | FRZ-CFG-02, FRZ-CFG-06, FRZ-SEC-04, FRZ-TEN-01, FRZ-TEN-10 | Tenant binding: sandbox sessions, object keys, tool context, artifact read, observation tenant, install CAS, MCP tenant, unscoped memory. cond paths: GAP-01 only if Q-TEN-03 is ruled option B (token on the request); GAP-02 and GAP-05 only if a cross-process compare or the backend contract is ruled into this ticket (draft: limits recorded, not in this ticket); GAP-04 only if Q-TEN-02 is ruled option 1 (required tenant_id); the CTRL-X catalog note only if its text pins the absence of tenant_id; each listed test only if the step-0 census shows the change turns it red (install_agent files: only those that install twice into one occupied slot); v14 Q-TEN-01 (orchestrator ruling: amend A1 with the extra paths and pinned tests of draft plan sections 4.1 and 4.2, w3-drafts/rw-tenant, digest 77e5d3db...32d2): firm GAP-06 (3 MCP composition call sites), GAP-07 (2 reindex forwarders), GAP-03 (the CTRL-X test that pins the absence of tenant_id) and the 4 storage-tool tests whose key contract changes; cond the option-dependent paths (GAP-01 admin_models.py for Q-TEN-03 option B; GAP-02 the 3 installation stores for a cross-process compare; GAP-04 the 4 files of Q-TEN-02 option 1; GAP-05 sandbox_host.py), the CTRL-X catalog note, the 5 reindex and 2 memory tests (expected unchanged), and the 25 install_agent( test files (census work/tenant_install_census.json, FACT_CHECK F-75). GAP-05 provider backends and GAP-08 (audit adapter home) name no path: step 0 decides, a need is a stop and an amendment. The 2 MCP tests of 4.2 stay green by design and are regression only |
| T21 | T-AF2-RW-TYP | T-IGX-051-G | PRODUCT | 7 | 2 / 6 / 0 | FRZ: FRZ-TYP-02, FRZ-TYP-04, FRZ-TYP-05 | FRZ-RPL-04, FRZ-TYP-01, FRZ-TYP-03 | Typing ratchet on frozen contract surfaces (TYP-05; TYP-02/04 per OD-AF-11). v8 (OD-AF-11; RWD7-01..03): ratchet gate on the frozen contract surfaces intergrax/contracts/** over Any, object annotations and the reflection metrics getattr_dynamic, hasattr, setattr, importlib, __import__, against ONE baseline: the HEAD measurement (138 Any, 357 object, getattr_dynamic 1, hasattr 3, setattr 0, importlib 0, __import__ 0; RATCHET_BASELINE); exceptions only as entries of typ_ratchet_baseline.v1.json with ticket and reason. TYP-05 reduction: the 5 module-level string-keyed lookup tables found by the defined detector (TYP05_REDUCTION) go to 0, each replaced by a typed key (enum or typed key object) or listed as a ratified exception with its reason; the 2 files holding them are the only product write scope. The rest of the platform is outside the freeze scope, stated in the freeze documents (T-AF2-MANIFESTS) |
| T22 | PT-T01 | T-IGX-043-L | NONPRODUCT | 8 | 0 / 5 / 0 | finding: A4-006 | - | Campaign B tracked negative/adversarial tests NT-01..NT-16 + tenant-propagation emitter repair (A4-006). v18 ruling (FOLLOW_UPS.md item 2): NT-07 (A2-049, held file tests/qualification/trace_x/test_t01_nt07_p6_classifier_new_file.py) and NT-04b (detector scope of A4-027) have no fixer in the train and are held out of the merge set; their code under test is test support outside A1 (tests/qualification/trace_x/_trace_x_p6_restart_resume_classification.py, tests/qualification/tenant_x/_r1_silent_substitution.py); the 'green when merged' rule covers only the NTs with a train fixer. v18: re-extracted from the w3-packet-4 draft plan (digest df3d2276...7471): equal to A1 (0 / 5 / 0); NT-07 and NT-04b are HELD (no fixer in the train, Q-T01-01). |

Ticket letters of PT-WE, PT-T01, PT-D08 and PT-E05 are reserved by the EQO-A answer (v2); the tickets are created when their nodes start. T-IGX-051-A..H follow OD-AF-01a.

**Approval route (OD-AF-22 (OPERATOR_PACKET_3, 2026-11-17)).** Applies to every train ticket: T-IGX-051-A..H and the EQO-A T-IGX-043-*, T-IGX-044-F, T-IGX-046-D tickets, including gate-meaning PT-WD and IIAC-material PT-E05. The orchestrator prepares one approval packet per train lane or wave (each ticket, plan file and plan digest); the Operator approves the packet once; the signing delegate then writes each Approved: line with a Delegation: pointer to the packet record. Train PRs merge into arch-freeze/rw-train after independent review (OD-AF-02). The merge of the train into development (FCS) stays the Operator's decision. The the train branch may be cut once the first wave packet (PT-WA) is approved.

## (b) File-scope overlap matrix

Cell = number of shared paths (firm and conditional scope, globs expanded over the tree at HEAD; a new path of one ticket under a glob of the other counts). Blank = disjoint. The shared paths follow the matrix.

| | T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08 | T09 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 | T20 | T21 | T22 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **T01** PT-WA | · |  | 1 |  |  |  |  |  |  | 7 |  |  |  | 1 |  | 1 |  |  | 1 | 1 |  |  |
| **T02** PT-WB |  | · |  |  |  |  |  |  |  |  |  |  | 2 |  | 1 |  |  |  | 1 |  |  |  |
| **T03** PT-FC2 | 1 |  | · |  |  |  |  |  |  | 1 |  | 1 |  |  |  | 1 |  |  |  | 4 |  |  |
| **T04** PT-FC4 |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T05** T-AF2-DEPS-BATCH |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T06** PT-E05 |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T07** PT-D08 |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T08** T-AF2-RW-RUNTIME |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T09** PT-WD |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T10** PT-FC1 | 7 |  | 1 |  |  |  |  |  |  | · |  |  |  | 16 |  | 1 |  |  | 6 | 2 |  |  |
| **T11** PT-FC2-E |  |  |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |
| **T12** T-AF2-RW-RECOVERY |  |  | 1 |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |
| **T13** T-AF2-A4007-CALLERS |  | 2 |  |  |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |
| **T14** PT-FC3 | 1 |  |  |  |  |  |  |  |  | 16 |  |  |  | · |  |  | 11 |  | 6 | 1 |  |  |
| **T15** T-AF2-RW-SUPPLY |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  | · |  |  |  |  | 2 |  |  |
| **T16** T-AF2-RW-GATES | 1 |  | 1 |  |  |  |  |  |  | 1 |  |  |  |  |  | · |  |  |  |  |  |  |
| **T17** T-AF2-RW-TRUST |  |  |  |  |  |  |  |  |  |  |  |  |  | 11 |  |  | · |  |  | 3 |  |  |
| **T18** PT-FC3-F |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | · |  |  |  |  |
| **T19** PT-WF | 1 | 1 |  |  |  |  |  |  |  | 6 |  |  |  | 6 |  |  |  |  | · | 2 |  |  |
| **T20** T-AF2-RW-TENANT | 1 |  | 4 |  |  |  |  |  |  | 2 |  |  |  | 1 | 2 |  | 3 |  | 2 | · |  |  |
| **T21** T-AF2-RW-TYP |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | · |  |
| **T22** PT-T01 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | · |

| Pair | Shared paths |
| --- | --- |
| PT-FC1 × PT-FC2 | tests/unit/applications/architecture/test_diag_platform_adoption_gate.py |
| PT-FC1 × PT-FC3 | applications/dispute_sim_application/host/factory.py; applications/governed_contractor_application/host/factory.py; applications/intergrax_assistant_application/host/factory.py; applications/lab_application/host/factory.py; applications/legal_application/host/factory.py; applications/local_workspace_application/host/host_runtime_composition.py; applications/poc_template_application/host/factory.py; applications/research_application/host/factory.py; intergrax/scaffold/new_application.py; intergrax/scaffold/new_application_product.py; tests/integration/runtime/test_terminal_diagnostic_production_e2e.py; tests/unit/applications/test_harness_guardrail_wiring.py; tests/unit/applications/test_mcp_server_p4_lifecycle.py; tests/unit/applications/test_reference_apps_mcp.py; tests/unit/debug/test_h3_interaction_intake.py; tests/unit/debug/test_h5_teams_interaction_intake.py |
| PT-FC1 × PT-WF | applications/attestation_demo/host/factory.py; applications/dispute_sim_application/host/factory.py; applications/legal_application/host/factory.py; applications/research_application/host/factory.py; intergrax/runtime/nexus/nexus_loop.py; intergrax/scaffold/new_application_product.py |
| PT-FC1 × T-AF2-RW-GATES | intergrax/applications/_shared/scenario_runtime_baseline.py |
| PT-FC1 × T-AF2-RW-TENANT | intergrax/applications/_shared/harness_host_runtime.py; intergrax/runtime/nexus/nexus_loop.py |
| PT-FC2 × T-AF2-RW-GATES | tests/qualification/plug_03/test_plug_03_gates.py |
| PT-FC2 × T-AF2-RW-RECOVERY | intergrax/applications/_shared/queue_worker_wiring.py |
| PT-FC2 × T-AF2-RW-TENANT | intergrax/applications/_shared/memory_provider_admission.py; intergrax/applications/_shared/memory_wiring.py; tests/unit/applications/test_mem_audit5a_production_provider_admission.py; tests/unit/applications/test_memory_wiring.py |
| PT-FC3 × PT-WF | applications/dispute_sim_application/host/factory.py; applications/legal_application/host/factory.py; applications/local_workspace_application/host/factory.py; applications/research_application/host/factory.py; intergrax/debug/app.py; intergrax/scaffold/new_application_product.py |
| PT-FC3 × T-AF2-RW-TENANT | intergrax/applications/_shared/environment_wiring.py |
| PT-FC3 × T-AF2-RW-TRUST | tests/integration/rag/rerankers/test_embedding_cosine_reranker.py; tests/unit/rag/bootstrap/test_rag_entry_point_load.py; tests/unit/rag/document_splitters/test_optional_langchain_splitter.py; tests/unit/rag/embedding/test_embedding_profile.py; tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py (new; under tests/unit/rag/*); tests/unit/rag/embedding/test_llama_cpp_embedding_provider.py; tests/unit/rag/embedding/test_p2_002_b3_embedding_runtime_binding.py; tests/unit/rag/embedding/test_vllm_embedding_provider.py; tests/unit/rag/vectorstore/test_fc3_vectorstore_selector_fail_closed.py (new; under tests/unit/rag/*); tests/unit/rag/vectorstore/test_integration_vectorstore_bootstrap.py; tests/unit/rag/vectorstore/test_integration_vectorstore_scope.py |
| PT-WA × PT-FC1 | intergrax/applications/_shared/harness_host_runtime.py; intergrax/applications/_shared/host_orchestration_backend_spec_builder.py; intergrax/applications/_shared/scenario_runtime_baseline.py; intergrax/runtime/execution/host_orchestration_environment_spec_builder.py; intergrax/runtime/execution/host_orchestration_loop_init_spec.py; tests/unit/applications/_shared/test_obs_diag_x3_universal_spine_adoption.py; tests/unit/applications/architecture/test_diag_platform_adoption_gate.py |
| PT-WA × PT-FC2 | tests/unit/applications/architecture/test_diag_platform_adoption_gate.py |
| PT-WA × PT-FC3 | applications/local_workspace_application/host/factory.py |
| PT-WA × PT-WF | applications/local_workspace_application/host/factory.py |
| PT-WA × T-AF2-RW-GATES | intergrax/applications/_shared/scenario_runtime_baseline.py |
| PT-WA × T-AF2-RW-TENANT | intergrax/applications/_shared/harness_host_runtime.py |
| PT-WB × PT-WF | applications/research_application/host/settings.py |
| PT-WB × T-AF2-A4007-CALLERS | intergrax/applications/_shared/agent_resolution.py; intergrax/applications/_shared/wiring.py |
| PT-WB × T-AF2-RW-SUPPLY | applications/governed_contractor_application/host/package_b_strict_production_serving_command.py |
| PT-WF × T-AF2-RW-TENANT | intergrax/applications/_shared/policy_wiring.py; intergrax/runtime/nexus/nexus_loop.py |
| T-AF2-RW-TENANT × T-AF2-RW-SUPPLY | intergrax/agent_distribution/admin_service.py; intergrax/runtime/architecture/online_evaluation_registry.py |
| T-AF2-RW-TENANT × T-AF2-RW-TRUST | tests/unit/rag/governance/test_embedding_version_policy.py; tests/unit/rag/governance/test_rw_tenant_reindex_hook_scope.py (new; under tests/unit/rag/*); tests/unit/rag/ingest/test_ingest_embedding_version_policy.py |

Known overlaps named by plan section G are reproduced: `harness_host_runtime.py` (PT-WA, PT-FC1, T-AF2-RW-TENANT; PT-WF does not edit it, WF plan 7.3) and `applications/_shared/*_wiring.py` (PT-FC2 with RW-TENANT on `memory_wiring.py`/`memory_provider_admission.py` and with RW-RECOVERY on `queue_worker_wiring.py`). PT-WB and RW-SUPPLY share no `_wiring.py` file; they share the strict serving command (W-B Part G, Q-RW-15).

## (c) Merge order and parallel lanes

Rules applied (plan G, K): PT-WA alone first; PT-WB alone second; then at most 5 tickets in parallel (also the K cap of 5 implementers); tickets that share a path never run in the same lane; the listed hard precedences hold. Every train PR merges into `arch-freeze/rw-train` with a merge commit, after its head SHA equals the pushed SHA and every red job, step or node of its CI is either an entry of table 1 below or a planned entry of table 2 that has been registered at the moment stated there.

**Start rule (binding, RWA-07).** At most 5 train tickets are in flight at any time (plan G step 3, K cap of 5 implementers); PT-WA and PT-WB run alone. Lanes are waves: a ticket starts only when (a) every predecessor it names has merged into arch-freeze/rw-train and (b) a slot is free. A ticket placed in a later lane for capacity ('lane N full') starts only when a slot frees, in lane order and, inside a lane, in the order of the lane table.

| Lane | Tickets | Kind |
| ---: | --- | --- |
| 0 | PT-WA | alone |
| 1 | PT-WB | alone |
| 2 | PT-FC2, PT-FC4, T-AF2-DEPS-BATCH, PT-E05, PT-D08 | parallel |
| 3 | T-AF2-RW-RUNTIME, PT-WD, PT-FC1, PT-FC2-E, T-AF2-RW-RECOVERY | parallel |
| 4 | T-AF2-A4007-CALLERS, PT-FC3, T-AF2-RW-SUPPLY, T-AF2-RW-GATES | parallel |
| 5 | T-AF2-RW-TRUST, PT-FC3-F | parallel |
| 6 | PT-WF | parallel |
| 7 | T-AF2-RW-TENANT, T-AF2-RW-TYP | parallel |
| 8 | PT-T01 | parallel |

Serialization reasons (every overlap and precedence that sets a ticket's lane):

| Ticket | Lane | After (reason) |
| --- | ---: | --- |
| PT-WA | 0 | - |
| PT-WB | 1 | PT-WA (plan G step 1) |
| PT-FC2 | 2 | PT-WA, PT-WB (plan G steps 1-2); after PT-WA: file overlap (1 shared: tests/unit/applications/architecture/test_diag_platform_adoption_gate.py) |
| PT-FC4 | 2 | PT-WA, PT-WB (plan G steps 1-2) |
| T-AF2-DEPS-BATCH | 2 | PT-WA, PT-WB (plan G steps 1-2) |
| PT-E05 | 2 | PT-WA, PT-WB (plan G steps 1-2) |
| PT-D08 | 2 | PT-WA, PT-WB (plan G steps 1-2) |
| T-AF2-RW-RUNTIME | 3 | PT-WA, PT-WB (plan G steps 1-2); lane 2 full (max 5 parallel tickets); placed in lane 3 |
| PT-WD | 3 | PT-WA, PT-WB (plan G steps 1-2); lane 2 full (max 5 parallel tickets); placed in lane 3 |
| PT-FC1 | 3 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: Operator merge order B, A, C (FC/OPERATOR_ANSWERS_TRANSCRIPTION.md item 6); after PT-FC2: file overlap (1 shared: tests/unit/applications/architecture/test_diag_platform_adoption_gate.py); after PT-WA: file overlap (7 shared: intergrax/applications/_shared/harness_host_runtime.py, intergrax/applications/_shared/host_orchestration_backend_spec_builder.py, intergrax/applications/_shared/scenario_runtime_baseline.py ...) |
| PT-FC2-E | 3 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: gate scripts merge after the FC-2 fix (gate would be red before the fix) |
| T-AF2-RW-RECOVERY | 3 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: plan Z: serialized with PT-FC2 on queue wiring; after PT-FC2: file overlap (1 shared: intergrax/applications/_shared/queue_worker_wiring.py) |
| T-AF2-A4007-CALLERS | 4 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: T-IGX-051-I imports wiring_posture.wiring_fails_closed, created by PT-FC2 (draft plan section 9); after PT-WB: file overlap (2 shared: intergrax/applications/_shared/agent_resolution.py, intergrax/applications/_shared/wiring.py); lane 3 full (max 5 parallel tickets); placed in lane 4 |
| PT-FC3 | 4 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC1: Operator merge order B, A, C; after PT-FC2: PT-FC3 imports wiring_posture.py created by PT-FC2; after PT-WA: file overlap (1 shared: applications/local_workspace_application/host/factory.py); after PT-FC1: file overlap (16 shared: applications/dispute_sim_application/host/factory.py, applications/governed_contractor_application/host/factory.py, applications/intergrax_assistant_application/host/factory.py ...) |
| T-AF2-RW-SUPPLY | 4 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: plan Z: serialized with PT-FC2/PT-WB on applications/_shared wiring; after PT-WB: file overlap (1 shared: applications/governed_contractor_application/host/package_b_strict_production_serving_command.py); lane 3 full (max 5 parallel tickets); placed in lane 4 |
| T-AF2-RW-GATES | 4 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: file overlap (1 shared: tests/qualification/plug_03/test_plug_03_gates.py); after PT-WA: file overlap (1 shared: intergrax/applications/_shared/scenario_runtime_baseline.py); after PT-FC1: file overlap (1 shared: intergrax/applications/_shared/scenario_runtime_baseline.py) |
| T-AF2-RW-TRUST | 5 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC3: file overlap (11 shared: tests/integration/rag/rerankers/test_embedding_cosine_reranker.py, tests/unit/rag/bootstrap/test_rag_entry_point_load.py, tests/unit/rag/document_splitters/test_optional_langchain_splitter.py ...) |
| PT-FC3-F | 5 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC3: CONFIG-X extension merges after FC-3 (gate would be red before the fix) |
| PT-WF | 6 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: PT-WF imports wiring_posture.py created by PT-FC2; after PT-FC3-F: proposed order W-A/W-B, B, A, C, F, then J (WF plan 7.3, Q-WF-12); after PT-FC3: file overlap (6 shared: applications/dispute_sim_application/host/factory.py, applications/legal_application/host/factory.py, applications/local_workspace_application/host/factory.py ...); after PT-WB: file overlap (1 shared: applications/research_application/host/settings.py); after PT-WA: file overlap (1 shared: applications/local_workspace_application/host/factory.py); after PT-FC1: file overlap (6 shared: applications/attestation_demo/host/factory.py, applications/dispute_sim_application/host/factory.py, applications/legal_application/host/factory.py ...) |
| T-AF2-RW-TENANT | 7 | PT-WA, PT-WB (plan G steps 1-2); after PT-WF: plan Z: RW-TENANT serialized with PT-FC1/PT-WF on tenant wiring; after PT-FC1: plan Z: harness_host_runtime.py; after T-AF2-RW-TRUST: file overlap (3 shared: tests/unit/rag/governance/test_embedding_version_policy.py, tests/unit/rag/governance/test_rw_tenant_reindex_hook_scope.py (new; under tests/unit/rag/*), tests/unit/rag/ingest/test_ingest_embedding_version_policy.py); after PT-FC3: file overlap (1 shared: intergrax/applications/_shared/environment_wiring.py); after PT-WF: file overlap (2 shared: intergrax/applications/_shared/policy_wiring.py, intergrax/runtime/nexus/nexus_loop.py); after PT-FC2: file overlap (4 shared: intergrax/applications/_shared/memory_provider_admission.py, intergrax/applications/_shared/memory_wiring.py, tests/unit/applications/test_mem_audit5a_production_provider_admission.py ...); after PT-WA: file overlap (1 shared: intergrax/applications/_shared/harness_host_runtime.py); after T-AF2-RW-SUPPLY: file overlap (2 shared: intergrax/agent_distribution/admin_service.py, intergrax/runtime/architecture/online_evaluation_registry.py); after PT-FC1: file overlap (2 shared: intergrax/applications/_shared/harness_host_runtime.py, intergrax/runtime/nexus/nexus_loop.py) |
| T-AF2-RW-TYP | 7 | PT-WA, PT-WB (plan G steps 1-2); after PT-WF: RW-TYP merges after the last train ticket that edits intergrax/contracts/** (W-F JB typed seams) so that the gate's first run covers those edits; the baseline stays the HEAD measurement (RWD7-02); after T-AF2-RW-SUPPLY: RW-TYP merges after the last train ticket that edits intergrax/contracts/** (RW-SUPPLY edits agent_run.py) so that the gate's first run covers those edits; the baseline stays the HEAD measurement (RWD7-02) |
| PT-T01 | 8 | PT-WA, PT-WB (plan G steps 1-2); after T-AF2-RW-TENANT: the NT-01..16 nodes that have a train fixer must be green when merged (no expected-red test without a ruling); NT-07 and NT-04b have no fixer and are held out of the merge set (v18 ruling); after PT-WF: NT-01..16 cover W-F defects; after PT-WD: NT-08 covers A6-007 |

Hard precedences in the schedule:

- PT-WA -> *: plan G order step 1: PT-WA alone first (largest blast radius)
- PT-WB -> *: plan G order step 2: PT-WB second
- PT-FC2 -> PT-FC1: Operator merge order B, A, C (FC/OPERATOR_ANSWERS_TRANSCRIPTION.md item 6)
- PT-FC1 -> PT-FC3: Operator merge order B, A, C
- PT-FC2 -> PT-FC3: PT-FC3 imports wiring_posture.py created by PT-FC2
- PT-FC2 -> PT-WF: PT-WF imports wiring_posture.py created by PT-FC2
- PT-FC3 -> PT-FC3-F: CONFIG-X extension merges after FC-3 (gate would be red before the fix)
- PT-FC2 -> PT-FC2-E: gate scripts merge after the FC-2 fix (gate would be red before the fix)
- PT-FC3-F -> PT-WF: proposed order W-A/W-B, B, A, C, F, then J (WF plan 7.3, Q-WF-12)
- PT-WF -> T-AF2-RW-TENANT: plan Z: RW-TENANT serialized with PT-FC1/PT-WF on tenant wiring
- PT-FC1 -> T-AF2-RW-TENANT: plan Z: harness_host_runtime.py
- PT-FC2 -> T-AF2-RW-SUPPLY: plan Z: serialized with PT-FC2/PT-WB on applications/_shared wiring
- PT-FC2 -> T-AF2-RW-RECOVERY: plan Z: serialized with PT-FC2 on queue wiring
- PT-FC2 -> T-AF2-A4007-CALLERS: T-IGX-051-I imports wiring_posture.wiring_fails_closed, created by PT-FC2 (draft plan section 9)
- T-AF2-RW-TENANT -> PT-T01: the NT-01..16 nodes that have a train fixer must be green when merged (no expected-red test without a ruling); NT-07 and NT-04b have no fixer and are held out of the merge set (v18 ruling)
- PT-WF -> PT-T01: NT-01..16 cover W-F defects
- PT-WD -> PT-T01: NT-08 covers A6-007
- PT-WF -> T-AF2-RW-TYP: RW-TYP merges after the last train ticket that edits intergrax/contracts/** (W-F JB typed seams) so that the gate's first run covers those edits; the baseline stays the HEAD measurement (RWD7-02)
- T-AF2-RW-SUPPLY -> T-AF2-RW-TYP: RW-TYP merges after the last train ticket that edits intergrax/contracts/** (RW-SUPPLY edits agent_run.py) so that the gate's first run covers those edits; the baseline stays the HEAD measurement (RWD7-02)

External precedence from train tickets into non-train nodes (RWA-11):

- PT-WA merged into the train -> T-AF2-MANIFESTS (final, W5) (FRZ-FRZ-06): the layer manifest cannot freeze while the Tier-2 bypass gate fails (matrix C2)
- PT-FC1 merged into the train -> T-AF2-MANIFESTS (final, W5) (FRZ-FRZ-08): composition findings A4-015 and A4-037 are FC-1's (matrix C2)
- T-AF2-RW-GATES merged into the train -> T-AF2-MANIFESTS (final, W5) (FRZ-PLG-08 / NC rows): NOT_COMPOSED marks are read by the RW-GATES non-composition gate

**Table 1. Registered expected-red set** (exactly the entries of `.tmp/session/arch-freeze/W0-ci-probe/expected_red_A01.json`; plan section I: it may only shrink and must equal the registered ENV/OUT set at M3). Owners are quoted from the published known-red register. No test may be merged red or marked xfail to pass a lane.

| CI job | A01 entry | Nodes | Trigger | KR row | Owner (known-red register) |
| --- | --- | ---: | --- | --- | --- |
| governance-fast | expected_red_new_vs_baseline | 15 | ACT-N11 (product path change intergrax/runtime/__init__.py) | KR-08 | IIAC rebind at the final product tree (plan R2 mitigation RB-2); not a product fix |
| diag-final-external-proof | already_red_at_baseline / diag-final-external-proof | 1 | red at baseline 7960ae7f6 | KR-06 | KR-06: UNASSIGNED |
| gate-governance-tier | already_red_at_baseline / gate-governance-tier | 7 | red at baseline 7960ae7f6 | KR-04 | KR-04: T-AF2-RW-GATES (owner not named in the plan) |
| gate-governance-wiring | already_red_at_baseline / gate-governance-wiring | 100 | red at baseline 7960ae7f6 | KR-05 | KR-05: UNASSIGNED (secret provisioning is an Operator or repo-admin matter) |
| durable-diagnostics-external-proof | already_red_at_baseline / durable-diagnostics-external-proof | 113 | red at baseline 7960ae7f6 | KR-07 | KR-07: UNASSIGNED |
| gate-tests | already_red_at_baseline / gate-tests (collection errors, 25 test modules; 40 ERROR lines incl. 3 xdist worker lines and duplicates ERROR lines) | 25 | red at baseline 7960ae7f6 | KR-01, KR-02, KR-03 | KR-01: UNASSIGNED (CI workflow sync line; not decided here); KR-02: UNASSIGNED (CI workflow sync line; not decided here); KR-03: T-AF2-RW-GATES (owner not named in the plan) |

**Table 2. Planned expected-red, to be registered** (not in `expected_red_A01.json`; a train PR may rely on an entry only after it is registered at the moment stated).

| Id | Where (job / step / node) | Owner | Registered when | Note |
| --- | --- | --- | --- | --- |
| registry-bypass step | job gate-governance-tier, step at .github/workflows/unit-tests.yml (FACT_CHECK F-54): scripts/maintenance/check_agent_registry_bypass.py (regex matches 10 agents/ files at HEAD) | PT-WA (code) + PT-G01 (gate rewrite, DP-14) | known-red register PR by the orchestrator before the train cut | not in expected_red_A01.json |
| upward-import gate node | job gate-tests (tests/unit -m 'gate and not no_ci', F-56): tests/unit/runtime/architecture/test_faudit_remediation.py::test_intergrax_no_applications_import_gate, if it carries the gate marker; unmeasured at A01 because gate-tests aborted at collection (KR-01..03) | PT-WA | known-red register PR before the train cut, after the job is re-measured once the collection errors are fixed or registered | not in expected_red_A01.json |
| dependency-ownership gate nodes | job gate-tests: tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py and tests/unit/test_dependency_governance.py (unmeasured at A01, as above) | T-AF2-DEPS-BATCH | known-red register PR before the train cut | not in expected_red_A01.json |
| KR-17 / KR-18 nodes | job gate-tests (unmeasured at A01): the KR-17 and KR-18 node ids of the known-red register | T-AF2-RW-GATES (KR-17), PT-WA (KR-18) | already rows KR-17/KR-18; the register PR adds the job/step once measured | KR rows exist; not in expected_red_A01.json |
| KR-19 nodes | job gate-tests (node ids not enumerated) | PT-FC2 | register PR after the PT-FC2 step-0 enumeration | KR row exists; not in expected_red_A01.json |
| L1 local scripts | check_intergrax_no_applications_imports.py, check_dependency_ownership.py, check_runtime_event_tenant_propagation.py: run as plan section I L1 commands only, not by unit-tests.yml (F-55) | PT-WA, T-AF2-DEPS-BATCH, PT-T01 | no CI register entry needed; the PR body of every train ticket lists their exit codes until the owner lands | local L1 reds, not CI |
| BASELINE:MANIFEST_DIGEST | ENT-AUDIT-X runs (enterprise-audit.yml jobs) after PT-WD changes the check manifest | PT-WD | train-introduced: grows the set, so it needs an explicit plan section I exception ruling (orchestrator/Operator) recorded before the PT-WD PR merges into the train; cleared by T-AF2-ENT-BASELINE-PREFREEZE (W5) | not part of the A01 set |

## (d) Negative-test table

**Red-first rule (binding for every row with basis B, S or G).** The test is written first and run once, alone, under the shared pytest lock, on a detached checkout of `a7f023615` (or the train base, whose product tree must equal it); the log goes under `.tmp/session/arch-freeze/<ticket>/red-first/` and is kept for T-AF2-RW-PLAN-AUDIT and PT-Q02. It counts only if: (1) the test is collected (no ImportError or collection error; an absent new API must make the test fail on an assertion, not at import); (2) the failing assertion is the one in the row; (3) the failing node is not a known-red row KR-01..KR-27 and not a gate that is already red at HEAD (`check_intergrax_no_applications_imports.py`, `check_agent_registry_bypass.py`, `check_dependency_ownership.py`, `check_runtime_event_tenant_propagation.py`, the KR-13/KR-14 scenario reds, the plug_03 flow); (4) the same test passes at the ticket's head in the train. Campaign probe results at B are supporting evidence only (`B -> HEAD` product drift is outside every scope), not the red-first run.

**Red-first base exception.** RF-EXC-1 (named exception, v17, RWD16-01): a ticket whose tests need code that an earlier train ticket merged runs its red-first at the train head at its own start (with those predecessors merged), not at the audit baseline; it records that SHA in its ticket checkpoint and states, with evidence, that the defect the row names is unchanged from the audit baseline; rules (1)-(4) and the log path apply unchanged. Named instances: T-IGX-051-I (T-AF2-A4007-CALLERS, NT-ROW-139: needs PT-WB's import_allowlist.py and PT-FC2's wiring_posture.py). Any other ticket needs a named entry here first.

**Re-disposition rule for basis-C rows (Q-RW-08, orchestrator ruling).** If the step-0 measurement at HEAD does not confirm a basis-C defect, the implementer records NOT_REPRODUCED with evidence (command, log under .tmp/session/arch-freeze/<ticket>/red-first/, digest) and stops work on that row. An FRZ matrix row is then re-disposed by the orchestrator in a register PR with an independent review; a DS-EXC item (an Operator-decided disposition) goes back to the Operator. The NOT_REPRODUCED record closes nothing.

Basis codes: B behavioural; S structural (AST/source over the product tree); G new gate or inventory that reports >= 1 real violation at HEAD; C conditional on a step-0 measurement at HEAD (Q-RW-08); M prevention control (ratchet, non-composition gate, inventory) whose negative test is a seeded mutation, accepted with conditions by the audit (Q-RW-07) and never counted as red-first; N no code change.

Sets: FRZ_C2 = matrix appendix C2 rows (v3); KNOWN_RED_RW = known-red FIX rows assigned in v2; PHASE3 = the 18 of WP-12 phase check 3; CODE_FIX_29 = all EQO-A CODE_FIX; FRZ_CODE_FIX_35 = matrix appendix C; SECTION_E = plan E REMEDIATION / NC items; CAMPAIGN_RW = B-1/B-2 items routed to the train.

| Row | Item | Sets | Owner | Basis | Test file | NT refs | Assertion (safe behaviour) | Red-first at HEAD before the fix |
| --- | --- | --- | --- | :---: | --- | --- | --- | --- |
| NT-ROW-001 | A4-003 | PHASE3, CODE_FIX_29 | PT-WA | S | `tests/unit/runtime/architecture/test_wa_boundary_inversion_gate.py` (NEW_PARENT_EXISTS) | NT-WA-01..04 (+NT-WA-06) | AST over the 12 seed modules and over intergrax/runtime/** (function-level, TYPE_CHECKING and importlib literals included) finds no import of intergrax.applications or applications; a fresh interpreter with a meta-path finder that raises on intergrax.applications* imports the 11 execution modules | NT-WA-01/02 fail on the 12 seed modules (closed-world RESULT 1.1). The existing red test test_intergrax_no_applications_import_gate and the red gate script do not count |
| NT-ROW-002 | A4-004 | PHASE3, CODE_FIX_29 | PT-WA | S | `tests/unit/agents/test_wa_agent_import_rule.py` (NEW_PARENT_EXISTS) | NT-WA-13, NT-WA-17 | non-test agents/** imports intergrax.integrations only through .contracts.* leaves; no integrations._shared, integrations.providers or tools.providers import | the new assertion (only .contracts.* leaves allowed) fails on 4 non-test files at HEAD: agents/local_search/steps/search_job.py, agents/tool_selection_qualifier/steps/tool_selection_job.py, agents/tool_selection_qualifier/tool_selection.py, agents/web_search_qualifier/search_provider_resolver.py (audit section 2 row 2). Context only: the old registry-bypass regex matches 10 files (B-1 R01b); that gate's red does not count |
| NT-ROW-003 | A4-005 | PHASE3, CODE_FIX_29, SECTION_E | T-AF2-DEPS-BATCH | B | `tests/unit/architecture/test_rw_deps_core_ownership_declared.py` (NEW_PARENT_EXISTS) | NT-RW-DEP-01 (NT-DEBT-02-1) | for every requirement in pyproject.toml [project].dependencies, check_project() of scripts/maintenance/check_dependency_ownership.py (imported in-process) reports no CORE_OWNERSHIP_VIOLATION; a seeded undeclared core requirement in a temporary copy of pyproject.toml is reported (sensitivity control) | the in-process check_project() at HEAD returns CORE_OWNERSHIP_VIOLATION: jsonschema and CORE_OWNERSHIP_VIOLATION: tzdata, because both are declared core requirements (pyproject.toml, FACT_CHECK F-49) but missing from CORE_ALLOWLIST (F-50a). The script's exit code does not count. If the chosen fix removes the two from core instead, the assertion becomes 'jsonschema and tzdata are not core requirements' with the same red-first run |
| NT-ROW-004 | A4-011 | PHASE3, CODE_FIX_29 | PT-G01 | G | `tests/unit/architecture/test_g01_seam_ledger_closed_world.py` (NEW_PARENT_EXISTS) | NT-G01-01 | every Tier-0/1 to Tier-3 import edge found by AST over intergrax/** (intergrax/applications excluded) is either absent or listed in governance/architecture/seam_ledger.yaml with owner and gate, and the layer-boundary owner doc status equals the ledger state | over HEAD the scan finds 27 files / 79 edges (closed-world RESULT 1.6) and no ledger exists, so the assertion fails |
| NT-ROW-005 | A4-018 | PHASE3, CODE_FIX_29, SECTION_E | T-AF2-RW-RUNTIME | S | `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py` (NEW_PARENT_EXISTS) | NT-RW-RUN-07 (NT-CTR-02-1; design of NT-WF-F01..F03) | AST over intergrax/** minus provider packages and over agents/**: no import of ollama; image captioning with an injected fake LLM adapter succeeds under a meta-path finder that raises on ollama | fails on intergrax/multimedia/images_loader.py and agents/model_routing_qualifier/qualification_types.py, which import ollama at HEAD (FACT_CHECK F-24, F-25) |
| NT-ROW-006 | A2-009 | PHASE3, CODE_FIX_29, CAMPAIGN_RW | PT-WA | S | `tests/unit/architecture/test_wa_integrations_shared_leakage.py` (NEW_PARENT_EXISTS) | NT-RW-WA-01 (NT-BND-04-1) | production modules under agents/ and applications/, minus the PT-G01 demo/proof exception list, import no intergrax.integrations._shared.* module | fails on the 8 files B-2 S01 found at B (1 agents, 7 applications; none changed B->H) |
| NT-ROW-007 | A4-006 | PHASE3, CODE_FIX_29 | PT-T01 | S | `tests/unit/scripts/test_t01_runtime_event_tenant_propagation_emitters.py` (NEW_PARENT_EXISTS) | NT-T01-A4-006 | every emitter path listed by scripts/maintenance/check_runtime_event_tenant_propagation.py exists in the tree; a seeded emitter that publishes a RuntimeEvent without tenant is reported (sensitivity control) | the list names intergrax/agents/uaep.py, which is absent at HEAD (FACT_CHECK F-26). The script's own exit 1 does not count |
| NT-ROW-008 | A6-007 | PHASE3, CODE_FIX_29, CAMPAIGN_RW | PT-WD | B | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py` (NEW_PARENT_EXISTS) | WD A01..A16 | an unmapped changed product path selects FULL (escalated_to_full); a mapped path stays narrow | A01 fails: B-1 R13 NT-08 showed floor-only selection at B; impact_map.py unchanged B->H |
| NT-ROW-009 | A3-008 | PHASE3, CODE_FIX_29 | PT-G02 | G | `tests/qualification/test_g02_frz_row_evidence_inventory.py` (NEW_PARENT_EXISTS) | NT-G02-01 | for each of the nine FRZ-BND/OWN/TYP/PLG rows and FRZ-PLG-08 the evidence inventory names >= 1 gate or test node that is collected at the SHA | no inventory exists at HEAD and the rows carry one scoped record or none (finding A3-008), so the assertion fails |
| NT-ROW-010 | A2-038 | PHASE3, CODE_FIX_29 | PT-E05 | B | `tests/unit/governance/test_e05_role_separation_identity.py` (NEW_PARENT_EXISTS) | NT-E05-01 | two IIAC manifests whose sessions differ only in label, case or whitespace, or one session id under two role labels, are rejected by verify_iiac_role_separation | at HEAD independence rests on string inequality and role labels (finding A2-038), so the manifests are accepted |
| NT-ROW-011 | A6-001 | PHASE3, CODE_FIX_29 | PT-E03 | B | `tests/qualification/ent_audit_x/acceptance/test_e03_debt_gate_content.py` (NEW_PARENT_EXISTS) | NT-E03-01 | the debt-register gate rejects a register row without owner, disposition or target | the gate is existence-only at HEAD (finding A6-001), so a content-free row passes |
| NT-ROW-012 | A6-002 | PHASE3, CODE_FIX_29 | PT-E02 | B | `tests/unit/governance/test_e02_approved_line_strict_form.py` (NEW_PARENT_EXISTS) | NT-E02-01 | verify_plan_approval rejects an Approved: line that is neither the Operator form nor a recorded delegate form | at HEAD any line matching Approved:\s*.+ is accepted (finding A6-002) |
| NT-ROW-013 | A6-006 | PHASE3, CODE_FIX_29 | PT-E03 | B | `tests/qualification/ent_audit_x/acceptance/test_e03_semantic_wrappers_fail_not_skip.py` (NEW_PARENT_EXISTS) | NT-E03-02 | in CI mode with ENT_AUDIT_X_SEMANTIC unset the semantic acceptance wrapper fails instead of skipping | at HEAD the wrappers skip silently unless the variable is set (finding A6-006) |
| NT-ROW-014 | A6-005 | PHASE3, CODE_FIX_29 | PT-WD | B | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py` (NEW_PARENT_EXISTS) | WD C01..C03 | no check of CMP, DUP, EXE, GOV, OBS, PLG, REC, REL, SEC, STA falls through to the FRZ-AUD-03 aggregator; all 85 checks are attributed explicitly | 30 checks fall through in _frz_for_check at HEAD (WD plan section 1, digests.py) |
| NT-ROW-015 | A5-001 | PHASE3, CODE_FIX_29 | PT-G01 | S | `tests/unit/architecture/test_g01_ebh3_scan_roots.py` (NEW_PARENT_EXISTS) | NT-G01-02 | the EBH-3 leakage gate scans agents/ and applications/ production modules; over HEAD it reports exactly the 8 real importers | the gate scans only intergrax/ at HEAD, so the scan-root assertion fails |
| NT-ROW-016 | A3-018 | PHASE3, CODE_FIX_29, SECTION_E | PT-G02 | S | `tests/qualification/state_x/test_g02_citation_targets_exist.py` (NEW_PARENT_EXISTS) | NT-G02-02 | every test node cited by STATE-X evidence rows (FRZ-REC-09 cites test_sxf_q21) is collected at the SHA | test_sxf_q21 is absent at HEAD (finding A3-018) |
| NT-ROW-017 | A6-003 | PHASE3, CODE_FIX_29 | PT-WD | B | `tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py` (NEW_PARENT_EXISTS) | WD B01, B02 | the evol_x test-support tree is mapped; CHANGED-SINCE computes for every one of the 12 stage cert SHAs | B-2 R25c: 9 of 12 cert SHAs exit 1 with 'unmapped evol_x paths' at B; files unchanged B->H |
| NT-ROW-018 | A6-008 | PHASE3, CODE_FIX_29 | PT-WD | B | `tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py` (NEW_PARENT_EXISTS) | WD D01..D06 | a non-certifying FAST run with only TRACKED_DEBT drift does not report PASS silently; the report discloses the promotion | runner.py promotes PARTIAL to PASS with exit 0 at HEAD (WD plan section 1) |
| NT-ROW-019 | A4-044 | CODE_FIX_29, CAMPAIGN_RW | PT-FC1 | B | `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-01..04 (03 control) in this file; NT-FC1-12 in tests/unit/architecture/test_fc1_router_and_loop_construction_gate.py | production NexusLoop + empty agent_id + non-production agent is rejected; production-eligible agent without owner/runbook rejected; production agent routes | B-1 R03 v1 CONFIRMED at router level at B; files unchanged B->H |
| NT-ROW-020 | A4-009 | CODE_FIX_29 | PT-FC1 | B | `tests/unit/runtime/nexus/test_fc1_agent_router_posture_and_fallback.py` (NEW_PARENT_EXISTS) | NT-FC1-07..11 (11 control) in this file | STRICT + no capability + no best match raises a typed error; LAB opt-in keeps fallback_used | AgentRouter falls back silently at HEAD (finding A4-009) |
| NT-ROW-021 | A4-037 | CODE_FIX_29, CAMPAIGN_RW | PT-FC1 | B | `tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-13..15, 23 in this file; NT-FC1-20..22, 26 in tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py | STRICT NexusLoop without a human decision store raises; reject and escalate verdicts are persisted durably | B-1 R03 v5 CONFIRMED at B |
| NT-ROW-022 | A4-017 | CODE_FIX_29 | PT-FC3 | B | `tests/unit/model_inference/test_fc3_vision_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-VIS1..11 | unconfigured Triton/HF raises a typed unconfigured error; zero detections return an empty list, never object.stub | the stub detection is fabricated at HEAD (finding A4-017) |
| NT-ROW-023 | A4-019 | CODE_FIX_29 | OPEN-OPERATOR | B | `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py` (NEW_PARENT_EXISTS) | contributed by PT-FC3: NT-FC3-E1..6, V1..4, N1..13 (the code fix); the closure condition (scan flags the defaults) is not delivered | unset embedding provider, vector store and notification backend raise (LAB opt-in excepted) | silent defaults to ollama, in-memory and log at HEAD (finding A4-019); the T-IGX-043-F scan is silent on the embedding default, vector-store fallback and guardrail stub, so closure stays open |
| NT-ROW-024 | A4-026 | CODE_FIX_29, CAMPAIGN_RW | PT-FC3 | B | `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-INT1..12 | unknown verification mode raises in STRICT; Slack/Teams mode requires a secret; NONE only with explicit acknowledgement | B-1 R03 v3 CONFIRMED at B |
| NT-ROW-025 | A4-002 | CODE_FIX_29 | PT-FC4 | B | `tests/unit/skills/test_skill_execution_pinning_durability_guard.py` (NEW_PARENT_EXISTS) | FC-4 NT-1..14 | in-memory KV or doc store + production_mode raises; KV without durability attestation raises; SQLite passes | is_durable=True is reported unconditionally at HEAD (finding A4-002) |
| NT-ROW-026 | A4-035 | CODE_FIX_29 | PT-FC2 | B | `tests/unit/runtime/capacity/test_fc2_capacity_fail_closed.py` (NEW_PARENT_EXISTS) | CAP-1..7 | PRODUCT + adapters enabled + Kubernetes URL unset is UNQUALIFIED and builds no adapters; in_memory never yields probe_passed | in-memory scaler substituted at HEAD (finding A4-035) |
| NT-ROW-027 | A4-036 | CODE_FIX_29 | PT-FC2 | B | `tests/unit/applications/test_fc2_queue_worker_fail_closed.py` (NEW_PARENT_EXISTS) | FC-2 queue tests | STRICT + enabled without a real broker raises; KAFKA/RABBITMQ are not rewritten to CELERY | rewrite to eager in-process Celery at HEAD (finding A4-036) |
| NT-ROW-028 | A4-030 | CODE_FIX_29 | PT-FC2 | B | `tests/unit/applications/test_fc2_memory_org_session_admission.py` (NEW_PARENT_EXISTS) | FC-2 memory tests | PRODUCT + org memory + Mongo raises; PRODUCT + no backend + persistent memory raises | in-memory org/session stores built at HEAD (finding A4-030) |
| NT-ROW-029 | A4-032 | CODE_FIX_29 | PT-FC2 | B | `tests/unit/applications/test_fc2_otlp_composition_fail_closed.py` (NEW_PARENT_EXISTS) | FC-2 OTLP/sandbox tests; NT-FC3-GR1..5 (part c) | enabled OTLP + empty endpoint raises in STRICT; STRICT + side-effect tools + no sandbox manager raises; guardrail downgrade raises unless allowed | silent degraded defaults at HEAD (finding A4-032) |
| NT-ROW-030 | FRZ-BND-04 | FRZ_CODE_FIX_35 | PT-WA | S | `tests/unit/architecture/test_wa_integrations_shared_leakage.py` (NEW_PARENT_EXISTS) | NT-BND-04-1 | as A2-009 | as A2-009 (8 files) |
| NT-ROW-031 | FRZ-BND-05 | FRZ_CODE_FIX_35 | PT-FC1 | S | `tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py` (NEW_PARENT_EXISTS) | NT-BND-05-1 (this file); related: NT-FC1-05/06 in tests/unit/runtime/nexus/test_fc1_agent_router_posture_and_fallback.py, NT-FC1-12 in tests/unit/architecture/test_fc1_router_and_loop_construction_gate.py, NT-FC1-24 in tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py; NT-BND-05-2 in tests/unit/architecture/test_rw_bnd05_provider_switch_gate.py (PT-WB) | a Tier-2 agent that constructs RuntimeContext outside the sanctioned composition root fails the construction gate; an enum/string provider switch in Tier-1/Tier-3 fails the provider-switch gate | B-1 R03 v2 CONFIRMED one STRICT-reachable relaxed build_context site at B; the switch half is C (B-2 S01 heuristic candidates, ruling DP-18) |
| NT-ROW-032 | FRZ-BND-06 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_transport_semantics.py` (NEW_PARENT_EXISTS) | NT-BND-06-1 | a hosting, queue, MCP or HTTP transport adapter that constructs or interprets a RuntimeEvent/envelope fails the transport-semantics gate | red only if the step-0 inventory at HEAD finds an adapter that owns runtime semantics; otherwise the row becomes evidence-only under PT-G02 |
| NT-ROW-033 | FRZ-OWN-02 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | G | `tests/unit/architecture/test_rw_gates_contract_census.py` (NEW_PARENT_EXISTS) | NT-OWN-02-1 | a second contract with the same field set and responsibility as a canonical one fails the census unless it has a ratification entry | the duplicate task_artifacts contracts and the three ArtifactRef types exist at HEAD without ratification (PDC-D8-R017/R018) |
| NT-ROW-034 | FRZ-OWN-03 | FRZ_CODE_FIX_35 | PT-WB | B | `tests/unit/applications/test_wb_llm_resolver_canonical.py` (NEW_PARENT_EXISTS) | NT-WB-01..07 (06, 07 controls); NT-OWN-03-1/2 | one host LLM resolver (Tier-1 canonical, Tier-3 delegates); GR-12 product-host composition passes under STRICT | two resolvers with divergent routing at HEAD (A4-016); NT-WB-01..05 are red at HEAD per the W-B plan (06, 07 are controls) |
| NT-ROW-035 | FRZ-OWN-05 | FRZ_CODE_FIX_35 | PT-WB | S | `tests/unit/architecture/test_rw_own05_duplicate_mechanisms.py` (NEW_PARENT_EXISTS) | NT-OWN-05-1 | for each of the five named responsibilities (plugin discovery, context-window tables, tool-input guard, evaluator contract, host LLM resolver) exactly one mechanism exists or the duplicate is ratified | two host LLM resolvers exist at HEAD; the other four are C until the step-0 scan |
| NT-ROW-036 | FRZ-CTR-01 | FRZ_CODE_FIX_35 | T-AF2-RW-RUNTIME | C | `tests/unit/llm_adapters/test_rw_runtime_capability_flag_parity.py` (NEW_PARENT_EXISTS) | NT-CTR-01-1 | parametrized over every shipped adapter: a declared streaming/tool-streaming capability without the method fails, and an implemented method with the flag false fails | PDC records misreporting flags (D3-R022) and Ollama without stream_with_tools (D3-R006); confirmed per adapter at step 0 |
| NT-ROW-037 | FRZ-CTR-02 | FRZ_CODE_FIX_35 | T-AF2-RW-RUNTIME | S | `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py` (NEW_PARENT_EXISTS) | NT-CTR-02-1 | as A4-018 | as A4-018 |
| NT-ROW-038 | FRZ-CTR-04 | FRZ_CODE_FIX_35 | PT-Q12 | N | `tests/qualification/pdc_x/d4/test_d4_public_api_manifest.py` (EXISTS) | NT-CTR-04-1 (replay) | no new test: the public-API manifest test is replayed at M3 | not applicable (RESOLVED_AT_BASE) |
| NT-ROW-039 | FRZ-CTR-06 | FRZ_CODE_FIX_35 | T-AF2-RW-RUNTIME | S | `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py` (NEW_PARENT_EXISTS) | NT-CTR-06-1 | as A4-018 | as A4-018 |
| NT-ROW-040 | FRZ-TYP-05 | FRZ_CODE_FIX_35 | T-AF2-RW-TYP | S | `tests/unit/architecture/test_rw_typ05_string_dispatch_reduction.py` (NEW_PARENT_EXISTS) | NT-TYP-05-2 (reduction, red-first); NT-TYP-05-1 in tests/unit/architecture/test_rw_typ_ratchet.py (ratchet, seeded new Any) | the TYP-05 detector (a module-level mapping in intergrax/contracts/** whose keys are str literals, str-constant or UPPER_CASE constant names, or tuples of those, used in the same module as a lookup NAME[...], NAME.get(...) or `in NAME` to select behaviour or an identity) finds 0 tables, apart from ratified entries in typ_ratchet_baseline.v1.json; and the ratchet fails on a seeded new Any or object annotation in a temporary copy of a real frozen contract file, against the HEAD baseline 138 Any / 357 object (Q-RW-07 conditions 1-3) | 5 tables at HEAD (work/typ05/typ05_measure.json): governed_execution_result.py _ACTION_TO_OPERATION (:57), _EFFECT_CONTRACT_OPERATION_KEY_TO_PROVIDER_OPERATION (:64), _PROVIDER_OPERATION_TO_DECISION_ACTION (:70); external_operation_redrive_trusted_effect_contracts.py _QUAL_SANCTIONED_EXECUTION_BINDING (:101), _CATALOG_TOOL_SANCTIONED_REDRIVE_BINDING (:117); above the target 0 (FACT_CHECK F-67..F-71). Meets audit condition (4) of Q-RW-07 |
| NT-ROW-041 | FRZ-PLG-01 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_plugin_contracts.py` (NEW_PARENT_EXISTS) | NT-PLG-01-1 | an evaluator registered outside the single evaluator contract fails registration | PDC-D5-R005 PARTIAL (two evaluator contracts); confirm at step 0 |
| NT-ROW-042 | FRZ-PLG-02 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_plugin_contracts.py` (NEW_PARENT_EXISTS) | NT-PLG-02-1 (new narrow test), NT-PLG-02-2 | an external chunker registered through the public plugin entry point is used without a core patch; no provider-plugin registration path exists outside LLMAdapterRegistry | the existing red plug_03 flow does NOT count; the new narrow test must be shown red at HEAD for the defect reason |
| NT-ROW-043 | FRZ-PLG-05 | FRZ_CODE_FIX_35 | PT-FC1 | B | `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-01..04; NT-PLG-05-1 in tests/unit/core/plugins/test_rw_supply_installed_file_digest.py | as A4-044; an unsigned or digest-mismatched plugin entry point is refused at admission | A4-044 half: B-1 R03 v1 CONFIRMED at B (behavioural red). Plugin half: entry-point provenance admission exists at HEAD (PluginProvenanceAdmission, F-58), so 'unsigned plugin refused' is not a red; the remaining plugin red is the installed-file digest mismatch without re-hash (EX-D6-03, NT-RW-SUP-06) and the defense-loader gap of FRZ-SEC-06 (basis C) |
| NT-ROW-044 | FRZ-PLG-06 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | G | `tests/unit/architecture/test_rw_gates_plugin_contracts.py` (NEW_PARENT_EXISTS) | NT-PLG-06-1 | every tool, skill, MCP and integration package is in the package inventory; a missing one fails | GR-11 does not cover these packages at HEAD; the new inventory gate reports them |
| NT-ROW-045 | FRZ-PLG-07 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_plugin_contracts.py` (NEW_PARENT_EXISTS) | NT-PLG-07-1 | a package that adds a capability beyond its manifest is refused | confirm at step 0 with the adversarial fixture against HEAD |
| NT-ROW-046 | FRZ-EXE-02 | FRZ_CODE_FIX_35 | PT-FC1 | B | `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-01..04; NT-EXE-02-1 = NT-WA-13 | as A4-044; a Tier-2 agent constructed outside the registry fails the registry-bypass rule | B-1 R03 v1 and R01b at B |
| NT-ROW-047 | FRZ-CFG-02 | FRZ_CODE_FIX_35 | PT-FC3 | B | `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-E*, V*, N*, VIS*, INT* | as A4-017, A4-019, A4-026 | as those findings |
| NT-ROW-048 | FRZ-CFG-06 | FRZ_CODE_FIX_35 | PT-FC3 | B | `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-E*, VIS*; NT-CFG-06-1 = NT-RW-RUN-07 | no alternate factory, fallback or direct SDK path reaches an unconfigured provider | as A4-017, A4-018, A4-019 |
| NT-ROW-049 | FRZ-CFG-08 | FRZ_CODE_FIX_35 | PT-FC3-F | G | `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py` (NEW_PARENT_EXISTS) | NT-FC3-CX1..12 | a vendor-slug default on one of the eight FC-3 paths turns the CONFIG-X gate red; hits outside the eight paths, and ambient os.environ reads outside the inventoried surfaces, are recorded (a printed list, one EDER row per real hit), not failing (Operator ruling option 3a) | over HEAD the extended scan reports 4 of the FC-3 selector defaults (notification, inbound verifier; F-A2 at B) and turns green only after PT-FC3 (hence F after C); it is silent on the embedding default, the vector-store fallback and the guardrail stub (A4-019 stays open) |
| NT-ROW-050 | FRZ-SEC-04 | FRZ_CODE_FIX_35 | PT-FC3 | B | `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-INT1..12; NT-SEC-04-1 | a STRICT or PRODUCT profile with an unset or invalid verification mode refuses to start | B-1 R03 v3 at B |
| NT-ROW-051 | FRZ-REL-07 | FRZ_CODE_FIX_35 | PT-FC2 | B | `tests/unit/runtime/capacity/test_fc2_capacity_fail_closed.py` (NEW_PARENT_EXISTS) | CAP-1..7; FC-2 queue, OTLP, sandbox tests | no silent backend substitution in STRICT/PRODUCT | as A4-032, A4-035, A4-036 |
| NT-ROW-052 | FRZ-REL-09 | FRZ_CODE_FIX_35 | PT-FC2 | B | `tests/unit/applications/test_fc2_queue_worker_fail_closed.py` (NEW_PARENT_EXISTS) | FC-2 queue tests | as A4-036 | as A4-036 |
| NT-ROW-053 | FRZ-REL-10 | FRZ_CODE_FIX_35 | PT-FC2 | B | `tests/unit/runtime/capacity/test_fc2_capacity_fail_closed.py` (NEW_PARENT_EXISTS) | CAP-1..7 | as A4-035 | as A4-035 |
| NT-ROW-054 | FRZ-PRD-03 | FRZ_CODE_FIX_35 | PT-FC2 | B | `tests/unit/applications/test_fc2_queue_worker_fail_closed.py` (NEW_PARENT_EXISTS) | FC-2 tests; NT-FC3-INT*; NT-PRD-03-1 | an unsupported production configuration in STRICT raises instead of substituting an in-memory or in-process backend | as A4-026, A4-035, A4-036 |
| NT-ROW-055 | FRZ-PRD-05 | FRZ_CODE_FIX_35 | PT-WB | S | `tests/unit/applications/test_wb_serving_bind_defaults.py` (NEW_PARENT_EXISTS) | NT-WB-14..17, 25 (firm, test_wb_serving_bind_defaults.py); NT-WB-26 is cond with Part G, in tests/unit/applications/test_wb_gca_hosted_strict_serving.py | no Tier-3 entrypoint starts serving outside the hosting lifecycle engine, except the declared set | firm part: NT-WB-14..16 red at HEAD (LKW reference production serves through uvicorn.run; the research bind default is not 127.0.0.1), B-2 S01/S03 counted 7 direct uvicorn.run files at B; NT-WB-17 is a control and NT-WB-25 a scan record. The strict production serving command (the B-2 promotion trigger) is Part G: its test NT-WB-26 is red-first only if Part G is ruled in (Q-RW-16 b) |
| NT-ROW-056 | FRZ-PRD-08 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_production_only_bypass.py` (NEW_PARENT_EXISTS) | NT-PRD-08-1, NT-PRD-08-2 | a code path active only in production mode that skips a control fails the gate; a PIS-only CREATE without provenance fails the inventory | confirm at step 0 (the matrix marks sensitivity UNVERIFIED) |
| NT-ROW-057 | FRZ-REG-09 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | S | `tests/unit/architecture/test_rw_gates_gate_completeness.py` (NEW_PARENT_EXISTS) | NT-REG-09-1, NT-REG-09-2 | the vendor-SDK banned lists contain every shipped SDK (ollama included); removing an entry fails the completeness check | the banned list lacks the ollama SDK at HEAD per WP-4 (FACT_CHECK F-27 records the grep) |
| NT-ROW-058 | FRZ-DOC-05 | FRZ_CODE_FIX_35 | PT-Q12 | N | `tests/qualification/pdc_x/d4/test_d4_public_api_manifest.py` (EXISTS) | NT-DOC-05-1 (replay) | no new test; documentation regression gates and the public-API manifest test are replayed at M3 | not applicable (RESOLVED_AT_BASE) |
| NT-ROW-059 | FRZ-DEBT-02 | FRZ_CODE_FIX_35 | T-AF2-DEPS-BATCH | B | `tests/unit/architecture/test_rw_deps_core_ownership_declared.py` (NEW_PARENT_EXISTS) | NT-DEBT-02-1 (= NT-RW-DEP-01) | for every requirement in pyproject.toml [project].dependencies, check_project() of scripts/maintenance/check_dependency_ownership.py (imported in-process) reports no CORE_OWNERSHIP_VIOLATION; a seeded undeclared core requirement in a temporary copy of pyproject.toml is reported (sensitivity control) | the in-process check_project() at HEAD returns CORE_OWNERSHIP_VIOLATION: jsonschema and CORE_OWNERSHIP_VIOLATION: tzdata, because both are declared core requirements (pyproject.toml, FACT_CHECK F-49) but missing from CORE_ALLOWLIST (F-50a). The script's exit code does not count. If the chosen fix removes the two from core instead, the assertion becomes 'jsonschema and tzdata are not core requirements' with the same red-first run |
| NT-ROW-060 | FRZ-DEBT-03 | FRZ_CODE_FIX_35 | T-AF2-DEPS-BATCH | B | `tests/unit/architecture/test_rw_deps_core_ownership_declared.py` (NEW_PARENT_EXISTS) | NT-DEBT-03-1 (= NT-RW-DEP-01) | for every requirement in pyproject.toml [project].dependencies, check_project() of scripts/maintenance/check_dependency_ownership.py (imported in-process) reports no CORE_OWNERSHIP_VIOLATION; a seeded undeclared core requirement in a temporary copy of pyproject.toml is reported (sensitivity control) | the in-process check_project() at HEAD returns CORE_OWNERSHIP_VIOLATION: jsonschema and CORE_OWNERSHIP_VIOLATION: tzdata, because both are declared core requirements (pyproject.toml, FACT_CHECK F-49) but missing from CORE_ALLOWLIST (F-50a). The script's exit code does not count. If the chosen fix removes the two from core instead, the assertion becomes 'jsonschema and tzdata are not core requirements' with the same red-first run |
| NT-ROW-061 | FRZ-DEBT-04 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_partial_inventory.py` (NEW_PARENT_EXISTS) | NT-DEBT-04-1 | every PARTIAL entry of the PDC debt register maps to exactly one owner node or a non-RW disposition, and no new PARTIAL row appears | none: the 13 rows have owners at HEAD (DS-EXC package); seeded-row test only (Q-RW-07) |
| NT-ROW-062 | FRZ-FRZ-10 | FRZ_CODE_FIX_35 | T-B7-PFP-ACTIVATE | B | `tests/unit/governance/test_pfp_post_freeze_policy_gate.py` (NEW_PARENT_EXISTS) | NT-FRZ-10-1 | after T-B7-PFP-ACTIVATE, a change to a frozen path without an exception record fails the post-freeze policy gate | behavioural: a seeded frozen-path change without an exception record is not refused at HEAD because no post-freeze policy gate exists (matrix FRZ-FRZ-10, WP-4). Bound to T-B7-PFP-ACTIVATE, a W5 node before LOCK (plan section J W5 row; FACT_CHECK S-27); T-B7-PFP-VERIFY re-checks it in W8; T-IGX-052-* letter to allocate |
| NT-ROW-063 | FRZ-TEN-01 | FRZ_CODE_FIX_35 | PT-WF | B | `tests/unit/applications/test_wf_tenant_evaluation_fail_closed.py` (NEW_PARENT_EXISTS) | NT-WF-A01..A04; NT-TEN-01-2 = NT-RW-TEN-04 | a STRICT/PRODUCT host with tenant_id None, empty or blank refuses to wire the shared evaluation registry; an observation without tenant_id is rejected | B-1 R08 g09a CONFIRMED at B |
| NT-ROW-064 | FRZ-TEN-10 | FRZ_CODE_FIX_35 | PT-WF | B | `tests/unit/applications/test_wf_serving_tenant_fail_closed.py` (NEW_PARENT_EXISTS) | NT-WF-A07..A14, A16 | a request with a missing tenant_id is rejected and never becomes 'default'; a client-asserted tenant different from the context tenant is refused | B-1 R03 v7 CONFIRMED at B; the 'default' field defaults exist at HEAD (FACT_CHECK F-22, F-23) |
| NT-ROW-065 | EX-L-09 | SECTION_E | T-AF2-RW-TENANT | B | `tests/unit/agent_distribution/test_rw_tenant_install_cas.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-05 (G-09 part: NT-WF-A01) | install_agent into an occupied slot without the expected-revision token is refused; two concurrent installs: exactly one wins | install replaces the active installation without admission or CAS at HEAD (DS-EXC row EX-L-09, G-12) |
| NT-ROW-066 | G-12 | SECTION_E | T-AF2-RW-TENANT | B | `tests/unit/agent_distribution/test_rw_tenant_install_cas.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-05 | as EX-L-09 | as EX-L-09 |
| NT-ROW-067 | EX-D4-04 | SECTION_E | T-AF2-RW-TENANT | B | `tests/unit/runtime/sandbox/test_rw_tenant_hosted_session_per_tenant.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-01 | two tenants get two backend sessions; a session is opened under the request tenant, never the fixed 'harness'/'bootstrap' identity | fixed tenant_id='harness' and task_id='bootstrap' at HEAD (FACT_CHECK F-01, F-02) |
| NT-ROW-068 | EX-D5-05 | SECTION_E | T-AF2-RW-TENANT | B | `tests/unit/applications/test_rw_tenant_online_evaluation_observation.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-04 | OnlineEvaluationObservation requires tenant_id; policy wiring of a STRICT host without tenant refuses | the base observation has no tenant_id at HEAD (FACT_CHECK F-04) |
| NT-ROW-069 | EX-D8-03 | SECTION_E | T-AF2-RW-TENANT | B | `tests/unit/tools/providers/test_rw_tenant_storage_workspace_keys.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-02 | ToolWiringContext carries the tenant; a storage.* object written for tenant A is not readable as tenant B; each raw storage.* call emits an audit event | ToolWiringContext has no tenant at HEAD (FACT_CHECK F-03) |
| NT-ROW-070 | EX-L-02 | SECTION_E | T-AF2-RW-TENANT | B | `tests/unit/tools/providers/test_rw_tenant_storage_workspace_keys.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-03 | storage and workspace object keys are bound to tenant; a key of another tenant is refused | keys are not tenant-bound at HEAD (DS-EXC EX-L-02) |
| NT-ROW-071 | EX-D8-04 | SECTION_E | T-AF2-RW-TENANT | B | `tests/unit/runtime/nexus/test_rw_tenant_artifact_read_binding.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-06 | a governed artifact read is bound to principal and run; a read by another run of the same tenant is refused | ArtifactAccessContext is tenant-wide at HEAD (FACT_CHECK F-05) |
| NT-ROW-072 | EX-D2-06 | SECTION_E | T-AF2-RW-TENANT | C | `tests/unit/applications/test_rw_tenant_strict_unscoped_memory.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-08 | a STRICT profile with an unscoped memory capability refuses composition (DS-EXC recommended option) | confirm at step 0 which memory capabilities are unscoped at HEAD |
| NT-ROW-073 | F-D2-11 | SECTION_E | T-AF2-RW-TENANT | B | `tests/unit/rag/governance/test_rw_tenant_reindex_hook_scope.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-09 | a reindex hook registered for tenant A does not run for tenant B | hooks are global and tenant-blind at HEAD (PDC debt register F-D2-11) |
| NT-ROW-074 | R-P8-01 | SECTION_E | T-AF2-RW-TENANT | B | `tests/unit/applications/test_rw_tenant_mcp_client_asserted_tenant.py` (NEW_PARENT_EXISTS) | NT-RW-TEN-07 | an MCP host mounted without harness auth refuses a client-asserted tenant (composition fails closed) | PDC_STATED fail-open identity; must be shown red at HEAD (B-1 X-A8-022 NOT_REPRODUCED concerned L2/L3 invoke security, a different property) |
| NT-ROW-075 | EX-D1-01 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-01 | an unsigned or tampered package is refused on the composed install/upgrade path; a caller-asserted trust record without signature material is refused | attestation is not verified on the composed path at HEAD (DS-EXC EX-D1-01) |
| NT-ROW-076 | EX-D1-02 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-02 | both launchers compose a revocation-state source; a revoked package is refused through each launcher | the source is optional and defaults to None at HEAD (FACT_CHECK F-06) |
| NT-ROW-077 | EX-D1-04 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-03 | a revoked revision without a trusted rollback target stops serving | behaviour unchanged at HEAD (DS-EXC EX-D1-04) |
| NT-ROW-078 | PDC-D1-R017 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-04 | the lifecycle evidence sink has a production consumer; an install event reaches the evidence plane | the sink is optional and inert at HEAD (DS-EXC Table B) |
| NT-ROW-079 | A-01 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/applications/test_rw_supply_trust_hook_before_persist.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-05 | a revision refused by the trust hook is not persisted, validated, registered or prepared | the hook runs after persist_candidate_revision at HEAD (FACT_CHECK F-07) |
| NT-ROW-080 | EX-D6-03 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/core/plugins/test_rw_supply_installed_file_digest.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-06 (= NT-PLG-05-1) | an installed plugin file whose digest differs from the recorded one is refused | no installed-file digest source exists at HEAD (DS-EXC EX-D6-03) |
| NT-ROW-081 | EX-D6-01 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/runtime/security/test_rw_supply_audit_trail_tamper_evidence.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-07 | a modified audit-trail entry is detected by verification; under the NOT_CLAIMED option the claim-text check fails on any tamper-evidence claim | entries carry no hash chain or signature at HEAD (DS-EXC EX-D6-01) |
| NT-ROW-082 | F-D6-02 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/runtime/security/test_rw_supply_audit_trail_tamper_evidence.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-07 | as EX-D6-01 | as EX-D6-01 |
| NT-ROW-083 | EX-D6-02 | SECTION_E | T-AF2-RW-SUPPLY | C | `tests/unit/applications/test_rw_supply_strict_validator_refuses_weak.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-08 | the production host factory composes the D6 controls | requalify first (PT-Q12); red only if no control is composed at HEAD |
| NT-ROW-084 | F-D6-04 | SECTION_E | T-AF2-RW-SUPPLY | C | `tests/unit/applications/test_rw_supply_strict_validator_refuses_weak.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-08 | as EX-D6-02 | as EX-D6-02 |
| NT-ROW-085 | F-D6-10 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/applications/test_rw_supply_strict_validator_refuses_weak.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-09 | the strict-config validator refuses weak or disabled security settings | it accepts them at HEAD (PDC-D6-R024) |
| NT-ROW-086 | F-D3-12 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/runtime/security/test_rw_supply_fail_open_set.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-10 | a malformed quota value denies the call instead of disabling the quota | the quota disables itself on a malformed value at HEAD (FACT_CHECK F-08) |
| NT-ROW-087 | PDC-D3-R018 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/runtime/security/test_rw_supply_fail_open_set.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-10 + TENANT-X carrier count | as F-D3-12; the TENANT-X carrier count of T-IGX-041-I is recorded | as F-D3-12 |
| NT-ROW-088 | F-D6-09 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/runtime/security/test_rw_supply_fail_open_set.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-11 | the poisoning default is the same (most restrictive) on the RAG service path and the plan-context path | the default differs by path at HEAD (PDC F-D6-09) |
| NT-ROW-089 | F-D6-11 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/runtime/security/test_rw_supply_fail_open_set.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-12 | a tenant-less resource is not readable by an arbitrary scoped tenant; the identity is normalised | open to any scoped tenant at HEAD (PDC F-D6-11) |
| NT-ROW-090 | F-D5-12 | SECTION_E | T-AF2-RW-SUPPLY | C | `tests/unit/runtime/security/test_rw_supply_fail_open_set.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-13 | a writer without the lease owner token cannot release or overwrite the registry lock; the id-existence oracle is closed | not reproduced across processes per the PDC register; confirm at step 0 |
| NT-ROW-091 | F-D8-14 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/runtime/nexus/test_rw_supply_retention_and_legal_hold.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-14 | a legal hold cannot be overwritten or lost | holds are overwritable and lossy at HEAD (PDC F-D8-14) |
| NT-ROW-092 | EX-D8-01 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/runtime/nexus/test_rw_supply_retention_and_legal_hold.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-15 | a shipped composition enforces retention/TTL and an AUDIT_EVIDENCE producer exists, or the explicit responsibility statement with a fail-closed gate is in place | no shipped composition enforces retention at HEAD (DS-EXC EX-D8-01) |
| NT-ROW-093 | F-D6-14 | SECTION_E | T-AF2-RW-SUPPLY | G | `tests/unit/scripts/test_rw_supply_static_secret_gate.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-16 | the static secret gate flags the known raw_secret_gate evasions (seeded) and every secret-shaped literal in product code | no static gate exists at HEAD (PDC-D6-R007); the new gate must report >= 1 real finding or the row is re-dispositioned |
| NT-ROW-094 | PDC-D6-R007 | SECTION_E | T-AF2-RW-SUPPLY | G | `tests/unit/scripts/test_rw_supply_static_secret_gate.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-16 | as F-D6-14 | as F-D6-14 |
| NT-ROW-095 | F-D6-03 | SECTION_E | T-AF2-RW-SUPPLY | B | `tests/unit/runtime/security/test_rw_supply_audit_trail_tamper_evidence.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-17 | the audit-trail listing does not silently drop entries beyond 1000 and secret-shaped values are redacted | oldest 1000 only and secrets stored at HEAD (PDC F-D6-03) |
| NT-ROW-096 | PDC-D6-R010 | SECTION_E | T-AF2-RW-SUPPLY | G | `tests/unit/scripts/test_rw_supply_static_secret_gate.py` (NEW_PARENT_EXISTS) | NT-RW-SUP-18 (check_sandbox_isolation_strength.py) | the sandbox isolation gate refuses a configuration that is present but weak | the gate checks presence, not strength, at HEAD (DS-EXC Table B) |
| NT-ROW-097 | EX-D2-01 | SECTION_E | T-AF2-RW-TRUST | B | `tests/unit/context/test_rw_trust_untrusted_never_system_role.py` (NEW_PARENT_EXISTS) | NT-RW-TRU-01 | an injection fixture in a RAG, web or memory fragment never reaches the model as role=system; ContextFragmentSource has an untrusted class | context blocks are assembled as role=system at HEAD (FACT_CHECK F-09, F-10, F-11) |
| NT-ROW-098 | EX-D2-02 | SECTION_E | T-AF2-RW-TRUST | B | `tests/unit/context/test_rw_trust_egress_default_most_restrictive.py` (NEW_PARENT_EXISTS) | NT-RW-TRU-02 | a fragment without sensitivity gets the most restrictive class; the egress default refuses it | the egress ceiling defaults to INTERNAL at HEAD (DS-EXC EX-D2-02) |
| NT-ROW-099 | EX-D2-05 | SECTION_E | T-AF2-RW-TRUST | B | `tests/unit/context/test_rw_trust_planner_ledger_full_message.py` (NEW_PARENT_EXISTS) | NT-RW-TRU-03 | the planning ledger sizes the full message and enforces the window | content-only sizing at HEAD (DS-EXC EX-D2-05) |
| NT-ROW-100 | EX-D2-07 | SECTION_E | T-AF2-RW-TRUST | B | `tests/unit/llm/test_rw_trust_orphan_tool_call_id_strict.py` (NEW_PARENT_EXISTS) | NT-RW-TRU-04 | in STRICT an orphan tool_call_id fails closed | an orphan id is accepted at HEAD (DS-EXC EX-D2-07) |
| NT-ROW-101 | EX-L-05 | SECTION_E | T-AF2-RW-TRUST | C | `tests/unit/rag/test_rw_trust_vectorstore_double_conformance.py` (NEW_PARENT_EXISTS) | NT-RW-TRU-05 | every FakeVectorstore double conforms to the vectorstore contract; the three reingest/takeover reds have a root cause and pass | 3 of 8 classified RAG reds have an undetermined cause; red-first per root cause at step 0 (existing reds do not count) |
| NT-ROW-102 | PDC-D2-R004 | SECTION_E | T-AF2-RW-TRUST | C | `tests/unit/rag/test_rw_trust_vectorstore_double_conformance.py` (NEW_PARENT_EXISTS) | NT-RW-TRU-05 | as EX-L-05 | as EX-L-05 |
| NT-ROW-103 | EX-L-08 | SECTION_E | T-AF2-RW-RUNTIME | B | `tests/unit/llm_adapters/test_rw_runtime_stream_open_breaker.py` (NEW_PARENT_EXISTS) | NT-RW-RUN-01 | a stream-open failure raised at first iteration is recorded by the breaker as a failure and retried within the bound | the resilience wrapper receives a lazy generator and records success before any SDK call at HEAD (FACT_CHECK F-12; B-2 R24 note 5) |
| NT-ROW-104 | EX-D3-01 | SECTION_E | T-AF2-RW-RUNTIME | B | `tests/unit/llm_adapters/test_rw_runtime_timeout_enforced.py` (NEW_PARENT_EXISTS) | NT-RW-RUN-02 | LLMCallConfig.timeout_sec reaches the SDK call; a hung fake SDK call is cancelled at the timeout | timeout_sec is computed and dropped at HEAD (DS-EXC EX-D3-01) |
| NT-ROW-105 | EX-D3-03 | SECTION_E | T-AF2-RW-RUNTIME | B | `tests/unit/llm_adapters/test_rw_runtime_typed_error_contract_v1.py` (NEW_PARENT_EXISTS) | NT-RW-RUN-03 | every provider error maps to a typed error class of contract v1 | no typed provider-error contract at HEAD (DS-EXC EX-D3-03) |
| NT-ROW-106 | EX-L-07 | SECTION_E | T-AF2-RW-RUNTIME | B | `tests/unit/llm_adapters/test_rw_runtime_refusal_normalization_matrix.py` (NEW_PARENT_EXISTS) | NT-RW-RUN-04 | refusal and finish normalization is pinned for generate, stream, generate_with_tools and stream_with_tools of every provider and the Bedrock invoke paths | only two methods are pinned at HEAD (DS-EXC EX-L-07) |
| NT-ROW-107 | PDC-D3-R005 | SECTION_E | T-AF2-RW-RUNTIME | B | `tests/unit/llm_adapters/test_rw_runtime_served_model_attribution.py` (NEW_PARENT_EXISTS) | NT-RW-RUN-05 | attribution records the served model, not the requested one | the requested model is recorded at HEAD (PDC F-D3-10) |
| NT-ROW-108 | PDC-D3-R019 | SECTION_E | T-AF2-RW-RUNTIME | S | `tests/unit/llm_adapters/test_rw_runtime_capability_flag_parity.py` (NEW_PARENT_EXISTS) | NT-RW-RUN-06 | every provider default has a VBN classification in tests/qualification/ebh_6/_findings_register.py | the classification is missing at HEAD (PDC F-D3-09) |
| NT-ROW-109 | EX-L-06 | SECTION_E | T-AF2-RW-RECOVERY | B | `tests/unit/queueing/test_rw_recovery_no_double_execution.py` (NEW_PARENT_EXISTS) | NT-RW-REC-01 | stop() then start() never runs two consumers; stop drains or joins the running thread | stop abandons a running thread at HEAD (DS-EXC EX-L-06) |
| NT-ROW-110 | EX-L-03 | SECTION_E | T-AF2-RW-RECOVERY | B | `tests/unit/queueing/test_rw_recovery_no_double_execution.py` (NEW_PARENT_EXISTS) | NT-RW-REC-02 | recover_interrupted_running is compare-and-set; a write in flight is not reset | not CAS at HEAD (FACT_CHECK F-13) |
| NT-ROW-111 | EX-D7-01 | SECTION_E | T-AF2-RW-RECOVERY | B | `tests/unit/hosting/test_rw_recovery_drain_wait_for_idle.py` (NEW_PARENT_EXISTS) | NT-RW-REC-03 | a product class implements the active-work controller; WorkerRuntime drains on stop; no double execution across restart or stop | no implementing class and no stop/drain API at HEAD (FACT_CHECK F-14) |
| NT-ROW-112 | EX-D4-01 | SECTION_E | T-AF2-RW-RECOVERY | B | `tests/unit/hosting/test_rw_recovery_drain_wait_for_idle.py` (NEW_PARENT_EXISTS) | NT-RW-REC-04 | the hosting engine waits for idle through the product adapter before shutdown | no product implementation of wait_for_idle at HEAD (DS-EXC EX-D4-01) |
| NT-ROW-113 | EX-D1-07 | SECTION_E | T-AF2-RW-RECOVERY | B | `tests/unit/applications/test_rw_recovery_durable_cas_projection.py` (NEW_PARENT_EXISTS) | NT-RW-REC-05 | the durable CAS compares revision_state, activated_at and the full bindings; a concurrent change is rejected | the comparison neutralizes those fields at HEAD (DS-EXC EX-D1-07) |
| NT-ROW-114 | PDC-D7-R015 | SECTION_E | T-AF2-RW-RECOVERY | S | `tests/unit/runtime/long_running/test_rw_recovery_uncertain_reconciler.py` (NEW_PARENT_EXISTS) | NT-RW-REC-06 | an UNCERTAIN run is reconciled exactly once; the double-execution check records a result | no reconciler exists at HEAD (DS-EXC Table B) |
| NT-ROW-115 | G-20 | SECTION_E | T-AF2-RW-RECOVERY | B | `tests/unit/runtime/long_running/test_rw_recovery_uncertain_reconciler.py` (NEW_PARENT_EXISTS) | NT-RW-REC-07 | a permanent materialization failure is retried a capped number of times; OverflowError is mapped | the claim is released every tick with no cap at HEAD (PDC A-08/G-20) |
| NT-ROW-116 | EX-D7-03 | SECTION_E | T-AF2-RW-RECOVERY | S | `tests/unit/queueing/test_rw_recovery_producer_identity.py` (NEW_PARENT_EXISTS) | NT-RW-REC-08 | every durable enqueue site passes TaskRequest.identity | 1 of 6 constructor sites passes identity at HEAD (closed-world RESULT 2.3) |
| NT-ROW-117 | EX-D5-03 | SECTION_E | T-AF2-RW-GATES | B | `tests/unit/scripts/test_rw_gates_promotion_gate_binding.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-01 | the promotion gate DENIES an authorization bound to an unrelated ticket or stage and checks the plan digest or delegation record | the binding is not checked at HEAD (DS-EXC EX-D5-03) |
| NT-ROW-118 | EX-L-01 | SECTION_E | T-AF2-RW-GATES | S | `tests/unit/architecture/test_rw_gates_scenario_get_steps_contract.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-02 | every scenario agent and the scenario runtime baseline use the single canonical get_steps(context) signature | the signature drifts at HEAD (DS-EXC EX-L-01). The 77 known reds (KR-13) do not count; this new structural test does |
| NT-ROW-119 | EX-L-11 | SECTION_E | T-AF2-RW-GATES | S | `tests/unit/architecture/test_rw_gates_scenario_get_steps_contract.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-02 | as EX-L-01 | as EX-L-01 |
| NT-ROW-120 | EX-D3-04 | SECTION_E | T-AF2-RW-GATES | S | `tests/unit/architecture/test_rw_gates_scenario_get_steps_contract.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-02 + the four fixture_strict_adapter_parity tests | the scenario manifest passes contract authority and the four parity tests pass | as EX-L-01; the four existing red tests alone do not count |
| NT-ROW-121 | EX-L-13 | SECTION_E | T-AF2-RW-GATES | B | `tests/unit/architecture/test_rw_gates_event_schema_reconciled.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-03 | a step_started payload on the finalized-trace path passes the schema guard (trace_bridge.v1 and graph_node.v1 reconciled) | the guard rejects it at HEAD (DS-EXC EX-L-13); the known red KR-14 does not count |
| NT-ROW-122 | PDC-D5-R002 | SECTION_E | T-AF2-RW-GATES | B | `tests/unit/architecture/test_rw_gates_eval_digest_enforced.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-04 | a dataset asset without a content digest, or with a mismatching one, is refused | the digest is optional and unenforced at HEAD (PDC F-D5-09) |
| NT-ROW-123 | PDC-D1-R015 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/agent_distribution/test_rw_gates_registry_reuse.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-05 | an installed agent reuses its registry entry; no duplicate entry is created | none: the row is PARTIAL for a missing reuse test, not a known defect (Q-RW-07) |
| NT-ROW-124 | PDC-D6-R015 | SECTION_E | T-AF2-RW-GATES | C | `tests/qualification/pdc_x/xd/test_rw_gates_model_provider_tenant_data.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-06 | model-provider tenant data of tenant A is not visible to tenant B (an XD pair) | unproven at HEAD (PDC F-D6-20); red only if the step-0 probe reproduces a leak |
| NT-ROW-125 | EX-D4-05 | SECTION_E | T-AF2-RW-GATES | S | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-07 | business_agents_deploy_enabled is read from the application profile only; the generic host profile no longer declares it | declared in the generic profile at HEAD (closed-world RESULT 2.3) |
| NT-ROW-126 | EX-L-04 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-L-04 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-127 | EX-D1-03 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-D1-03 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-128 | EX-D1-05 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-D1-05 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-129 | EX-D1-06 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-D1-06 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-130 | EX-D2-03 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-D2-03 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-131 | EX-D3-02 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-D3-02 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-132 | EX-D5-02 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-D5-02 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-133 | EX-D7-02 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-D7-02 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-134 | EX-D8-02 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-D8-02 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-135 | EX-L-10 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-L-10 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-136 | EX-L-14 | SECTION_E | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-NC-EX-L-14 | composing the NOT_COMPOSED capability from a STRICT/PRODUCT root raises; a new production import edge to it fails the non-composition gate (seeded) | none: dormant by measurement (closed-world RESULT 2.3); this is a non-composition gate, not a red-first test of a fix |
| NT-ROW-137 | F-D3-16 | SECTION_E | PT-D07 | N | `-` (n/a) | - | docs only | not applicable |
| NT-ROW-138 | X-A4-004 | CAMPAIGN_RW | PT-WA | S | `tests/unit/agents/test_wa_agent_import_rule.py` (NEW_PARENT_EXISTS) | NT-WA-13, NT-WA-17 | as A4-004 | the new assertion (only .contracts.* leaves allowed) fails on 4 non-test files at HEAD: agents/local_search/steps/search_job.py, agents/tool_selection_qualifier/steps/tool_selection_job.py, agents/tool_selection_qualifier/tool_selection.py, agents/web_search_qualifier/search_provider_resolver.py (audit section 2 row 2). Context only: the old registry-bypass regex matches 10 files (B-1 R01b); that gate's red does not count |
| NT-ROW-139 | A4-007 | CAMPAIGN_RW | T-AF2-A4007-CALLERS | B | `tests/unit/applications/test_a4007_caller_import_refusal.py` (NEW_PARENT_EXISTS) | NT-A7C-01..08 (08 cond) in this file; NT-A7C-09, 10 in tests/unit/scripts/test_a4007_migration_gate_allowlist.py; controls CTL-A7C-01..04; UT-A7C-01..05 in tests/unit/applications/test_a4007_import_policy_derivation.py; contributed (PT-WB, merged): NT-WB-18..23, 30 (control 24) in tests/unit/applications/test_wb_import_allowlist.py | b1: in a PRODUCT or STRICT context every product caller of the agent loaders (build_agent_from_binding with serialized import_path, factory_path, empty builders, typed factory with a serialized path; EnvironmentSkillToolConsistencyCheck.validate_binding; the roster contract-authority lab compat) refuses a module outside the registered set and never imports it (recording wrapper around importlib.import_module); b2: both migration-gate entry points refuse an unregistered application package with migrations and never import its manifest | red-first base under RF-EXC-1: the train head at the start of T-IGX-051-I, with PT-WB and PT-FC2 merged (SHA recorded in the ticket checkpoint); not a7f023615, because the tests need PT-WB code (ImportPathNotAllowedError and the allowlist parameters exist only in import_allowlist.py on the train). The b1/b2 defect is unchanged from a7f023615: no product caller passes an allowlist or a fail-closed posture at either SHA (B-1 R02 b1, b2 CONFIRMED at B; delta audit AUDIT_DELTA_V16 section 3). NT-A7C-01..07, 09, 10 fail on the recorded import at that base (derived from the call chains, not run). Log: .tmp/session/arch-freeze/T-IGX-051-I/red-first/. The loader-level NT-WB tests of PT-WB do not count for b1/b2 |
| NT-ROW-140 | A4-015 | CAMPAIGN_RW | PT-FC1 | B | `tests/unit/runtime/nexus/test_fc1_agent_router_posture_and_fallback.py` (NEW_PARENT_EXISTS) | NT-FC1-05, 06 in this file; NT-FC1-12 in tests/unit/architecture/test_fc1_router_and_loop_construction_gate.py; NT-FC1-24 (Part C1) in tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py | STRICT host: Tier-2 build_context yields production_mode True; NexusLoop without explicit posture raises | B-1 R03 v2 at B |
| NT-ROW-141 | A4-027 | CAMPAIGN_RW | PT-WF | B | `tests/unit/applications/test_wf_serving_tenant_fail_closed.py` (NEW_PARENT_EXISTS) | NT-WF-A07..A14, A16 | as FRZ-TEN-10 | B-1 R03 v7 at B |
| NT-ROW-142 | A4-031 | CAMPAIGN_RW | PT-FC1 | B | `tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-25 (part C2) in this file | a STRICT host composition hands UAEP both a context engine and an LLM adapter, or refuses | B-1 R03 v6 CONFIRMED at B (composition level) |
| NT-ROW-143 | X-A4-024 | CAMPAIGN_RW | PT-FC1 | B | `tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py` (NEW_PARENT_EXISTS) | NT-FC1-19 in this file; NT-FC1-16, 17 (control 18) in tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py | a STRICT profile with all memory kinds off does not open the task-memory store when INTERGRAX_TASK_MEMORY_DB is set | B-1 R03 v4 at B |
| NT-ROW-144 | X-ESC-E1 | CAMPAIGN_RW | PT-WF | B | `tests/unit/runtime/execution/test_wf_child_admission_narrowing.py` (NEW_PARENT_EXISTS) | NT-WF-G1-01..03 | child admission refuses scopes outside the parent even with an injected wide policy | B-1 R03 v8 at B |
| NT-ROW-145 | A8-020 | CAMPAIGN_RW | PT-WF | B | `tests/unit/compat/langchain/test_wf_trusted_scope_conversion.py` (NEW_PARENT_EXISTS) | NT-WF-G3-01..04 | compat conversion takes tenant from the trusted scope, never from foreign document metadata | B-1 R04 nt11a/nt11b at B |
| NT-ROW-146 | X-A4-016 | CAMPAIGN_RW | PT-WB | B | `tests/unit/applications/test_wb_llm_resolver_canonical.py` (NEW_PARENT_EXISTS) | NT-WB-02, 03, 27, 28 | a routing record is consumable only by the adapter and tenant that produced it | B-1 R04 nt12a/nt12b at B |
| NT-ROW-147 | X-A8-003 | CAMPAIGN_RW | PT-WB | B | `tests/unit/contracts/test_wb_profile_projection.py` (NEW_PARENT_EXISTS) | NT-WB-08..13, 29 | each configured Tier-3-only value is carried by the projection or the projection refuses | B-1 R04 nt13a at B (criterion open: Q-WB-03); NT-WB-08 (unmapped field) and NT-WB-09 (extra='ignore' target) are refusals red at HEAD |
| NT-ROW-148 | X-ESC-E4 | CAMPAIGN_RW | PT-WF | B | `tests/unit/debug/test_wf_experiment_tenant_scope.py` (NEW_PARENT_EXISTS) | NT-WF-G4-01..04 | tenant B cannot list or show an experiment registered by tenant A | B-1 R04 nt15b at B |
| NT-ROW-149 | X-ESC-E5 | CAMPAIGN_RW | PT-WF | B | `tests/unit/applications/test_wf_debug_router_guard_idp.py` (NEW_PARENT_EXISTS) | NT-WF-G5-01..02 (5b held: G5-03..06) | the harness guard denies an anonymous caller when an identity provider is configured | B-1 R04 nt15d at B |
| NT-ROW-150 | X-A5-011 | CAMPAIGN_RW | PT-WF | B | `tests/unit/applications/test_wf_tenant_evaluation_fail_closed.py` (NEW_PARENT_EXISTS) | NT-WF-A01..A04 | as FRZ-TEN-01 | B-1 R08 g09a at B |
| NT-ROW-151 | X-ESC-E2 | CAMPAIGN_RW | PT-WF | B | `tests/unit/runtime/execution/test_wf_decision_finalization_strict.py` (NEW_PARENT_EXISTS) | NT-WF-G2-01..03 | a shipped STRICT host binds a durable decision finalization store or refuses | B-1 R12 nt09b at B |
| NT-ROW-152 | X-A6-003 | CAMPAIGN_RW | PT-WD | B | `tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py` (NEW_PARENT_EXISTS) | WD B01, B02 | as A6-003 | B-2 R25c at B |
| NT-ROW-153 | A4-040 | CAMPAIGN_RW | PT-WB | S | `tests/unit/applications/test_wb_serving_bind_defaults.py` (NEW_PARENT_EXISTS) | NT-WB-14..17, 25 (firm, test_wb_serving_bind_defaults.py); NT-WB-26 is cond with Part G, in tests/unit/applications/test_wb_gca_hosted_strict_serving.py | as FRZ-PRD-05 | firm part: NT-WB-14..16 red at HEAD (LKW reference production serves through uvicorn.run; the research bind default is not 127.0.0.1), B-2 S01/S03 counted 7 direct uvicorn.run files at B; NT-WB-17 is a control and NT-WB-25 a scan record. The strict production serving command (the B-2 promotion trigger) is Part G: its test NT-WB-26 is red-first only if Part G is ruled in (Q-RW-16 b) |
| NT-ROW-154 | A4-020 | CAMPAIGN_RW | PT-WB | C | `tests/unit/architecture/test_rw_bnd05_provider_switch_gate.py` (NEW_PARENT_EXISTS) | NT-BND-05-2 | a provider-selection switch outside IntegrationProfile in Tier-1/Tier-3 fails, except files on the DP-18 named list | red only for a file ruled a bypass (B-2: heuristic candidates) |
| NT-ROW-155 | A4-025 | CAMPAIGN_RW | OPEN-OPERATOR | N | `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py` (NEW_PARENT_EXISTS) | none (NT-CFG-08-1 is withdrawn: no plan defines it) | no failing test: by the Operator ruling (option 3a) the T-IGX-043-F scan records ambient environment reads outside the inventoried CONFIG-X surfaces instead of failing on them; A4-025 is an open Operator item (no owner yet) | not applicable (no failing assertion is delivered); B-2 S01 counted 22 runtime files / 27 sites, 9 of 14 named by no CONFIG-X gate |
| NT-ROW-156 | KR-18 | KNOWN_RED_RW | PT-WA | S | `tests/unit/architecture/test_wa_kr18_contract_root_no_shared_import.py` (NEW_PARENT_EXISTS) | NT-WA-KR18-1, NT-WA-KR18-3 (S); NT-WA-KR18-2, NT-WA-KR18-4 (C) | KR18-1: AST over intergrax/applications/contracts/** (function-level imports included) obeys the full rule of the register node: no import with prefix intergrax.runtime. or intergrax.applications._shared., no import containing .registry. outside the four intergrax.{integrations,llm_adapters,tools,skills}.contracts. prefixes, none of the three exact forbidden modules; KR18-3 (variant P2): intergrax/tools/contracts/tool_availability.py has no module-level .registry. import and imports the catalog and is_tool_enabled inside available_tool_ids_for_profile (pins the laziness the EBH-2A extraction cannot see); KR18-2/4 controls: the inline admission equals the _shared helper, and the skills.registry re-export is the same object with an unchanged result | KR18-1: root.py imports strict_host_dependency_concurrency_admission from applications._shared at HEAD (FACT_CHECK F-44); KR18-3: tool_availability.py is absent at HEAD (FACT_CHECK F-73). The known-red node of KR-18 does not count; under P2 it is expected green by its scope boundary (plan 7b, read, not run; S1 runs it at the wave head) |
| NT-ROW-157 | KR-19 | KNOWN_RED_RW | PT-FC2 | C | `tests/unit/runtime/observability/test_fc2_kr19_import_order_independent.py` (NEW_PARENT_EXISTS) | NT-FC2-KR19 | in fresh interpreters, importing observability_otlp then export_boundary and the reverse order both succeed, and the provider is imported only on selection | order-dependent cycle (KR-19 evidence); red only if step 0 reproduces it in a fresh interpreter at HEAD; the existing strict-xfail cases do not count |
| NT-ROW-158 | KR-17 | KNOWN_RED_RW | T-AF2-RW-GATES | S | `tests/unit/architecture/test_rw_gates_kr17_vendor_sdk_inside_opens.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-KR17 | inside intergrax/integrations/providers/document_store/mongodb/, pymongo is imported only by the declared opens module | client.py imports pymongo.errors at HEAD (FACT_CHECK F-45). The known-red node of KR-17 does not count |
| NT-ROW-159 | FRZ-SEC-06 | FRZ_C2 | T-AF2-RW-SUPPLY | C | `tests/unit/runtime/security/test_rw_supply_defense_plugin_provenance_required.py` (NEW_PARENT_EXISTS) | NT-SEC-06-1 | a STRICT or PRODUCT root that loads security-defense entry points refuses a plugin without provenance: load_security_defense_plugin_report is never reached with SecurityDefenseAdmissionPolicy.provenance=None on such a root | entry-point admission already verifies provenance at HEAD (PluginProvenanceAdmission, intergrax/core/plugins/admission.py, FACT_CHECK F-58), so that half is not a red. The remaining gap: defense_plugin_loader.py defaults provenance to None (F-59) and skips the check when it is None (F-60). Red only if step 0 shows a strict/product root reaching load_security_defense_plugin_report without provenance (caller: intergrax/core/security_bootstrap.py); otherwise the re-disposition rule applies. The no-re-hash limit belongs to EX-D6-03 |
| NT-ROW-160 | FRZ-PLG-08 | FRZ_C2 | T-AF2-RW-GATES | M | `tests/unit/architecture/test_rw_gates_not_composed.py` (NEW_PARENT_EXISTS) | NT-PLG-08-1 | RETIRE and skill-lifecycle capabilities that are not composed carry a NOT_COMPOSED mark, and composing them from a STRICT/PRODUCT root fails the non-composition gate (seeded); the UPGRADE part is out of this row (Q-RW-19 / DP-30) | none: the DS-EXC dispositions keep the composition out of the train (EX-D1-05/EX-D1-06 ACCEPTABLE_SCOPED, W-F JD held); the 'N/A-with-evidence (Path B)' label stays invalid; prevention control under the Q-RW-07 ruling |
| NT-ROW-161 | FRZ-TYP-01 | FRZ_C2 | PT-WF | S | `tests/unit/runtime/nexus/test_wf_exposure_payload_typed_seam.py` (NEW_PARENT_EXISTS) | NT-WF-B01..B06 | the Nexus exposure path carries a bounded typed payload: no AuthoritativeDecisionExposure[object], ExposureAccepted[object] or DecisionFlowResult[object] annotation remains | AuthoritativeDecisionExposure[object] occurs at HEAD (FACT_CHECK F-52); the EBH-6 Q2-D2 gate files are inverted only by the Operator (Q-WF-04) |
| NT-ROW-162 | FRZ-TYP-03 | FRZ_C2 | PT-WF | S | `tests/unit/runtime/nexus/test_wf_exposure_payload_typed_seam.py` (NEW_PARENT_EXISTS) | NT-WF-B01..B06 | as FRZ-TYP-01 (same Q2-D2 seam) | as FRZ-TYP-01 |
| NT-ROW-163 | FRZ-RPL-04 | FRZ_C2 | PT-WF | S | `tests/unit/applications/test_wf_colocated_drain_protocol.py` (NEW_PARENT_EXISTS) | NT-WF-C01..C05 | no production module checks isinstance against VenvBundleColocatedRuntimeDeploymentAdapter: AST over intergrax/** and applications/** (tests excluded) finds 0 such checks; the four sites at HEAD move to a typed protocol method | 4 sites at HEAD (FACT_CHECK F-53, F-61, F-62): colocated_serving_runtime_composition.py (2), production_lifecycle.py (1), strict_production_serving_launcher.py (1); all four files are in the PT-WF scope |
| NT-ROW-164 | KR-03 | KNOWN_RED_RW | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_kr03_kr04_kr07_cause.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-KR-03 | test_reference_harness_strict.py is collected: the symbol it imports is exported where it imports it from, or the test imports it from its defining module | collection error at HEAD: lab_harness_context_from_modality_tooling is defined in intergrax/runtime/execution/lab_reference_agent_runtime.py (F-63) and absent from reference_harness_runtime.py (F-64). The known-red nodes themselves do not count. Step 0 measures the cause: product side -> fix in RW-GATES (cond scope) with this test red-first; test side -> T-AF2-RED-CLASSES ruling, fix outside the train, row re-disposed under the re-disposition rule |
| NT-ROW-165 | KR-04 | KNOWN_RED_RW | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_kr03_kr04_kr07_cause.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-KR-04 | the 7 gate-governance-tier tests run with an active governance identity, or the pre-model principal check is changed by a ruled contract change | the 7 nodes fail at pre_model_principal.py on a missing governance identity (KR-04, A01 table 1). The known-red nodes themselves do not count. Step 0 measures the cause: product side -> fix in RW-GATES (cond scope) with this test red-first; test side -> T-AF2-RED-CLASSES ruling, fix outside the train, row re-disposed under the re-disposition rule |
| NT-ROW-166 | KR-07 | KNOWN_RED_RW | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_kr03_kr04_kr07_cause.py` (NEW_PARENT_EXISTS) | NT-RW-GAT-KR-07 | MongoDBDocumentStoreIntegration satisfies the ConditionalDocumentStore protocol checked by the conformance helper | isinstance against the protocol fails for MongoDBDocumentStoreIntegration (KR-07; integration.py F-65). The known-red nodes themselves do not count. Step 0 measures the cause: product side -> fix in RW-GATES (cond scope) with this test red-first; test side -> T-AF2-RED-CLASSES ruling, fix outside the train, row re-disposed under the re-disposition rule |
| NT-ROW-167 | FRZ-TYP-02 | FRZ_CODE_FIX_35 | T-AF2-RW-TYP | M | `tests/unit/architecture/test_rw_typ_ratchet.py` (NEW_PARENT_EXISTS) | NT-TYP-02-1 | a seeded new Any or object annotation in a temporary copy of a real frozen contract file under intergrax/contracts/** fails the ratchet; the baseline is the HEAD measurement (138 Any, 357 object), the counts may only fall, and exceptions exist only as entries of typ_ratchet_baseline.v1.json with ticket and reason | none by construction (a ratchet passes on its HEAD baseline); prevention control under the Q-RW-07 ruling |
| NT-ROW-168 | FRZ-TYP-04 | FRZ_CODE_FIX_35 | T-AF2-RW-TYP | M | `tests/unit/architecture/test_rw_typ_ratchet.py` (NEW_PARENT_EXISTS) | NT-TYP-04-1 | a seeded new getattr(obj, name) dispatch (dynamic attribute name) in a temporary copy of a real frozen contract file under intergrax/contracts/** fails the ratchet; the ratchet also covers hasattr, setattr, importlib.import_module and __import__; baseline = HEAD measurement (getattr_dynamic 1, hasattr 3, setattr 0, importlib 0, __import__ 0); counts may only fall; exceptions only in typ_ratchet_baseline.v1.json | none by construction (a ratchet passes on its HEAD baseline); prevention control under the Q-RW-07 ruling; the seeded getattr dispatch tests TYP-04's own property (RWD7-03) |
| NT-ROW-169 | A4-004 | PHASE3, CODE_FIX_29 | PT-WA | S | `tests/unit/applications/test_wa_lkw_roster_search_provider.py` (NEW_PARENT_EXISTS) | NT-WA-23, NT-WA-25 (S); NT-WA-24 (C) | BLOCKER-1 of the PR #642 review (a consequence of the S2d builder refusal, DP-15): NT-WA-23 the default composed LKW roster builds with a stubbed LLM resolver and has no web_search_qualifier and no agent offering local.workspace.web_search_qualification; NT-WA-24 the product function build_local_workspace_agent_bindings with a search_provider slot enables the binding and the builder accepts that profile; NT-WA-25 invariant: binding.enabled == local_workspace_profile_has_search_provider(environment.integration_profile) for the module-constant manifest, build_local_workspace_manifest(settings) and the lkw_ac3_projection environment, plus an AST check that manifest.py and agent_factories.py call the one function | NT-WA-23 red at HEAD (manifest.py mounts WebSearchQualifierAgent without enabled, FACT_CHECK F-74) and at the implementation commit 09d2771e8 (the build raises IntegrationConfigurationError, implementer probe blocker1_probe.log, data from another session); NT-WA-25 red at HEAD (the derivation function is absent); NT-WA-24 is a control that fails on the absent API at HEAD |

**Basis-M rows and their conditions** (T-AF2-RW-PLAN-AUDIT section 8: ACCEPT WITH CONDITIONS). These rows are prevention controls; they are reported apart from the red-first count.

| Row | Item | Owner | Conditions |
| --- | --- | --- | --- |
| NT-ROW-061 | FRZ-DEBT-04 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check; (6) the seeded new-PARTIAL-row test is accepted |
| NT-ROW-123 | PDC-D1-R015 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check; (5) run the reuse test unmutated at HEAD first: if it fails, the row becomes basis B with that run as red-first; if it passes, the seeded duplicate-registration mutation is its negative test and the PARTIAL row closes as 'test added', not 'defect fixed' |
| NT-ROW-126 | EX-L-04 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-127 | EX-D1-03 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-128 | EX-D1-05 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-129 | EX-D1-06 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-130 | EX-D2-03 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-131 | EX-D3-02 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-132 | EX-D5-02 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-133 | EX-D7-02 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-134 | EX-D8-02 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-135 | EX-L-10 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-136 | EX-L-14 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-160 | FRZ-PLG-08 | T-AF2-RW-GATES | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-167 | FRZ-TYP-02 | T-AF2-RW-TYP | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |
| NT-ROW-168 | FRZ-TYP-04 | T-AF2-RW-TYP | (1) prevention control, reported apart from the red-first count and never counted toward phase check 3 as red-first; (2) the mutation is applied to the real gate input (a temporary copy of a real product file with an injected import edge, composition call or Any, or a seeded register row), never a toy fixture; the unmutated run passes at HEAD; (3) both runs (red on the mutation, green on the clean tree) are logged under .tmp/session/arch-freeze/<ticket>/red-first/ with command and digest; PT-Q02 re-runs the mutation check |

**TYP-05 reduction set (RWD7-01).** TYP-05 detector: a module-level mapping in intergrax/contracts/** whose >= 2 keys are str literals, str-constant names or UPPER_CASE constant names (or tuples of those), written as a dict literal (optionally wrapped in MappingProxyType/dict/frozendict), and used in the same module as a lookup (NAME[...] load, NAME.get(...), or `in NAME`) to select behaviour or an identity in place of a typed key. Dicts built inside functions (payloads) are out by construction. Script: .tmp/session/arch-freeze/rw-plan/work/typ05/measure_typ05.py At HEAD: 5 tables, target 0.

| File | Table | Line | Keys | Key kinds | Lookups |
| --- | --- | ---: | ---: | --- | --- |
| `intergrax/contracts/external_operation_redrive_trusted_effect_contracts.py` | `_QUAL_SANCTIONED_EXECUTION_BINDING` | 101 | 5 | tuple of str | get@164 |
| `intergrax/contracts/external_operation_redrive_trusted_effect_contracts.py` | `_CATALOG_TOOL_SANCTIONED_REDRIVE_BINDING` | 117 | 5 | literal | get@249, in@233 |
| `intergrax/contracts/governed_execution_result.py` | `_ACTION_TO_OPERATION` | 57 | 3 | UPPER_CASE constant name | get@84, get@194, get@311 |
| `intergrax/contracts/governed_execution_result.py` | `_EFFECT_CONTRACT_OPERATION_KEY_TO_PROVIDER_OPERATION` | 64 | 3 | literal | get@98 |
| `intergrax/contracts/governed_execution_result.py` | `_PROVIDER_OPERATION_TO_DECISION_ACTION` | 70 | 6 | literal | get@91 |

Not counted: `intergrax/contracts/agent_contract_section12.py` `DEFAULT_INPUT_SCHEMA` (:9): module-level str-keyed literal with no lookup in its module (schema default); `intergrax/contracts/agent_contract_section12.py` `DEFAULT_OUTPUT_SCHEMA` (:15): module-level str-keyed literal with no lookup in its module (schema default). Removed from the RW-TYP scope in v8: `intergrax/contracts/capability_metering/usage_event.py`: payload dict built from local values inside a function (audit RWD7-01: line 147); not a lookup table; `intergrax/contracts/runtime_execution_context.py`: payload dict built inside a function (line 181); not a lookup table; `intergrax/contracts/sanctioned_external_work_operation.py`: payload dict built inside a method (line 141); not a lookup table.

**Audit-flag owner binding of the 6 phase-3 conflicts** (REV-MASTER RM-03; machine-recorded in `audit_flags`, RWA-05):

| Finding | Owner | Effective flag | Auditing node | Binding |
| --- | --- | --- | --- | --- |
| A4-005 | T-AF2-DEPS-BATCH | yes | PT-Q02 | T-IGX-051-H ticket file carries the audit line when created (no prior plan) |
| A4-018 | T-AF2-RW-RUNTIME | yes | PT-Q02 | T-IGX-051-D ticket file carries the audit line when created (no prior plan) |
| A6-005 | PT-WD | yes | PT-Q02 (AUD-ENF-3) | W-D plan section 11 |
| A6-003 | PT-WD | yes | PT-Q02 (AUD-ENF-3) | W-D plan section 11 |
| A6-008 | PT-WD | yes | PT-Q02 (AUD-ENF-3) | W-D plan section 11 |
| A3-018 | PT-G02 | yes | PT-Q02 | WP-12 TICKETS_DRAFT row PT-G02 names 'yes - exact SHA (PT-Q02)': the W2 node's exact-SHA audit is folded into the W4 wave audit, which runs at G2 over every commit since a73d15ab3 |

## (e) Regression scope per ticket

**L1 on every train PR, the RW head and M3** (MEP-AF2 v2.1 section I, level L1 (every PR; RW head; M3)):

- `uv run python scripts/maintenance/check_intergrax_no_applications_imports.py`
- `uv run python scripts/maintenance/check_agent_registry_bypass.py`
- `uv run python scripts/maintenance/check_agents_no_tier3_imports.py`
- `uv run python scripts/maintenance/check_agents_no_vendor_sdk_imports.py`
- `uv run python scripts/maintenance/check_integration_vendor_imports.py`
- `uv run python scripts/maintenance/check_dependency_ownership.py`
- `uv run python -m scripts.governance.igx_gov ticket`
- `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py`
- `uv run python scripts/governance/arch_freeze_b1_closed_world_inventory.py --strict-certify`

HEAD-red gates are expected red on the train until their owner ticket lands. The registered expected-red set is exactly the entries of .tmp/session/arch-freeze/W0-ci-probe/expected_red_A01.json (section c, table 1); every other red is a planned entry with an owner and a registration moment (section c, table 2). The set may only shrink, except for a train-introduced red that has an explicit plan section I exception ruling (BASELINE:MANIFEST_DIGEST).

**L2** (MEP-AF2 v2.1 section I, level L2 (every PR, targeted; full at M3 in GitHub Actions)): `uv run pytest tests/unit -m "gate and not no_ci" (CI only; broad suite)`; `uv run pytest tests/unit/governance (one process, under the shared lock)`. Broad suites run in GitHub Actions; locally one pytest process at a time under the shared lock.

| Ticket | Targeted suites (existing paths verified at HEAD in the JSON) | Source |
| --- | --- | --- |
| PT-WA | `tests/unit/runtime/architecture/test_faudit_remediation.py`; `tests/unit/architecture/test_ebh_4_r1_r3_b6_typing_gate.py`; `tests/unit/applications/test_orchestration_wiring.py`; `tests/unit/applications/test_harness_host_runtime_llm.py`; `tests/qualification/harness_01/nexus_import_inventory.py`; `tests/unit/architecture/test_ebh_2d_d_application_contract_surface_recertification.py`; `tests/unit/architecture/test_ebh_2a_public_contract_boundary_gate.py`; `tests/unit/applications/shared/test_strict_host_dependency_concurrency_admission.py`; `tests/unit/applications/test_orchestration_environment_presets.py`; `applications/local_workspace_application/tests/host/test_local_workspace_host_smoke.py`; `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py`; `tests/unit/applications/local_workspace_application/test_lkw_host_runtime_composition_authority.py`; `tests/unit/applications/test_environment_read_allowlist_wiring.py`; `tests/unit/applications/test_application_startup_guards.py` | FC/WA/PLAN_T-IGX-043-G.md section 11 steps S1-S7; .tmp/session/arch-freeze/closed-world/bnd_tests_importing_t3_targets.txt (S7 collect-only, authorization WA-A12); contract typing ratchet check (RWD7-02): no Any/object/reflection metric in intergrax/contracts/** above the HEAD baseline; exceptions only in typ_ratchet_baseline.v1.json; and the TYP-05 detector (work/typ05/measure_typ05.py) reports no new string-keyed lookup table (still exactly the 5 of TYP05_REDUCTION); S3d LKW host regression (Amendment 1): the five files below from applications/local_workspace_application/tests/host and tests/unit/applications; they run red at base and head in the development environment (no LLM provider configured, ollama dependency absent; implementer logs blocker1_{base,head}{,_ollama}.log), so no pass is claimed; the reproducible proof is the stubbed-LLM roster build of NT-WA-23. test_lkw_production_host_bootstrap.py is also a firm path of PT-WB, which merges after PT-WA, so PT-WB re-runs it after its edit; KR-18 under P2 (plan S1): the register node and the EBH-2A gate, which reads tools/contracts/tool_availability.py but sees module-level imports only |
| PT-WB | `tests/unit/llm_adapters/routing/test_evaluating_adapter.py`; `tests/acceptance/llm_routing/test_concurrent_run_routing_isolation.py` | FC/WB/REGRESSION_NODES_VERIFIED.txt (209 node ids); contract typing ratchet check (RWD7-02): no Any/object/reflection metric in intergrax/contracts/** above the HEAD baseline; exceptions only in typ_ratchet_baseline.v1.json; and the TYP-05 detector (work/typ05/measure_typ05.py) reports no new string-keyed lookup table (still exactly the 5 of TYP05_REDUCTION) |
| PT-FC2 | `tests/unit/runtime/capacity/test_ecp_depth_gate.py`; `applications/legal_application/tests/host/test_legal_queue_worker_wiring.py`; `applications/dispute_sim_application/tests/host/test_dispute_sim_queue_worker_wiring.py`; `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py` | FC/FC-2/REGRESSION_NODES_VERIFIED.txt |
| PT-FC4 | `tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py`; `tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py` | FC/FC-4/PLAN_T-IGX-043-D.md section 6 |
| T-AF2-DEPS-BATCH | `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py`; `scripts/maintenance/check_dependency_ownership.py` | uv sync --frozen in CI; dependency-ownership gate; full L2 in CI (lockfile) |
| PT-E05 | `tests/unit/governance/test_iiac_negative_guards.py`; `scripts/governance/verify_iiac_policy_activation.py` | IIAC negative guards; verify_iiac_policy_activation dry run (non-certifying) |
| PT-D08 | `scripts/docs/check_docs_domain_pairs.py` | docs pair check |
| T-AF2-RW-RUNTIME | `tests/unit/llm_adapters`; `tests/qualification/ebh_5`; `tests/qualification/ebh_6`; `tests/unit/multimedia` | llm adapters, EBH-5/EBH-6 replays, multimedia |
| PT-WD | `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py`; `tests/qualification/ent_audit_x/unit/test_od_ent_02_verifiers.py` | FC/WD/regression_nodes.txt |
| PT-FC1 | `tests/unit/architecture/test_ebh_2d_c_r6_canonical_agent_routing_decision_closure.py` | FC/FC-1/PLAN_T-IGX-043-A.md section 6 |
| PT-FC2-E | `scripts/maintenance/check_production_capacity_adapters.py`; `scripts/maintenance/check_sandbox_policy_wiring.py` | the two gate scripts run once each after PT-FC2 |
| T-AF2-RW-RECOVERY | `tests/unit/queueing`; `tests/unit/hosting`; `tests/unit/background_tasks`; `tests/unit/runtime/long_running`; `tests/qualification/pdc_x/xd` | queueing, hosting, background tasks, long-running, XD pairs |
| T-AF2-A4007-CALLERS | `tests/unit/applications/contracts/test_agent_resolution.py`; `tests/unit/applications/test_migration_wiring.py`; `tests/unit/applications/test_environment_diff_wiring.py`; `tests/unit/applications/test_package_wiring.py`; `tests/unit/applications/test_agent_binding_conformance.py`; `tests/unit/applications/test_agent_certification_gate.py`; `tests/unit/applications/test_agent_runtime_governance_wiring.py`; `tests/unit/applications/test_application_capability_projection.py`; `tests/unit/applications/test_capability_graph_deploy_gate.py`; `tests/unit/applications/test_ac5_phase3_factory_e2e.py`; `tests/unit/applications/test_production_factory_invocation_ac5.py`; `tests/unit/applications/test_production_registry_authority_guards_ac3.py`; `tests/unit/applications/test_prod_q_r1_r7_q_p2_c_package_closure_authority.py`; `tests/unit/applications/governed_contractor/test_prod_q_r1_pis_02_provider_invocation_lifecycle_binding.py`; `tests/unit/architecture/test_ebh_2d_c_r2_canonical_agent_factory_contract.py`; `tests/unit/architecture/test_ebh_2d_c_r3_tier2_agent_factory_result_contract.py`; `tests/unit/architecture/test_ebh_2e_ar1_a_r1_r1_install_authority_closure.py`; `tests/unit/architecture/test_ebh_2e_ar1_a_r1_r1_r1_install_to_rollback_contract_authority.py`; `tests/unit/architecture/test_ebh_2e_ar1_a_r1_revision_bound_agent_contract_authority.py`; `tests/unit/architecture/test_ebh_2e_r4_r1_r1_canonical_agent_contract_ownership.py`; `tests/unit/architecture/test_ebh_2e_r4_r1_structural_replaceability_and_metadata_resolution.py`; `tests/unit/runtime/architecture/test_platform_execution_unification_u3_agent_plugin_execution_closure.py` | draft plan section 11: V1 the three new test files (one command); V2 the 23 existing files below (one command; 180 test definitions); tests/unit/applications/test_wb_import_allowlist.py (the 23rd file) exists only on the train (created by PT-WB) and is listed here, not in paths, because paths are checked at the facts SHA; V3 uv run python scripts/maintenance/check_application_migrations.py (exit 0, exit code to the PR body); V4 the git-grep caller census at step 0 and at the end; V5 ruff and pyright on the changed files; V6 igx_gov ticket; tests/unit/scripts/test_check_application_migrations.py is not regression: it is no_ci and points at a script path that does not exist (draft finding) |
| PT-FC3 | `tests/unit/model_inference/test_remote_vision_adapters.py`; `tests/unit/rag/vectorstore/test_integration_vectorstore_bootstrap.py`; `tests/qualification/config_x` | FC/FC-3/regression_nodes.txt |
| T-AF2-RW-SUPPLY | `tests/unit/agent_distribution`; `tests/unit/core/plugins`; `tests/unit/runtime/security`; `tests/qualification/pdc_x/xd`; `scripts/maintenance/check_immutable_security_audit_trail.py`; `scripts/maintenance/check_sandbox_policy_wiring.py` | agent distribution, plugins, runtime security, PDC D1/D6 qualification, audit-trail gate; contract typing ratchet check (RWD7-02): no Any/object/reflection metric in intergrax/contracts/** above the HEAD baseline; exceptions only in typ_ratchet_baseline.v1.json; and the TYP-05 detector (work/typ05/measure_typ05.py) reports no new string-keyed lookup table (still exactly the 5 of TYP05_REDUCTION) |
| T-AF2-RW-GATES | `tests/unit/scripts`; `tests/unit/applications/test_scenario_runtime_baseline.py`; `tests/unit/llm_adapters/test_fixture_strict_adapter_parity.py`; `tests/qualification/plug_03`; `tests/qualification/trace_x` | gate scripts, scenario baseline, strict adapter parity, plug_03, TRACE-X |
| T-AF2-RW-TRUST | `tests/unit/context`; `tests/unit/llm`; `tests/unit/rag`; `tests/integration/rag`; `tests/qualification/pdc_x/d2_repair` | context, llm, rag (extras profile for integration) |
| PT-FC3-F | `tests/qualification/config_x` | FC/FC-3/regression_nodes_F.txt |
| PT-WF | `tests/qualification/tenant_x`; `tests/unit/compat/langchain/test_documents.py`; `tests/integration/decision_system/test_concurrent_finalization.py` | FC/WF/regression_nodes.txt; contract typing ratchet check (RWD7-02): no Any/object/reflection metric in intergrax/contracts/** above the HEAD baseline; exceptions only in typ_ratchet_baseline.v1.json; and the TYP-05 detector (work/typ05/measure_typ05.py) reports no new string-keyed lookup table (still exactly the 5 of TYP05_REDUCTION) |
| T-AF2-RW-TENANT | `tests/qualification/tenant_x`; `tests/unit/runtime/sandbox`; `tests/unit/tools/providers`; `tests/unit/agent_distribution`; `tests/qualification/mcp_server_x`; `tests/unit/applications/test_mcp_identity_mode_fail_closed.py`; `tests/unit/rag/governance` | tenant suites, sandbox, storage tools, agent distribution, MCP identity |
| T-AF2-RW-TYP | `tests/unit/architecture/test_ebh_4_r1_r3_b6_typing_gate.py` | typing gates |
| PT-T01 | `tests/qualification/tenant_x`; `tests/qualification/trace_x` | NT-01..NT-16 at the train head (red-first at B recorded by PT-B01) |

**Contract typing ratchet: one baseline (RWD7-02, RWD7-03).** Scope `intergrax/contracts/**`; source closed-world/typ_measure.json (repo_head 7960ae7f6; product tree identical at a7f023615, F-50). Any 138, object annotations 357, getattr_dynamic 1, hasattr 3, setattr 0, importlib 0, __import__ 0, TYP-05 lookup tables 5 (target 0). ONE baseline: the HEAD measurement above. No train ticket may raise any of these counts. An exception is valid only as an entry in tests/unit/architecture/typ_ratchet_baseline.v1.json naming the ticket, the file, the metric and the reason; no other grandfathering.

Wave-packet check (RWD7-02): at the ticket head, a run of the TYP measurement over intergrax/contracts/** reports no metric above RATCHET_BASELINE (Any 138, object 357, getattr_dynamic 1, hasattr 3, setattr 0, importlib 0, __import__ 0); any increase is either removed or listed in typ_ratchet_baseline.v1.json with this ticket and the reason; and the TYP-05 detector reports no lookup table beyond the 5 of TYP05_REDUCTION (no new module-level string-keyed lookup table in intergrax/contracts/**). Until RW-TYP has merged, the check is run by the implementer with .tmp/session/arch-freeze/closed-world/typ_measure.py (Any, object, reflection) and .tmp/session/arch-freeze/rw-plan/work/typ05/measure_typ05.py (TYP-05 detector), both over a git archive of intergrax/contracts at the ticket head, and recorded in the PR body; the wave packet lists both per ticket.

| Contract-editing ticket | Contract files it edits |
| --- | --- |
| PT-WA | 6 new contract files (intergrax/contracts/{application_graph_topology,delegation_budget_policy,host_orchestration_wiring_port,host_orchestration_environment,rag_retrieval_source_scope,workspace_tool_ids}.py); v13: rag_retrieval_source_scope.py typed with JsonValue, no Any (DR-4, no baseline exception entry); intergrax/tools/contracts/tool_availability.py (new, P2) is outside the ratchet scope intergrax/contracts/** and carries 0 Any/object/reflection, checked in its S6 pyright step (plan 7b, review LOW-5) |
| PT-WB | intergrax/contracts/profile_projection.py (new), intergrax/contracts/host_profile_slices.py |
| PT-WF | intergrax/contracts/decision_authoritative_exposure.py and, if JG-5b is decided, provider_invocation*.py and enterprise_reliability/admission_boundary.py |
| T-AF2-RW-SUPPLY | intergrax/contracts/agent_run.py |

Regression rows for the non-train owners of phase-3 findings (RWA-04):

| Node | Findings | Paths (exist at HEAD) | Bound in |
| --- | --- | --- | --- |
| PT-G01 | A4-011, A5-001 | `tests/unit/runtime/architecture/test_faudit_remediation.py`; `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py`; `scripts/maintenance/check_intergrax_no_applications_imports.py`; `scripts/maintenance/check_agent_registry_bypass.py` | WP-12 TICKETS_DRAFT row PT-G01 (W2, NONPRODUCT); audit PT-Q02 |
| PT-G02 | A3-008, A3-018 | `tests/qualification/state_x`; `tests/qualification/ent_audit_x/unit/test_frz_coverage_matrix_complete.py` | WP-12 TICKETS_DRAFT row PT-G02 (W2); audit PT-Q02 |
| PT-E02 | A6-002 | `tests/unit/governance`; `scripts/governance/verify_plan_approval.py` | WP-12 TICKETS_DRAFT row PT-E02 (W1); audit PT-AUD-ENF1 |
| PT-E03 | A6-001, A6-006 | `tests/qualification/ent_audit_x/acceptance` | WP-12 TICKETS_DRAFT row PT-E03 (W2); audit PT-AUD-ENF1 |

Per-finding regression map for the 18 phase-3 findings (RM-03, RWA-04: every entry names >= 1 path that exists at HEAD):

| Finding | Owner | Regression paths | Node / note |
| --- | --- | --- | --- |
| A4-003 | PT-WA | `tests/unit/runtime/architecture/test_faudit_remediation.py`; `tests/unit/architecture/test_ebh_4_r1_r3_b6_typing_gate.py` + L1 | - |
| A4-004 | PT-WA | `tests/unit/architecture/test_ebh_2e_r4_agent_llm_composition_injection.py`; `scripts/maintenance/check_agent_registry_bypass.py` + L1 | - |
| A4-005 | T-AF2-DEPS-BATCH | `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py`; `tests/unit/test_dependency_governance.py` + L1 | - |
| A4-011 | PT-G01 | `tests/unit/runtime/architecture/test_faudit_remediation.py`; `scripts/maintenance/check_intergrax_no_applications_imports.py` + L1 | - |
| A4-018 | T-AF2-RW-RUNTIME | `tests/unit/multimedia`; `tests/qualification/ebh_5` + L1 | - |
| A2-009 | PT-WA | `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py`; `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py` + L1 | - |
| A4-006 | PT-T01 | `tests/qualification/tenant_x`; `scripts/maintenance/check_runtime_event_tenant_propagation.py` + L1 | - |
| A6-007 | PT-WD | `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py`; `tests/qualification/ent_audit_x/integration/test_changed_since_git.py` + L1 | tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py::test_unknown_platform_path_fail_closed (WD regression_nodes.txt tier 1) |
| A3-008 | PT-G02 | `tests/qualification/ent_audit_x/unit/test_frz_coverage_matrix_complete.py`; `tests/qualification/roadmap_replay_x` + L1 | - |
| A2-038 | PT-E05 | `tests/unit/governance/test_iiac_negative_guards.py` + L1 | - |
| A6-001 | PT-E03 | `tests/qualification/ent_audit_x/acceptance` + L1 | - |
| A6-002 | PT-E02 | `tests/unit/governance`; `scripts/governance/verify_plan_approval.py` + L1 | - |
| A6-006 | PT-E03 | `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_bnd_01.py` + L1 | - |
| A6-005 | PT-WD | `tests/qualification/ent_audit_x/unit/test_frz_coverage_matrix_complete.py`; `tests/qualification/ent_audit_x/unit/test_validators_all_artifacts.py` + L1 | - |
| A5-001 | PT-G01 | `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py` + L1 | - |
| A3-018 | PT-G02 | `tests/qualification/state_x` + L1 | - |
| A6-003 | PT-WD | `tests/qualification/ent_audit_x/unit/test_ten_vbn_negative.py`; `tests/qualification/evol_x/gates/test_evol_x_selection.py` + L1 | tests/qualification/ent_audit_x/unit/test_ten_vbn_negative.py::test_changed_since_fail_closed_unmapped_evol_path (WD regression_nodes.txt tier 1) |
| A6-008 | PT-WD | `tests/qualification/ent_audit_x/unit/test_certifying_and_bundle.py`; `tests/qualification/ent_audit_x/unit/test_baseline_promotion.py` + L1 | - |

## (f) Dedupe

| Id | Defect | Items covering it | One owner | Why |
| --- | --- | --- | --- | --- |
| D-01 | UAEP path without context engine / LLM adapter (B-1 R03 v6) | B-1 'extra' row (no ID); B-1 table row labelled A4-031 (HITL); B-2 A4-031; EQOA-A4-031 (FC-1 part C2) | PT-FC1 | probe docstring pp_gov.py:758-759 maps v6 to A4-031; FC-1 owns A4-031 (part C2 now firm). The B-1 results table rows A4-026/A4-027/A4-037/A4-031 carry shifted titles; the PREREG table (PREREG.md:54-62) and the probe docstrings are used (Q-RW-11). |
| D-02 | task-memory store opened through an env var under a disabling profile | B-1 X-A4-024; EQOA-A4-024 (WP-12 lists it under PT-WB) | PT-FC1 | decision D-FC1-01; the W-B plan section 1a records no W-B code (NT-FC1-16..19). |
| D-03 | three importlib seams accept any module path | B-1 A4-007; W-A Part F (Q-WA-08); W-B Part E (DP-21); T-IGX-051-I (A4-007 b1/b2 callers) | T-AF2-A4007-CALLERS | v16: W-B Part E (merged) created import_allowlist.py, the loader parameters and the b4 scoreboard fix, but left seams b1/b2 (the product callers) as a residual (PT-WB ticket checkpoint); T-IGX-051-I threads the allowlist and fail-closed posture through those callers and owns the A4-007 closure. W-B Part E creates the single allowlist module import_allowlist.py and covers all three seams plus wiring.py and mcp_import_guard.py (same loader class); W-A Part F is separable. Both plans write agent_resolution.py, migration_wiring.py and scoreboard.py, so one owner removes the file overlap. |
| D-04 | missing or client tenant becomes 'default' in Tier-3 serving | FRZ-TEN-10 (matrix: T-IGX-051-A); B-1 A4-027; W-F Part JA | PT-WF | W-F JA already names the 20 files (serving schemas, routers, factories) and NT-WF-A07..A16; RW-TENANT keeps the MCP client-asserted tenant (R-P8-01) only. |
| D-05 | direct ollama SDK use outside the provider boundary | EQOA-A4-018 (WP-12: PT-WF); W-F Part JF; plan E WP12-Q02 -> T-AF2-RW-RUNTIME; FRZ-CTR-02, FRZ-CTR-06, FRZ-CFG-06 (SDK part), FRZ-REG-09 (banned list) | T-AF2-RW-RUNTIME | plan section E assigns A4-018 to T-AF2-RW-RUNTIME and the matrix assigns CTR-02/06 to T-IGX-051-D; W-F JF's three files and the NT-WF-F01..F03 design move with it. The banned-list extension (REG-09) stays with RW-GATES. |
| D-06 | dependency-ownership gate red (jsonschema, tzdata) | EQOA-A4-005 (WP-12: PT-WF); FRZ-DEBT-02; FRZ-DEBT-03; PDC handoff 5.9 | T-AF2-DEPS-BATCH | plan section E WP12-Q02. |
| D-07 | tenant quota disables itself on a malformed value | F-D3-12 (A-12 set, RW-SUPPLY); PDC-D3-R018 (plan E PARTIAL table: RW-RUNTIME) | T-AF2-RW-SUPPLY | one defect in one file (llm_adapters/governance/quota.py); the PARTIAL row moves with its root cause. The RW-RUNTIME scope does not include quota.py. |
| D-08 | audit trail without tamper evidence | EX-D6-01; F-D6-02 | T-AF2-RW-SUPPLY | F-D6-02 is the finding behind EX-D6-01 (PDC exception register). |
| D-09 | production composition of PDC scope features (retire, skill lifecycle, recurrence, artifact store) | W-F Part JD (held, Q-WF-02); EX-D1-05, EX-D1-06, EX-D7-02, EX-D8-02 (DS-EXC: ACCEPTABLE_SCOPED -> RW-GATES); EX-D8-01 (DS-EXC: RW-SUPPLY) | T-AF2-RW-GATES (NC gates) / T-AF2-RW-SUPPLY (EX-D8-01) | DS-EXC was decided with the recommended dispositions (OPERATOR_PACKET_1); JD is not train scope unless the Operator reverses that (Q-RW-06). |
| D-10 | production host composes no D6 controls | EX-D6-02 (REQUALIFICATION); F-D6-04 (plan E security row) | T-AF2-RW-SUPPLY | same finding (PDC exception register row EX-D6-02 = F-D6-04); requalification first, fix if confirmed. |
| D-11 | Q2-D2 typed seam and F02-14 isinstance | FRZ-TYP-01, FRZ-TYP-03, FRZ-RPL-04 (matrix C2: T-IGX-051-G); W-F Parts JB, JC | PT-WF | W-F JB/JC have the write set and NT-WF-B01..B06, C01..C05; RW-TYP keeps TYP-05 and the OD-AF-11 ratchet. These rows are not among the 35. |
| D-12 | FRZ rows the matrix gave to T-IGX-043-J without W-F code | FRZ-BND-06; FRZ-OWN-02; FRZ-PLG-01; FRZ-PLG-02; FRZ-PLG-06; FRZ-PLG-07; FRZ-PRD-08 | T-AF2-RW-GATES | W-F plan section 4, last row: 32 of 37 rows have no W-F code; their gates and inventories need one train owner. |
| D-13 | tenant-propagation emitter list | EQOA-A4-006; B-1 A4-006 (RW: no) | PT-T01 | NONPRODUCT script fix; WP-12 row PT-T01. |
| D-14 | CHANGED-SINCE evol_x mapping | EQOA-A6-003; B-2 X-A6-003 | PT-WD | same defect (W-D Part B). |
| D-15 | G-09 untenanted evaluation registry | EX-L-09 (G-09 part); B-1 X-A5-011; FRZ-TEN-01 | PT-WF | DP-23; EX-L-09 stays owned by RW-TENANT for its G-12 part. |
| D-16 | registry-bypass imports in agents/ | EQOA-A4-004; B-1 X-A4-004 | PT-WA | same 10-file census. |
| D-17 | integrations._shared leakage outside intergrax/ | EQOA-A2-009; EQOA-A5-001; FRZ-BND-04; B-2 A2-009 | PT-WA (A2-009, BND-04: imports moved) / PT-G01 (A5-001: gate scan roots) | two distinct properties: the imports (product) and the gate scope (non-product). |
| D-18 | inbound verifier no-op default | EQOA-A4-026; B-1 A4-026 (table row titled 'silent-substitution detector scope'); FRZ-SEC-04 | PT-FC3 | probe v3 docstring pp_gov.py:466-467. |
| D-19 | HITL reject/escalate persistence | EQOA-A4-037; B-1 A4-037 (table row titled 'inbound verifier') | PT-FC1 | probe v5 docstring pp_gov.py:640-641. |

## (g) Coverage check

Every EX, PARTIAL, phase-3, FRZ CODE_FIX and FRZ C2 ID has exactly one disposition: an owner train ticket (kind `RW...`) or a non-RW node (kind `NON-RW: ...`). Contributors help but do not own. The table also lists the section E items outside those universes, the campaign items and the known-red rows. The 6 NONPRODUCT C2 rows (FRZ-REG-04, FRZ-DOC-06, FRZ-FRZ-05..08) are owned outside the train (PT-G02, T-AF2-MANIFESTS); the 5 PRODUCT C2 rows have train owners and NT rows (RWA-02).

| ID | Owner | Kind | Contributors | Note |
| --- | --- | --- | --- | --- |
| A-01 | T-AF2-RW-SUPPLY | RW | - | - |
| A2-009 | PT-WA | RW (Part E + S2d) | PT-G01 | Part E firm (OPERATOR_PACKET_3, Q-WA-05 option A) |
| A2-012 | PT-D08 | RW (DOC_PATCH on product-path docs; no code, no negative test) | - | - |
| A2-038 | PT-E05 | RW (MATERIAL train member) | - | - |
| A3-008 | PT-G02 | NON-RW: W2 evidence surface / STATE-X citation repair | - | - |
| A3-018 | PT-G02 | NON-RW: W2 evidence surface / STATE-X citation repair | - | - |
| A4-002 | PT-FC4 | RW | - | - |
| A4-003 | PT-WA | RW | - | - |
| A4-004 | PT-WA | RW | - | - |
| A4-005 | T-AF2-DEPS-BATCH | RW | - | plan E WP12-Q02; WP-12 PT-WF row superseded |
| A4-006 | PT-T01 | RW (NONPRODUCT train member) | - | - |
| A4-007 | T-AF2-A4007-CALLERS | RW | PT-WB | v16: seams b1/b2 (product callers of the allowlist and fail-closed parameters) close in T-IGX-051-I; PT-WB Part E (DP-21, merged) contributed the loaders, import_allowlist.py and b4 (scoreboard); W-A Part F dropped (D-03) |
| A4-009 | PT-FC1 | RW | - | - |
| A4-011 | PT-G01 | NON-RW: W2 gate/seam-ledger node | - | - |
| A4-015 | PT-FC1 | RW | - | - |
| A4-017 | PT-FC3 | RW | - | - |
| A4-018 | T-AF2-RW-RUNTIME | RW | - | plan E WP12-Q02; PT-WF Part JF moved here |
| A4-019 | OPEN-OPERATOR | OPEN (no owner yet; Operator item, Q-F-2 ruling) | PT-FC3, PT-FC3-F | v18: PT-FC3 removes the silent defaults (NT-FC3-E/V/N tests), but the recorded closure condition (the CONFIG-X scan flags the vendor-slug defaults and is green after the fix) is not met: T-IGX-043-F's narrowed scan is silent on the embedding default, the vector-store fallback and the guardrail stub. Not closed by T-IGX-043-F; owner to be named by the Operator (lane 4 to 7 packet) |
| A4-020 | PT-WB | RW-CLASSIFY-FIRST (Part C; code only for a file ruled a bypass) | - | - |
| A4-025 | OPEN-OPERATOR | OPEN (no owner yet; Operator item, Q-F-3 ruling) | PT-FC3-F | v18: by the Operator ruling (option 3a) T-IGX-043-F fails the gate only for hits on the eight FC-3 paths; ambient environment reads outside the inventoried CONFIG-X surfaces are recorded (a printed list, one EDER row per real hit), not failing. A4-025 is not delivered; it stays open or moves to a gate ticket (Q-F-3 default) |
| A4-026 | PT-FC3 | RW | - | - |
| A4-027 | PT-WF | RW | - | - |
| A4-030 | PT-FC2 | RW | - | - |
| A4-031 | PT-FC1 | RW (conditional part C2 becomes firm: V6 CONFIRMED) | - | B-1 R03 v6; B-1 table mislabels the row (D-01) |
| A4-032 | PT-FC2 | RW | PT-FC3 | parts a, b in FC-2; part c (guardrail) in FC-3 |
| A4-035 | PT-FC2 | RW | - | - |
| A4-036 | PT-FC2 | RW | - | - |
| A4-037 | PT-FC1 | RW | - | - |
| A4-040 | PT-WB | RW (if ruled in scope: Part D; Part G for the strict serving command) | T-AF2-RW-SUPPLY | B-2 promotion trigger met statically; the strict serving command is shared with RW-SUPPLY (serialized); a scope ruling (Operator Q4) lands in W-B Part G or RW-SUPPLY, not in a third place |
| A4-041 | PT-D08 | RW (DOC_PATCH on product-path docs; no code, no negative test) | - | - |
| A4-044 | PT-FC1 | RW | - | - |
| A5-001 | PT-G01 | NON-RW: W2 gate/seam-ledger node | - | - |
| A6-001 | PT-E03 | NON-RW: W2 | - | - |
| A6-002 | PT-E02 | NON-RW: W1 | - | - |
| A6-003 | PT-WD | RW | - | - |
| A6-005 | PT-WD | RW | - | - |
| A6-006 | PT-E03 | NON-RW: W2 | - | - |
| A6-007 | PT-WD | RW | - | - |
| A6-008 | PT-WD | RW | - | - |
| A8-020 | PT-WF | RW | - | - |
| EX-D1-01 | T-AF2-RW-SUPPLY | RW | - | - |
| EX-D1-02 | T-AF2-RW-SUPPLY | RW | - | - |
| EX-D1-03 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| EX-D1-04 | T-AF2-RW-SUPPLY | RW | - | - |
| EX-D1-05 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| EX-D1-06 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| EX-D1-07 | T-AF2-RW-RECOVERY | RW | - | - |
| EX-D2-01 | T-AF2-RW-TRUST | RW | - | - |
| EX-D2-02 | T-AF2-RW-TRUST | RW | - | - |
| EX-D2-03 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| EX-D2-04 | T-AF2-MANIFESTS | NON-RW: ACCEPTABLE_OUTSIDE_FROZEN_SCOPE | - | - |
| EX-D2-05 | T-AF2-RW-TRUST | RW | - | - |
| EX-D2-06 | T-AF2-RW-TENANT | RW (DS-EXC option: STRICT forbids unscoped memory) | - | - |
| EX-D2-07 | T-AF2-RW-TRUST | RW | - | - |
| EX-D3-01 | T-AF2-RW-RUNTIME | RW | - | - |
| EX-D3-02 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| EX-D3-03 | T-AF2-RW-RUNTIME | RW | - | - |
| EX-D3-04 | T-AF2-RW-GATES | RW | - | - |
| EX-D4-01 | T-AF2-RW-RECOVERY | RW | - | - |
| EX-D4-02 | T-AF2-MANIFESTS | NON-RW: DS-EXC claim (OS child watch NOT_CLAIMED) | - | - |
| EX-D4-03 | T-AF2-MANIFESTS | NON-RW: ACCEPTABLE_OUTSIDE_FROZEN_SCOPE | - | - |
| EX-D4-04 | T-AF2-RW-TENANT | RW | - | - |
| EX-D4-05 | T-AF2-RW-GATES | RW (DS-EXC option: flag moves to the application profile) | - | - |
| EX-D5-01 | T-AF2-MANIFESTS | NON-RW: DS-EXC claim (groundedness NOT_CLAIMED; D5-R005 reworded by PT-D07) | PT-D07 | - |
| EX-D5-02 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| EX-D5-03 | T-AF2-RW-GATES | RW | - | - |
| EX-D5-04 | T-AF2-MANIFESTS | NON-RW: ACCEPTABLE_OUTSIDE_FROZEN_SCOPE | - | - |
| EX-D5-05 | T-AF2-RW-TENANT | RW | - | - |
| EX-D6-01 | T-AF2-RW-SUPPLY | RW | - | - |
| EX-D6-02 | T-AF2-RW-SUPPLY | RW-REQUAL-FIRST | PT-Q12 | same defect as F-D6-04: requalify first; the fix is RW-SUPPLY if the production host composes no D6 controls |
| EX-D6-03 | T-AF2-RW-SUPPLY | RW | - | - |
| EX-D6-04 | PT-Q12 | NON-RW: requalification (mcp extra) | PT-Q11 | - |
| EX-D7-01 | T-AF2-RW-RECOVERY | RW | - | - |
| EX-D7-02 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| EX-D7-03 | T-AF2-RW-RECOVERY | RW (DS-EXC option: durable producers set an identity) | - | - |
| EX-D8-01 | T-AF2-RW-SUPPLY | RW | - | retention + AUDIT_EVIDENCE producer; PT-WF Part JD (held) does not also compose it (D-09) |
| EX-D8-02 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| EX-D8-03 | T-AF2-RW-TENANT | RW | - | - |
| EX-D8-04 | T-AF2-RW-TENANT | RW | - | - |
| EX-G-01 | T-AF2-VBN-DOSSIER | NON-RW: bounded claim | - | - |
| EX-G-02 | T-AF2-MANIFESTS | NON-RW: ACCEPTABLE_OUTSIDE_FROZEN_SCOPE | - | - |
| EX-G-03 | PT-G02 | NON-RW: DS-EXC N/A WITH EVIDENCE (evidence file) | - | - |
| EX-G-04 | PROD-Q | NON-RW: TRANSFER_TO_LIVE_PROD_Q (limited to named PDC rows) | - | - |
| EX-G-05 | PT-Q04 | NON-RW: test reds (DP-60) | PT-Q05 | - |
| EX-G-06 | PT-Q11 | NON-RW: requalification | - | - |
| EX-G-07 | T-AF2-MANIFESTS | NON-RW: ACCEPTABLE_OUTSIDE_FROZEN_SCOPE | - | - |
| EX-L-01 | T-AF2-RW-GATES | RW | - | - |
| EX-L-02 | T-AF2-RW-TENANT | RW | - | - |
| EX-L-03 | T-AF2-RW-RECOVERY | RW | - | - |
| EX-L-04 | T-AF2-RW-GATES | RW-NC-GATE (DS-EXC: accept as dormant; needs non-composition gate + NOT_COMPOSED mark) | T-AF2-MANIFESTS | - |
| EX-L-05 | T-AF2-RW-TRUST | RW | - | - |
| EX-L-06 | T-AF2-RW-RECOVERY | RW | - | - |
| EX-L-07 | T-AF2-RW-RUNTIME | RW | - | - |
| EX-L-08 | T-AF2-RW-RUNTIME | RW | - | - |
| EX-L-09 | T-AF2-RW-TENANT | RW | PT-WF | G-12 install-over-occupied CAS here; G-09 part fixed by PT-WF Part JA (DP-23) |
| EX-L-10 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| EX-L-11 | T-AF2-RW-GATES | RW | - | - |
| EX-L-12 | PT-Q12 | NON-RW: requalification (verify T-IGX-041-P) | PT-Q11 | - |
| EX-L-13 | T-AF2-RW-GATES | RW | - | - |
| EX-L-14 | T-AF2-RW-GATES | RW-NC-GATE (ACCEPTABLE_SCOPED_WITH_EVIDENCE through reachability gates) | T-AF2-MANIFESTS | - |
| F-D2-11 | T-AF2-RW-TENANT | RW | - | tenant part (EX-D2-03 row) |
| F-D3-12 | T-AF2-RW-SUPPLY | RW | - | fail-open set A-12 |
| F-D3-16 | PT-D07 | NON-RW: docs | PT-D06 | - |
| F-D5-12 | T-AF2-RW-SUPPLY | RW | - | fail-open set A-12 |
| F-D6-02 | T-AF2-RW-SUPPLY | RW | - | same defect as EX-D6-01 (tamper evidence) |
| F-D6-03 | T-AF2-RW-SUPPLY | RW | - | D6-R007 |
| F-D6-04 | T-AF2-RW-SUPPLY | RW | - | same defect as EX-D6-02 |
| F-D6-09 | T-AF2-RW-SUPPLY | RW | - | fail-open set A-12 |
| F-D6-10 | T-AF2-RW-SUPPLY | RW | - | PDC-D6-R024 |
| F-D6-11 | T-AF2-RW-SUPPLY | RW | - | fail-open set A-12 |
| F-D6-14 | T-AF2-RW-SUPPLY | RW | - | D6-R007 |
| F-D8-14 | T-AF2-RW-SUPPLY | RW | - | fail-open set A-12 |
| FRZ-BND-04 | PT-WA | RW | PT-G01 | - |
| FRZ-BND-05 | PT-FC1 | RW | PT-WB, PT-G01 | A4-015 RuntimeContext construction is FC-1's; A4-020 switches are W-B Part C |
| FRZ-BND-06 | T-AF2-RW-GATES | RW | PT-G02 | matrix named T-IGX-043-J; the W-F plan writes no code for it (WF plan 4, last row) |
| FRZ-CFG-02 | PT-FC3 | RW | T-AF2-RW-RUNTIME, PT-WB, T-AF2-RW-TENANT | - |
| FRZ-CFG-06 | PT-FC3 | RW | T-AF2-RW-RUNTIME, PT-WB, T-AF2-RW-TENANT | - |
| FRZ-CFG-08 | PT-FC3-F | RW | - | - |
| FRZ-CTR-01 | T-AF2-RW-RUNTIME | RW | PT-WB | - |
| FRZ-CTR-02 | T-AF2-RW-RUNTIME | RW | PT-WB | - |
| FRZ-CTR-04 | PT-Q12 | NON-RW: RESOLVED_AT_BASE, requalification only | PT-D01, PT-D09 | - |
| FRZ-CTR-06 | T-AF2-RW-RUNTIME | RW | PT-WB | - |
| FRZ-DEBT-02 | T-AF2-DEPS-BATCH | RW | PT-R01, PT-E03 | dependency part; EDER part stays with PT-R01..R03 |
| FRZ-DEBT-03 | T-AF2-DEPS-BATCH | RW | PT-R01, PT-E03 | dependency part; EDER part stays with PT-R01..R03 |
| FRZ-DEBT-04 | T-AF2-RW-GATES | RW | PT-G02 | - |
| FRZ-DOC-05 | PT-Q12 | NON-RW: RESOLVED_AT_BASE, requalification only | PT-D01, PT-D09 | - |
| FRZ-DOC-06 | T-AF2-MANIFESTS | NON-RW: C2 NONPRODUCT (DP-40 manifests) | - | - |
| FRZ-EXE-02 | PT-FC1 | RW | T-AF2-RW-SUPPLY, PT-WA | - |
| FRZ-FRZ-05 | T-AF2-MANIFESTS | NON-RW: C2 NONPRODUCT (DP-40 manifests) | - | - |
| FRZ-FRZ-06 | T-AF2-MANIFESTS | NON-RW: C2 NONPRODUCT (layer manifest; external edge: after PT-WA merges into the train) | - | - |
| FRZ-FRZ-07 | T-AF2-MANIFESTS | NON-RW: C2 NONPRODUCT (DP-40 manifests) | - | - |
| FRZ-FRZ-08 | T-AF2-MANIFESTS | NON-RW: C2 NONPRODUCT (composition findings A4-015/A4-037; external edge: after PT-FC1 merges into the train) | - | - |
| FRZ-FRZ-10 | T-B7-PFP-ACTIVATE | NON-RW: T-B7-PFP-ACTIVATE is a W5 node (plan section J, W5 row; before LOCK) and T-B7-PFP-VERIFY a W8 node; the matrix C note 'W7/W8' names the pair (T-IGX-052-* letter not allocated) | T-B7-PFP-VERIFY | - |
| FRZ-OWN-02 | T-AF2-RW-GATES | RW | T-AF2-MANIFESTS, PT-D09 | contract-census gate; ratification entries in the contract manifest |
| FRZ-OWN-03 | PT-WB | RW | PT-FC1, T-AF2-RW-GATES | - |
| FRZ-OWN-05 | PT-WB | RW | PT-FC1, T-AF2-RW-GATES | - |
| FRZ-PLG-01 | T-AF2-RW-GATES | RW | PT-G02 | - |
| FRZ-PLG-02 | T-AF2-RW-GATES | RW | PT-G02 | - |
| FRZ-PLG-05 | PT-FC1 | RW | T-AF2-RW-SUPPLY, PT-WA | - |
| FRZ-PLG-06 | T-AF2-RW-GATES | RW | PT-G02 | - |
| FRZ-PLG-07 | T-AF2-RW-GATES | RW | PT-G02 | - |
| FRZ-PLG-08 | T-AF2-RW-GATES | RW-NC-GATE for the RETIRE and skill-lifecycle parts only (C2 PRODUCT; dormant code, DS-EXC EX-D1-05/EX-D1-06); UPGRADE part: DP-30 exception decided (OPERATOR_PACKET_3, Q-RW-19) | T-AF2-MANIFESTS, PT-D06 | Q-RW-19 decided before the train cut: the DP-30 exception supersedes OD-PDC-SCOPE-01 A for the UPGRADE part only. Audit condition 3: the exception text with a reopen condition goes into the freeze documents (T-AF2-MANIFESTS, PT-D06); the UPGRADE part stays raised until built, is neither PASS nor N/A, the 'N/A-with-evidence (Path B)' label stays invalid, and the non-composition gate of the FRZ-PLG-08 NT row is not evidence for UPGRADE |
| FRZ-PRD-03 | PT-FC2 | RW | PT-FC2-E, PT-FC3 | - |
| FRZ-PRD-05 | PT-WB | RW | PT-FC1, T-AF2-RW-GATES | - |
| FRZ-PRD-08 | T-AF2-RW-GATES | RW | PT-G02 | - |
| FRZ-REG-04 | PT-G02 | NON-RW: C2 NONPRODUCT (allowlist inventory and gate) | T-AF2-RED-CLASSES | - |
| FRZ-REG-09 | T-AF2-RW-GATES | RW | PT-G02 | - |
| FRZ-REL-07 | PT-FC2 | RW | PT-FC2-E, PT-FC3 | - |
| FRZ-REL-09 | PT-FC2 | RW | PT-FC2-E, PT-FC3 | - |
| FRZ-REL-10 | PT-FC2 | RW | PT-FC2-E, PT-FC3 | - |
| FRZ-RPL-04 | PT-WF | RW (C2 PRODUCT; W-F Parts JB/JC, dedupe D-11) | T-AF2-RW-TYP | - |
| FRZ-SEC-04 | PT-FC3 | RW | T-AF2-RW-RUNTIME, PT-WB, T-AF2-RW-TENANT | - |
| FRZ-SEC-06 | T-AF2-RW-SUPPLY | RW (C2 PRODUCT; plugin admission PDC-D6-R016) | - | - |
| FRZ-TEN-01 | PT-WF | RW | T-AF2-RW-TENANT | TEN-10 moved from T-IGX-051-A to W-F Part JA (dedupe D-04) |
| FRZ-TEN-10 | PT-WF | RW | T-AF2-RW-TENANT | TEN-10 moved from T-IGX-051-A to W-F Part JA (dedupe D-04) |
| FRZ-TYP-01 | PT-WF | RW (C2 PRODUCT; W-F Parts JB/JC, dedupe D-11) | T-AF2-RW-TYP | - |
| FRZ-TYP-02 | T-AF2-RW-TYP | RW (OD-AF-11: ratchet on the frozen contract surfaces) | T-AF2-MANIFESTS | not in the 35 or C2; owner recorded by OD-AF-11 |
| FRZ-TYP-03 | PT-WF | RW (C2 PRODUCT; W-F Parts JB/JC, dedupe D-11) | T-AF2-RW-TYP | - |
| FRZ-TYP-04 | T-AF2-RW-TYP | RW (OD-AF-11: ratchet on the frozen contract surfaces) | T-AF2-MANIFESTS | not in the 35 or C2; owner recorded by OD-AF-11 |
| FRZ-TYP-05 | T-AF2-RW-TYP | RW (OD-AF-11: ratchet + reduction target) | T-AF2-MANIFESTS | - |
| G-12 | T-AF2-RW-TENANT | RW | - | part of EX-L-09 |
| G-20 | T-AF2-RW-RECOVERY | RW | - | A-08: claim released every tick with no cap |
| KR-03 | T-AF2-RW-GATES | RW-CAUSE-FIRST (known-red FIX rows, RWD-03): step-0 cause measurement decides product side (fix in the train, cond scope) or test side (T-AF2-RED-CLASSES ruling, fix outside the train) | T-AF2-RED-CLASSES | KR-03 and KR-04 owner as named by the known-red register; KR-07 was UNASSIGNED |
| KR-04 | T-AF2-RW-GATES | RW-CAUSE-FIRST (known-red FIX rows, RWD-03): step-0 cause measurement decides product side (fix in the train, cond scope) or test side (T-AF2-RED-CLASSES ruling, fix outside the train) | T-AF2-RED-CLASSES | KR-03 and KR-04 owner as named by the known-red register; KR-07 was UNASSIGNED |
| KR-07 | T-AF2-RW-GATES | RW-CAUSE-FIRST (known-red FIX rows, RWD-03): step-0 cause measurement decides product side (fix in the train, cond scope) or test side (T-AF2-RED-CLASSES ruling, fix outside the train) | T-AF2-RED-CLASSES | KR-03 and KR-04 owner as named by the known-red register; KR-07 was UNASSIGNED |
| KR-17 | T-AF2-RW-GATES | RW (known-red FIX row, Q-RW-10) | - | pymongo imported outside the declared opens module |
| KR-18 | PT-WA | RW (known-red FIX row, Q-RW-10) | - | environment_profile/root.py imports applications._shared; v13: variant P2 (Operator, 2026-11-17; Q-WA-09 PACKET_0 default superseded, DR-2); residual EDER row: the contract root reaches the tools catalog through a lazy import in tools/contracts/tool_availability.py that the EBH-2A import extraction cannot see |
| KR-19 | PT-FC2 | RW (known-red FIX row, Q-RW-10) | - | import cycle observability_otlp -> export_boundary; node ids not enumerated in the register |
| PDC-D1-R009 | PT-D06 | NON-RW: docs | - | - |
| PDC-D1-R015 | T-AF2-RW-GATES | RW | - | - |
| PDC-D1-R017 | T-AF2-RW-SUPPLY | RW | - | - |
| PDC-D2-R004 | T-AF2-RW-TRUST | RW | - | - |
| PDC-D3-R005 | T-AF2-RW-RUNTIME | RW | - | - |
| PDC-D3-R018 | T-AF2-RW-SUPPLY | RW (reassigned from RW-RUNTIME: root cause F-D3-12 is in the A-12 set) | T-AF2-RW-RUNTIME | - |
| PDC-D3-R019 | T-AF2-RW-RUNTIME | RW | T-AF2-VBN-DOSSIER | - |
| PDC-D5-R002 | T-AF2-RW-GATES | RW | - | - |
| PDC-D6-R007 | T-AF2-RW-SUPPLY | RW | - | - |
| PDC-D6-R010 | T-AF2-RW-SUPPLY | RW | - | - |
| PDC-D6-R015 | T-AF2-RW-GATES | RW | - | - |
| PDC-D7-R015 | T-AF2-RW-RECOVERY | RW | - | - |
| PDC-D7-R022 | PT-D06 | NON-RW: docs | - | - |
| R-P8-01 | T-AF2-RW-TENANT | RW | - | closes only after B-1 X-A8-022 (NOT_REPRODUCED at B) and an independent audit |
| X-A4-004 | PT-WA | RW | - | same census as A4-004 |
| X-A4-016 | PT-WB | RW | - | - |
| X-A4-024 | PT-FC1 | RW | - | D-FC1-01; W-B records no code |
| X-A5-011 | PT-WF | RW | - | - |
| X-A6-003 | PT-WD | RW | - | same defect as A6-003 |
| X-A8-003 | PT-WB | RW | - | - |
| X-ESC-E1 | PT-WF | RW | - | - |
| X-ESC-E2 | PT-WF | RW | - | - |
| X-ESC-E4 | PT-WF | RW | - | - |
| X-ESC-E5 | PT-WF | RW | - | - |

Output of `python -I coverage_check.py` (verbatim):

```text
coverage_check.py at head_sha a7f02361540752426f0015b628f8a40b6cbdbb2a

EX (60): parsed 60 (expected 60); with owner 60; RW-owned 46; non-RW disposition 14; missing none; duplicated dispositions none
PARTIAL (13): parsed 13 (expected 13); with owner 13; RW-owned 11; non-RW disposition 2; missing none; duplicated dispositions none
PHASE3 (18): parsed 18 (expected 18); with owner 18; RW-owned 11; non-RW disposition 7; missing none; duplicated dispositions none
FRZ CODE_FIX (35): parsed 35 (expected 35); with owner 35; RW-owned 32; non-RW disposition 3; missing none; duplicated dispositions none
FRZ C2 (11): parsed 11 (expected 11); with owner 11; RW-owned 5; non-RW disposition 6; missing none; duplicated dispositions none
FRZ C2 rows: 11 (PRODUCT 5, NONPRODUCT 6); PRODUCT rows with a train owner and an NT row: 5
open findings without an owner (Operator items, not closed by any ticket): ['A4-019', 'A4-025']
negative-test rows: 169; items needing a row: 166; basis counts: {'B': 92, 'C': 19, 'G': 8, 'M': 16, 'N': 4, 'S': 30}
phase-3 rows with a red-first basis (B/S/G): 18 of 18; with >= 1 regression path existing at head_sha: 18 of 18
red-first rows (B/S/G): 130; prevention controls (M, reported apart, Q-RW-07 conditions attached): 16; conditional (C): 19; no code change (N): 4
audit-flag bindings (6 conflicts): 6 of 6 recorded with owner and auditing node
scope entries: 847; not verified at head_sha: 0
expected-red (registered, from expected_red_A01.json): 6 entries; planned, to be registered: 7 (all with owner and registration moment: True)
known-red FIX rows assigned (Q-RW-10): 6; with owner and negative-test row: 6
regression paths: 106; missing at head_sha: 0; tickets without a regression scope: 0
tickets: 22; lanes: 9; overlapping pairs: 23 (all serialized: True)
fact checks: 88; not verified: none

COVERAGE GAPS: 0
```

The check is mechanical: it proves exactly-once ownership, owner validity, row completeness, scope verification and lane rules over the data; it does not judge whether a negative test is the right one (that is T-AF2-RW-PLAN-AUDIT).

## (h) Open questions for the orchestrator

- **Q-RW-16** (Operator) Remaining Operator items after OPERATOR_PACKET_3: (b) the A4-040 scope ruling (Q4), which lands in W-B Part G or RW-SUPPLY on package_b_strict_production_serving_command.py; (d) authorization of the W-B Part G PROD-Q node runs. Items (a) and (c) are settled by OD-AF-22. Draft default: Part G excluded from the PT-WB packet until ruled.
- **Q-RW-17** (orchestrator) Text amendments the EQO-A answer assigns to the orchestrator: the W-D plan (option B binding, baseline JSON removed) and the MEP-AF2 section 0(e) wording. KR-19 has no enumerated node ids; the PT-FC2 implementer enumerates them at step 0 for the known-red register PR. Draft default: orchestrator text edit; no scope change.
- **Q-WB-09** (Operator) Edits to the two gate tests that pin the resolver files (tests/unit/architecture/test_ebh_2e_r6_r2_r3_routing_evaluator_end_to_end_injection.py, test_ebh_2e_r6_r1_r5_tier1_failover_chain_ordering_convergence.py) need the Operator's personal ruling before the PT-WB approval. Draft default (EQO-A): edit inside this ticket with that ruling (option A).
- **Q-WB-13** (Operator) A4-020 per-file bypass ruling over the 4 Part C candidates (PROVIDER_IMPORT_CLASSIFICATION.tsv). Draft default (EQO-A): the Operator rules per file after reading the TSV (option A); none ruled means no product edit, and the 4 files plus test_wb_provider_roots.py (NT-WB-31) stay cond.
- **Q-WB-14** (Operator) The gate-meaning part of the two RW-plan architecture tests in PT-WB (test_rw_bnd05_provider_switch_gate.py, test_rw_own05_duplicate_mechanisms.py). Draft default (EQO-A): listed firm, content as the RW plan rows say (NT-ROW-154 NT-BND-05-2, NT-ROW-035 NT-OWN-05-1), any gate-meaning part decided by the Operator before approval.
- **Q-FC1-01** (Operator) [PT-FC1] Part C1 (4 agent paths) firm (V2 CONFIRMED at B) or A1's conditional class; also decides NT-FC1-24 and NT-BND-05-1. Draft default: firm; counts 35 / 148 / 24 (A1 class kept: 31 / 144 / 28, conditional product exactly 21).
- **Q-FC1-02** (Operator) [PT-FC1] What is the 'sanctioned composition root' of NT-ROW-031 (NT-BND-05-1)? Draft default: the posture rule of plan section 5: no literal or lab-constant production_mode in a Tier-2 build_context.
- **Q-FC1-03** (orchestrator) [PT-FC1] Carrier of the host posture for C1 (RuntimeRequest field or injected host config; a contract edit is outside the write set). Draft default: fixed at step 0 from what build_context(request) already reads; stop and amend if none exists.
- **Q-FC1-04** (Operator) [PT-FC1] Existing STRICT tests that meet B4/C2 (13 NexusLoop(production_mode=True) sites in 2 files; an unmeasured part of 53 build_harness_host_runtime callers). Draft default: the 2 files inside the firm set; any other file is a write-set extension (new digest), found by the step-0 census.
- **Q-FC1-05** (orchestrator) [PT-FC1] Accept scenario_runtime_baseline.py as the 17th conditional product path. Draft default: accept (inside the ceiling of up to 21 conditional product paths, transcription item 2).
- **Q-FC1-06** (orchestrator) [PT-FC1] C2 refuses every STRICT loop without a context engine and an LLM adapter, also a STRICT host that plans without an LLM. Draft default: intended fail-closed behaviour, risk R-10.
- **Q-B-01** (orchestrator) [PT-FC2] tests/unit/docs/test_platform_configuration_reference.py firm (A1 v13) or cond (plan). Draft default: cond, applied in v14 (10 / 23 / 7); reversible to firm (10 / 24 / 6).
- **Q-B-02** (Operator) [PT-FC2] Opt-ins outside the named write set (concerns ACT-N11 paths). Draft default (v15, RWD14-01): only the 22 census test files enumerated in the PT-FC2 scope as cond (census-confirmed opt-in only, F-77); any file outside that list, and every applications/** path including applications/*/tests/, is a stop and an amendment.
- **Q-B-03** (orchestrator) [PT-FC2] KR-19 if the cycle closes only through intergrax/runtime/integrations/__init__.py or observability/__init__.py. Draft default: stop and amend; not in this write set.
- **Q-B-04** (orchestrator) [PT-FC2] KR-19 not reproduced at step 0. Draft default: NOT_REPRODUCED record, paths untouched, KR-19 re-disposed per Q-RW-08; counts drop to 8 / 22.
- **Q-B-05** (orchestrator) [PT-FC2] The FC-1 plan lacks the wiring_posture.py import sentence. Draft default: step-0 check of PT-FC2; PT-FC1's own packet fixes its plan.
- **Q-D-01** (Operator) [PT-FC4] Who may attest Redis durability (OPEN-ATT-1). Draft default: stays open under fallback (b): SKILLS.md states the attestation shape only, names no role.
- **Q-D-02** (orchestrator) [PT-FC4] Amend the plan (new digest, new delegation record) or keep the old digest and change only the ticket. Draft default: amend, as drafted (digest 2dfe0ad9...dd30).
- **Q-D-04** (orchestrator) [PT-FC4] EBH-7 F10 classification of the six product paths. Draft default: decided by the wave classifier probe, not by the ticket.
- **Q-E-1** (Operator) [PT-FC2-E] Does the packet approval of digest b641f93f... count as the 'own Operator decision' that delegation record 1 requires for a gate-meaning change? Draft default: yes; the packet names E as a gate-meaning item and quotes item 5 and addendum 4.
- **Q-E-2** (Operator) [PT-FC2-E] Keep both parts (capacity and sandbox) or rule the sandbox part out now (new digest; the sandbox gate stays red). Draft default: both parts (addendum 4: as drafted).
- **Q-E-3** (orchestrator) [PT-FC2-E] Add a section (d) negative-test row for the five script-level cases. Draft default: no; E-1/E-3 are the evidence; v14 adds no row.
- **Q-E-4** (Operator) [PT-FC2-E] Re-confirm that the amended content is the one the Operator saw (the drafted digest 00ee0873... was never in a packet). Draft default: no separate step; the packet lists the new digest.
- **Q-WD-11** (Operator) [PT-WD] Check design for AUD, CTL, DEBT, FRZ, HRN, TRC (entrypoints, exercised criteria). Draft default: one check per family on one existing suite chosen by the ENT-AUDIT-X owner.
- **Q-WD-12** (Operator) [PT-WD] Modes of the new checks. Draft default: FULL, FREEZE, CHANGED-SINCE; not FAST.
- **Q-WD-13** (Operator) [PT-WD] Acceptance wrappers and the semantic count. Draft default: generate six wrappers with the existing generator, coordinated with PT-E03.
- **Q-WD-NT** (orchestrator) [PT-WD] The plan has 41 tests; B03, B04, C04..C09 and E01..E06 (the only tests of A6-004) have no section (d) row; also the BASELINE:MANIFEST_DIGEST table wording (plan 9.3 rows 1-4) and a planned expected-red row for the two PDC wave-diff tests made latent red by PT-WA (5 unmapped agents/ modules). Draft default: no draft default for the rows; v14 keeps the (d) rows and the expected-red tables unchanged until ruled.
- **Q-E05-01** (Operator) [PT-E05] How a manifest is forced to be v2 (cut-over SHA or date). Draft default: cut-over SHA recorded in the protocol document.
- **Q-E05-02** (Operator) [PT-E05] Gate companion wiring plus adding the four PT-E04 files to MATERIAL_GOVERNANCE_PATHS, or a separate non-material CI ticket. Draft default: gate wiring with the material-set extension (an A1 amendment only if a CI job is chosen).
- **Q-E05-03** (Operator) [PT-E05] Delegation resolver: read-only copy in iiac_common.py or reuse of ticket_approval.resolve_delegation (adds a material path). Draft default: copy in iiac_common.py.
- **Q-E05-04** (Operator) [PT-E05] Strict SHA chain for the activation-type manifest of the RB-2 re-pin. Draft default: v2 final for the re-pin manifest.
- **Q-E05-05** (Operator) [PT-E05] Per-role signing keys or signed commits (WP-7 OQ-7 b). Draft default: not in this ticket; independence stays process_enforced.
- **Q-E05-06** (orchestrator) [PT-E05] The train lacks PT-E04/PT-E02 files (29 commits behind development): who merges development into the train, and when. Draft default: the train owner, before lane 2 starts.
- **Q-E05-07** (Operator) [PT-E05] May GOVERNANCE_WRITE_SURFACE include qualification evidence directories for the auditor commit check. Draft default: evidence of the audited stage between the two SHAs only.
- **Q-D08-01** (Operator) [PT-D08] Add intergrax/tools/providers/rag/USAGE.md (L63, also lists INTERGRAX_RAG_GRAPH_STORE) as a third path. Draft default: leave out, record a follow-up; counts stay 2 / 2 / 0.
- **Q-D08-02** (Operator) [PT-D08] Lifecycle column of the roster: contract lifecycle_state or PDC/IIAC status. Draft default: contract lifecycle_state, labelled as not a certification statement.
- **Q-D08-03** (Operator) [PT-D08] Name the two files in the RB-2 re-pin material list, or rely on PRODUCT_PATH_PREFIXES. Draft default: covered by PRODUCT_PATH_PREFIXES; the packet states it.
- **Q-TEN-02** (Operator) [T-AF2-RW-TENANT] EX-D5-05: required tenant_id on the base observation (option 1, needs the 4 GAP-04 cond paths, one of them SUPPLY's) or an optional validated field plus strict-host refusal (option 2); NT-ROW-068 says 'requires'. (A DS-EXC semantic decision; the orchestrator re-derives scope after the ruling.) Draft default: option 2; the CTRL-X pin is inverted in both.
- **Q-TEN-03** (orchestrator) [T-AF2-RW-TENANT] EX-L-09 G-12: refuse occupied-slot installs (inside A1), a token on the request (GAP-01) or a cross-process compare (GAP-02); NT-ROW-065/066 name a token the request does not have. Draft default: refuse; in-process lock; limits recorded.
- **Q-TEN-04** (Operator) [T-AF2-RW-TENANT] EX-D8-03 / EX-L-02: tenant prefix scheme or retire the raw storage.* tools; legacy keys unreachable. Draft default: prefix scheme; no migration.
- **Q-TEN-05** (orchestrator) [T-AF2-RW-TENANT] Audit or event port for the raw storage audit (GAP-08). Draft default: decided by step S0-3; a new event type is a gap and a stop.
- **Q-TEN-06** (orchestrator) [T-AF2-RW-TENANT] Audit and keys for the shadow-workspace file tools. Draft default: no.
- **Q-TEN-07** (Operator) [T-AF2-RW-TENANT] EX-D8-04: bind the principal as well as the run. Draft default: run only; principal as a conditional node.
- **Q-TEN-08** (Operator) [T-AF2-RW-TENANT] R-P8-01: keep D-4 and add an opt-in fails_closed guard, or invert D-4; NT-ROW-074's 'red at HEAD' holds only for the new guard. (Inverting D-4 reverses an Operator decision; the guard design is then the implementer's.) Draft default: keep D-4; add the guard; amend the callers (GAP-06).
- **Q-TEN-09** (orchestrator) [T-AF2-RW-TENANT] Platform-wide reindex hooks stay legal. Draft default: yes.
- **Q-TEN-10** (Operator) [T-AF2-RW-TENANT] EX-D2-06: STRICT forbids unscoped memory. Draft default: refuse composition (class C).
- **Q-TEN-11** (orchestrator) [T-AF2-RW-TENANT] Tools with a context that has no tenant. Draft default: refuse everywhere.
- **Q-TEN-12** (orchestrator) [T-AF2-RW-TENANT] Non-strict host without a tenant and a hosted sandbox. Draft default: keep two named lab constants; strict hosts refuse.
- **Q-TEN-REG** (orchestrator) [T-AF2-RW-TENANT] Regression additions of the draft (ctrl_x, pdc_x d8_repair/p8_repair/d2/d2_repair, four unit files, tests/unit/runtime/nexus) and NT-ROW-068 test_ten_04d, which may be green at the TENANT base because PT-WF merges first. Draft default: accept the additions (v14 does not change section e until ruled); test_ten_04d is not a red-first row if green.
- **Q-SUP-01** (orchestrator) [T-AF2-RW-SUPPLY] Amend A1 with the draft's gaps G-01..G-11, G-13 (12 product paths) and the pinned tests G-12, or narrow the findings; NT-ROW-088 (F-D6-09) cannot close without G-06 (intergrax/runtime/nexus/tools/plan_context_invocation.py, in no A1 block). Draft default: amend G-06 only; the rest after step 0 (v14 keeps the A1 write set: no ruling yet).
- **Q-SUP-02** (orchestrator) [T-AF2-RW-SUPPLY] The GCA strict path composes no admin service: accept a standalone revision-trust admission (a new product module). Draft default: decide after step S0-4; the pipeline fails closed meanwhile.
- **Q-SUP-03** (Operator) [T-AF2-RW-SUPPLY] EX-D6-01: build tamper evidence, or NOT_CLAIMED with reviewed wording. Draft default: build the hash chain; record the tail-truncation limit.
- **Q-SUP-04** (orchestrator) [T-AF2-RW-SUPPLY] The attested package digest of a plugin distribution is the installed-tree digest. Draft default: yes.
- **Q-SUP-05** (Operator) [T-AF2-RW-SUPPLY] F-D6-11: is a tenant-less resource open to a scoped tenant at task intake by intent. Draft default: no; explicit opt-in.
- **Q-SUP-06** (orchestrator) [T-AF2-RW-SUPPLY] Holder-scoped release of legal holds. Draft default: not in this ticket.
- **Q-SUP-07** (orchestrator) [T-AF2-RW-SUPPLY] What 'weak' means for the sandbox-strength gate. Draft default: the attributes found at step 0.
- **Q-SUP-08** (orchestrator) [T-AF2-RW-SUPPLY] Evasion rules in raw_secret_gate.py (not in A1, G-10) or in the new script. Draft default: in the new script.
- **Q-SUP-09** (Operator) [T-AF2-RW-SUPPLY] EX-D8-01: compose retention and the producer, or an explicit responsibility statement. Draft default: compose.
- **Q-SUP-10** (orchestrator) [T-AF2-RW-SUPPLY] A4-040 stays in PT-WB Part G; RW-SUPPLY adds nothing to it. Draft default: yes.
- **Q-SUP-REG** (orchestrator) [T-AF2-RW-SUPPLY] Regression additions (pdc_x d1_repair, d1, d6, d8, d8_repair, test_llm_quota.py, test_ac3_phase5_production_lifecycle_e2e.py); 'both launchers' read as ProductionLifecycleLauncher (PT-WF path) and ReferenceProductionLifecycleLauncher (in no A1 block), covered through deploy_activate_pipeline.py; NT-ROW-093 'no static gate exists' corrected to 'no script or CI step'. Draft default: accept the additions; A1 files kept; NT text read as the draft states.
- **Q-TR-02** (orchestrator) [T-AF2-RW-TRUST] Expectation-only edits of existing tests that pin the changed behaviour. Draft default (v15, RWD14-02): yes, no weakening; the 12 '[context:' modules (draft F-21, F-78) are enumerated in the RW-TRUST scope as cond; planner and engine tests are not enumerated, each one found at step 0 is a stop and an amendment.
- **Q-TR-03** (Operator) [T-AF2-RW-TRUST] Trust-class model (3 classes derived from the source enum) and the replacement role user (DS-EXC row). Draft default: as drafted; provider compatibility of consecutive user messages is a step-0 read.
- **Q-TR-04** (orchestrator) [T-AF2-RW-TRUST] STRICT call sites (context_manager.py, agent_engine.py, llm_router.py) as conditional paths, or contract and proof only (NT-ROW-100). Draft default: contract and proof only; EX-D2-07 not reported closed.
- **Q-TR-05** (Operator) [T-AF2-RW-TRUST] Accept that unclassified untrusted fragments are withheld in production flows until a classifier exists (NT-ROW-098 read literally withholds trusted system fragments too). Draft default: accept; no producer edit.
- **Q-TR-06** (Operator) [T-AF2-RW-TRUST] A product-side RAG root cause as a conditional path or a separate ticket; authorize the RAG extras-profile diagnostic run (C10/C15). Draft default: separate ticket; run only on the Operator's word.
- **Q-TR-07** (orchestrator) [T-AF2-RW-TRUST] Accept five extra read-only regression nodes. Draft default: yes.
- **Q-TR-08** (orchestrator) [T-AF2-RW-TRUST] Stored session-history system turns stay outside the ticket. Draft default: outside; residual recorded.
- **Q-RUN-01** (orchestrator) [T-AF2-RW-RUNTIME] Owner of PDC-D3-R018 (RW-PLAN: RW-SUPPLY, D-07; dispatch and plan of record: RW-RUNTIME). Draft default: contributed only (RW-PLAN kept).
- **Q-RUN-02** (Operator) [T-AF2-RW-RUNTIME] Adapters that cannot forward a timeout: fail closed or record 'not enforced' (EX-D3-01 reopen condition). Draft default: fail closed.
- **Q-RUN-03..09** (orchestrator) [T-AF2-RW-RUNTIME] SDK text in typed errors; mid-stream errors; pre-existing error classes by reference; served_model without A1 amendment; NT-ROW-106 may be green at base; 8 provider default rows; image-captioning seam. Draft default: as the draft states (plan section 19); no A1 amendment.
- **Q-RUN-10** (orchestrator) [T-AF2-RW-RUNTIME] Regression additions (PDC-D3 stream/refusal, provider cancellation, dependency isolation, harness_02, deadline authority, roadmap-replay ledger). Draft default: add (v14 does not change section e until ruled).
- **Q-RUN-11** (orchestrator) [T-AF2-RW-RUNTIME] A1 glob intergrax/runtime/llm/* is firm but no edit is expected. Draft default: keep as ceiling.
- **Q-RC-01** (orchestrator) [T-AF2-RW-RECOVERY] Add the schedule-store files as conditional paths for PDC-D7-R015 (postgresql_task_checkpoint_store.py is a PT-WB cond path: a new overlap pair), or keep A1 and accept the residual. Draft default: keep A1.
- **Q-RC-02** (orchestrator) [T-AF2-RW-RECOVERY] EX-D1-07: descriptor edit (registry_projection_descriptor.py, outside A1) as a conditional path. Draft default: stop and report, no weakening.
- **Q-RC-03..05** (orchestrator) [T-AF2-RW-RECOVERY] Controller wiring into the document-store worker compositions; the 3 enqueue sites in no train ticket (narrow NT-ROW-116 to the 2 A1 sites); re-point XD-05. Draft default: follow-up; narrow NT-ROW-116; out of ticket.
- **Q-RC-06..10** (orchestrator) [T-AF2-RW-RECOVERY] flush_services seam; extra regression nodes; reconciler policy; capped release; NT-ROW-115 OverflowError leg is a control (green at HEAD), the cap is the red-first leg. Draft default: as drafted.
- **Q-GAT-GM** (Operator) [T-AF2-RW-GATES] Gate-meaning parts before approval: GM-1 promotion-gate binding of EX-D5-03 (mandatory), GM-2 the 13 NOT_COMPOSED capabilities, GM-3 census and inventory pass/fail rules, GM-4 extending the banned vendor-SDK lists (needs A1 request R-3: the two check scripts are not in A1, NT-ROW-057), GM-6 the typed field the generic host profile loses (DS-EXC EX-D4-05); GM-5 is Q-GAT-GM5. Draft default: as in plan section 11; A1 kept until R-1..R-7 are ruled.
- **Q-GAT-GM5** (orchestrator) [T-AF2-RW-GATES] GM-5: which event schema wins on the finalized-trace path. Draft default: decided after step 0.
- **Q-GAT-01** (orchestrator) [T-AF2-RW-GATES] Canonical get_steps signature (plan of record names get_steps(context); protocol and executor use get_steps()). Draft default: the runtime protocol's signature; the test derives it from UAEPAgent.
- **Q-GAT-02** (Operator) [T-AF2-RW-GATES] Promotion-gate binding of EX-D5-03 (ticket class, subject field, verifier; = GM-1). Draft default: closed, empty stage-to-ticket table.
- **Q-GAT-02b** (orchestrator) [T-AF2-RW-GATES] Local re-implementation of the strict approval grammar, because scripts/governance/ticket_approval.py is not on the train. Draft default: local verifier.
- **Q-GAT-03** (orchestrator) [T-AF2-RW-GATES] EX-L-13 cause outside A1. Draft default: stop and amend.
- **Q-GAT-04** (orchestrator) [T-AF2-RW-GATES] Digest enforcement scope. Draft default: admission API plus test; residual recorded.
- **Q-GAT-05** (orchestrator) [T-AF2-RW-GATES] Defect behind a conditional C row without an A1 product path. Draft default: stop and request an amendment.
- **Q-GAT-06** (orchestrator) [T-AF2-RW-GATES] 'Composing a NOT_COMPOSED feature raises' read as a gate failure (none of the 13 capability modules is in A1). Draft default: gate failure.
- **Q-GAT-07** (orchestrator) [T-AF2-RW-GATES] Operator procedure document for EX-D1-03. Draft default: out of this ticket.
- **Q-GAT-08** (orchestrator) [T-AF2-RW-GATES] Test-side files of KR-03/04 (NT-ROW-164/165 send a test-side cause to T-AF2-RED-CLASSES; A1 lists them cond). Draft default: cond, edited only on an in-train ruling.
- **Q-GAT-09** (Operator) [T-AF2-RW-GATES] Ratification of the existing duplicate contracts. Draft default: not in this ticket.
- **Q-GAT-REQ** (orchestrator) [T-AF2-RW-GATES] A1 requests R-1 (tests/qualification/pdc_x/d5/test_xd08_eval_pass_cannot_authorize_promotion.py), R-2 (pdc_x/d4/test_d4_no_product_vocabulary.py and TIER3_APPLICATION_ENVIRONMENT.md), R-3 (check_agents_no_vendor_sdk_imports.py, check_integration_vendor_imports.py; GM-4), R-4..R-7. Draft default: recorded, not applied; A1 kept until ruled.
- **Q-TYP-01..06** (orchestrator) [T-AF2-RW-TYP] Per-file plus total ratchet (stricter than the NT-ROW-167/168 text); a count below the baseline fails STALE; typed-key definition for TYP-05; read-only regression additions; no CI script registration; no extra reflection metrics. Draft default: as drafted; re-measurement at the train head 05699b633 equals the HEAD baseline (138 / 357, 1 / 3 / 0 / 0 / 0, 5 tables).
- **Q-DEP-01..03** (orchestrator) [T-AF2-DEPS-BATCH] The four bot PRs (#631, #637, #638, #639) are closed unmerged: carry none? range-only versus version upgrade; owner class and whether a two-line map change is gate-meaning. Draft default: carry none; range-only; CORE_FOUNDATION, not gate-meaning.
- **Q-DEP-04** (Operator) [T-AF2-DEPS-BATCH] Declare jsonschema and tzdata (a) or move them out of core (b, an install-contract change). Draft default: (a).
- **Q-DEP-05..06** (orchestrator) [T-AF2-DEPS-BATCH] Docs edit and CI wiring of the ownership script; correct the known-red register node names (test_ebh_3_dependency_ownership_gate.py does not read pyproject.toml; name test_dependency_governance.py::test_current_project_passes_dependency_governance). Draft default: neither here; register PR re-names and re-measures.
- **Q-A7C-01** (orchestrator) [T-AF2-A4007-CALLERS] Which registered source supplies the permitted Tier-2 agent set (product manifest registry, fleet inventory build artefact, the agents/ tree, or revision package digests). A choice that widens authority is an amendment for the Operator. Draft default: the product manifest registry.
- **Q-A7C-02** (orchestrator) [T-AF2-A4007-CALLERS] Add package_wiring.py (the only env-holding caller of the compat roster resolver, reachable in PRODUCT-BALANCED hosts) as a conditional path. Draft default: yes, conditional (in v16 as cond); otherwise C-7 stays a recorded residual.
- **Q-A7C-03** (orchestrator) [T-AF2-A4007-CALLERS] The gate scripts hold no env: guarded_import_policy() (fail_closed=True, sets from the registry) or a posture computed through wiring_fails_closed. Draft default: guarded_import_policy().
- **Q-A7C-04** (orchestrator) [T-AF2-A4007-CALLERS] environment_diff_wiring.py:108 cannot import (C-6): edit for uniformity or leave with the control test. Draft default: leave, with CTL-A7C-03 (path cond in v16).
- **Q-A7C-05** (orchestrator) [T-AF2-A4007-CALLERS] b2 permitted set = <product_id>_application of registered products (unregistered application packages with migrations would fail the gate). A registry package field would be a contract change and an amendment for the Operator. Draft default: the naming convention, guarded by UT-A7C-05.
- **Q-A7C-06** (Operator) [T-AF2-A4007-CALLERS] Are the two script edits a gate-meaning change (personal route, OD-AF-22) or an input hardening of an unchanged check. Draft default: not gate-meaning (same checks, messages and exit codes; identical result at the train head); if ruled gate-meaning, Part C moves to the wave packet and Parts A and B stay.
- **Q-A7C-07** (orchestrator) [T-AF2-A4007-CALLERS] Register text in T-IGX-050-M for the A4-007 b1/b2 ownership move (the RW-PLAN part, A1 block, lane entry and ownership, is done in v16). Draft default: as in FOLLOW_UPS.md 2026-11-17.
- **Q-A7C-08** (orchestrator) [T-AF2-A4007-CALLERS] Residuals RES-1 (load_agent_class, wiring.py:604-608, no product caller) and RES-2 (reference-roster compat call): give them a node or accept them. Draft default: accepted as recorded residuals.
- **Q-A4-019** (Operator) [open item] OPERATOR-ONLY (not covered by the standing rules OD-AF-23/S3; the Operator's own decision). A4-019 (embedding default, vector-store fallback, guardrail stub) is not closed by T-IGX-043-F (Q-F-2 ruling); who owns its closure condition, or is the closure condition restated so that PT-FC3's negative tests close it? Draft default: goes into the lane 4 to 7 packet as an explicit Operator item; no owner until ruled; the T-IGX-043-F PR body states that it does not close A4-019.
- **Q-A4-025** (Operator) [open item] OPERATOR-ONLY (not covered by the standing rules OD-AF-23/S3; the Operator's own decision). A4-025 (ambient environment reads outside the inventoried CONFIG-X surfaces) is recorded, not failing, under ruling 3a; keep it open or move it to a gate ticket. Draft default: stays open; a failing rule would be a new gate ticket.
- **Q-C-01** (Operator) [PT-FC3] Edits to census-found test files (fixtures only) outside the named write set; concerns ACT-N11 if an applications/** path is touched. Draft default: only the 76 census files enumerated in the PT-FC3 scope as cond (F-80; RWD14-01 rule); any other file, and every applications/** path, is a stop and an amendment.
- **Q-C-02** (orchestrator) [PT-FC3] Who lands the old lint findings of the 50 write-set files. Draft default: one separate mechanical commit inside the ticket (step 1a), no path outside the write set.
- **Q-C-03** (Operator) [PT-FC3] legal/dispute_sim/governed_contractor/research profiles carry no notification_channel and refuse at composition in PRODUCT/STRICT after FC-3; who supplies the binding. Draft default: the host environment or profile owner (L-1), not this ticket.
- **Q-C-04** (orchestrator) [PT-FC3] The dep1 node is shared with FC-2's KR-19. Draft default: FC-3 keeps node id and assertions and passes the opt-in; T-IGX-043-B owns the KR-19 disposition.
- **Q-C-05** (orchestrator) [PT-FC3] security_configuration.py has no row in the IIAC impact matrix. Draft default: not edited here; the RB-2 session decides.
- **Q-F-1** (Operator) [PT-FC3-F] Does the packet approval of 903e830d... count although revision 4 narrows the design after the F-A2 measurement. Draft default: yes (accepted, standing rule S3).
- **Q-F-4** (orchestrator) [PT-FC3-F] Keep the SCAN_BLIND_PATHS coverage table of NT-FC3-CX7. Draft default: keep (accepted, S3).
- **Q-F-5** (orchestrator) [PT-FC3-F] Where hits outside the eight paths are recorded. Draft default: the list printed by the new test plus one EDER row per real hit after the audit (accepted, S3).
- **Q-WF-04** (Operator) [PT-WF] OPERATOR-ONLY (not covered by the standing rules OD-AF-23/S3; the Operator's own decision). Inversion of the two EBH-6 Q2-D2 gate files (tests/qualification/ebh_6/_q2d2_technical_qualification.py, test_ebh_6_q2d2_dual_path.py) in Part JB is a gate-meaning change. It blocks the merge of Part JB (8 product paths); Parts JA, JC and JG may proceed. Draft default: the Operator rules personally; the inversion lands in the same wave as JB.
- **Q-WF-03..18** (orchestrator) [PT-WF] Q2-D2 seam type (03), STRICT host tenant authority (05), LAB default tenant (06), observation tenant out of W-F (07), DP-33 evidence (08), held debt parts (09), merge order (12), census stop above 15 files (14), JG-5b stays cond (15), JG-1 always on (16), S-1 line-137 pin (17), launcher protocol breadth (18). Q-WF-03 and Q-WF-05 keep their defaults (orchestrator). Draft default: as in the draft table (INPUT_043-J.md).
- **Q-WF-14b** (orchestrator) [PT-WF] The census of existing STRICT/PRODUCT tests without a tenant is a step-0 output and cannot be enumerated now (same family as RWD14-01). Draft default: the step-0 list goes into an amendment before any of those files is edited; above 15 files is a stop.
- **Q-T01-01** (Operator) [PT-T01] OPERATOR-ONLY (not covered by the standing rules OD-AF-23/S3; the Operator's own decision). Open items without a fixer in the train (v18 ruling): NT-07 (A2-049: the P6 restart/resume classifier treats a new file under a covered prefix as canonical) and NT-04b (TENANT-X silent-substitution detector scope of A4-027). Both are held out of the merge set; their code under test is test support that A1 does not contain: tests/qualification/trace_x/_trace_x_p6_restart_resume_classification.py and tests/qualification/tenant_x/_r1_silent_substitution.py. Owner needed: a fixer ticket (tests-only paths, a new A1 block) or an Operator-accepted residual. Draft default: HELD (written, red-first recorded at A, not merged); PT-T01 counts stay 0 / 5 / 0.
- **Q-T01-02..10** (orchestrator) [PT-T01] A1-only test location (02), probes V4/V6/V8 out of the ticket (03), test marks (04), no emitter-gate widening (05), G-09b kept only if buildable (06), NT-08 asserts the rule (07), section d rows and edges PT-FC1/PT-FC3/A4007 -> PT-T01 (08), red-first base a7f023615 and B for the auditor (09), separate lint commit (10). Draft default: as in the draft table (INPUT_043-L.md); the L1 red-gate list keeps check_runtime_event_tenant_propagation.py until PT-T01 merges.
- **Q-WE-DORMANT** (Operator) [PT-WE, withdrawn] OPERATOR-ONLY (not covered by the standing rules OD-AF-23/S3; the Operator's own decision). withdrawn from the RW train (orchestrator ruling, FOLLOW_UPS.md 2026-11-17 ORCHESTRATOR RULINGS item 1): the A1 condition 'only if the audit reproduces one' cannot be met inside RW, because RA-1 (PT-Q08) runs after the wave; a reproduction after the wave goes to RB-3 or a second wave by Operator decision (plan section O, item 15). The draft (w3-packet-4/043-K, digest 57c9dc5f...581a) stays DORMANT, not merged; no section d row; the Operator is told in the lane 4 to 7 packet. Draft default: DORMANT; a post-wave reproduction is routed to RB-3 or a second wave by Operator decision.

## Limits of this record

- No test or probe was run: none was strictly needed, because every red-first condition is stated for the implementer and checked by the audit; no lock was taken.
- Write sets of the EQO-A tickets are extracted mechanically from their plans; the extraction excludes the plan line ranges listed in `rw_plan_data.py` (`FC_EXTRACT_EXCLUDE`, `FC_EXTRACT_DROP`). A path a plan names only in prose outside its write-set section is not in scope.
- Conditional parts (W-B Parts E-conditional and G, W-F JG-5b, RW-SUPPLY security_runtime_bridge and core/security_bootstrap, RW-GATES KR-03/04/07 product files, the three conditional W-A tests) are in the overlap matrix; they are binding only if their condition is decided.
- B-1/B-2 results are diagnostic, non-certifying runs at B; they are cited as inputs, not as evidence of a fix or of red-first.

## Appendix A0. Answers applied in v2

| Q | Resolution |
| --- | --- |
| Q-RW-01 | PT-WE = T-IGX-043-K, PT-T01 = T-IGX-043-L, PT-D08 = T-IGX-046-D, PT-E05 = T-IGX-044-F (EQO-A answer) |
| Q-RW-02 | A4-007 -> PT-WB Part E; W-A without Part F (both answers) |
| Q-RW-04 | PT-WD option B binding; ENTERPRISE_AUDIT_BASELINE.json out of the write set; BASELINE:MANIFEST_DIGEST expected-red with owner PT-WD; PT-WD Approved line is the Operator's (gate-meaning) |
| Q-RW-05 | D-04, D-05, D-07, D-11, D-12 accepted |
| Q-RW-06 | W-F Part JD stays outside the train |
| Q-RW-07 | audit ruling ACCEPT WITH CONDITIONS (T-AF2-RW-PLAN-AUDIT section 8); conditions 1-6 attached to every basis-M row (m_conditions in the JSON, table in section d) |
| Q-RW-08 | re-disposition rule added to section d |
| Q-RW-09 | DEPS-BATCH re-listed at the train cut |
| Q-RW-10 | KR-18 -> PT-WA, KR-19 -> PT-FC2, KR-17 -> T-AF2-RW-GATES, each with a negative-test row |
| Q-RW-11 | orchestrator adds an erratum note to the B-1 results record; PREREG and probe docstrings are authoritative |
| Q-RW-12 | orchestrator path-correction docs PR (FOLLOW_UPS) |
| Q-RW-13 | confirmed: PT-T01 in the last lane, no expected-red marking |
| Q-RW-15 | W-B Part G stays inside W-B, serialized with RW-SUPPLY; the overlap on package_b_strict_production_serving_command.py is in the matrix; A4-040 lands in that file |
| Q-RW-18 | v4 (RWD-03): KR-03, KR-04, KR-07 owned by T-AF2-RW-GATES with a basis-C step-0 cause measurement and cond product scope; test-side cause -> T-AF2-RED-CLASSES ruling; to be confirmed in the orchestrator known-red register PR before the SCOPE gate |
| Q-RW-03 | v6: Operator chose option A (OPERATOR_PACKET_3, Q-WA-05): W-A Part E is firm; the fallback is dropped |
| Q-RW-14 | v6: OD-AF-11 decided (OPERATOR_PACKET_3): RW-TYP = ratchet on the frozen contract surfaces + TYP-05 reduction target (4 -> 0 dispatch candidates in 4 named files); no conditional intergrax/contracts/** scope |
| Q-RW-19 | v6: DP-30 exception decided (OPERATOR_PACKET_3) before the train cut; audit condition 3 recorded on FRZ-PLG-08 |
| Q-RW-16 (a), (c) | v6: OD-AF-22 approval route (wave packet + signing delegate with Delegation pointer) covers PT-WD, PT-E05 and PT-D08 |
| Q-RW-20 | v7: PT-WA re-extracted from the final W-A plan (digest 0c0cd523...e714, INPUT_043-G.md); overlap matrix and lanes re-run |
| Q-RW-21 | v9: PT-WB (digest e37ee1ec...b2b6) and PT-WD (digest 8f142c58...d4a5) re-extracted from their final plans; the other eight EQO-A plans have the same path sets and were not re-extracted; overlap matrix and lanes re-run |
| Q-RW-22 | v13: PT-WA re-extracted from the amended W-A plan (Amendment 1 v2, digest 5b7d2827...c8d1, REVIEW_PASS); the Operator approved that digest and variant P2 for KR-18; Q-WA-09: P2 supersedes the PACKET_0 default (DR-2); overlap matrix, lanes and coverage re-run |
| Q-TR-01 | v14 (orchestrator ruling): intergrax/context/formatter.py added to T-AF2-RW-TRUST as firm; the use of provider_lifecycle.py (trust-class provenance recording only) is as the draft states |
| Q-TEN-01 | v14 (orchestrator ruling): T-AF2-RW-TENANT A1 amended with the paths of draft plan sections 4.1 and 4.2 (10 firm, 42 cond; see the ticket note) |
| Q-RW-23 | v14: the seven EQO-A lane 2/3 drafts (043-A, B, D, E, I, 044-F, 046-D) re-extracted from w3-packet-2; NT-FC1 file column follows the 043-A plan; MEP drafts reconciled (write set kept except Q-TR-01, Q-TEN-01) |
| Q-RW-24 | v16 (orchestrator ruling option b): the A4-007 b1/b2 callers are a separate train ticket T-IGX-051-I (T-AF2-A4007-CALLERS); the 043-B digest 14e4de95... stays unchanged; A4-007 is owned by the new node with PT-WB as contributor |
| Q-F-2 | v18 (orchestrator ruling): T-IGX-043-F does not close A4-019; A4-019 is an open Operator item without an owner (OPEN-OPERATOR) |
| Q-F-3 | v18 (orchestrator ruling): RW-PLAN follows the Operator ruling option 3a: NT-ROW-049 restated (only hits on the eight FC-3 paths fail; others recorded), NT-CFG-08-1 withdrawn, A4-025 open (OPEN-OPERATOR), its row basis N |
| PT-WE | v18 (orchestrator ruling): withdrawn from the train, DORMANT; removed from lanes, overlap matrix and packet scope; no section d row |
| PT-T01 held set | v18 (orchestrator ruling): NT-07 and NT-04b held out of the merge set; open Operator item Q-T01-01; 'green when merged' applies to NTs with a train fixer |
| Q-WF-01 | v18 (orchestrator, accepted default): PT-WF stays one ticket as in A1; Parts JA, JB, JC, JG separately cuttable |

## Appendix A1. Binding scope per ticket

Version v18; facts at `a7f02361540752426f0015b628f8a40b6cbdbb2a`. Each EQO-A ticket's `Source` cites the plan file and line it was extracted from. PT-WA was re-extracted in v7 (digest `0c0cd523...e714`) and again in v13 from the amended plan (digest `5b7d2827185ae3f06562d5e8d76a4cdec067f26b5283929c0e1ebcc4333ac8d1`); PT-FC1, PT-FC2, PT-FC4, PT-FC2-E, PT-WD, PT-E05 and PT-D08 were re-extracted in v14 from the w3-packet-2 drafts (digests in the v14 changelog); PT-WD (digest `8f142c58...d4a5`) was re-extracted in v9 and PT-WB (digest `f5f0d630...ccc9`) in v12; the other EQO-A tickets keep the 2026-11-16 extraction, whose path sets equal their current plans.

### PT-WA (T-IGX-043-G)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/contracts/application_graph_topology.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:185 |
| `intergrax/contracts/delegation_budget_policy.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:186 |
| `intergrax/contracts/host_orchestration_wiring_port.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:187 |
| `intergrax/contracts/host_orchestration_environment.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:188 |
| `intergrax/contracts/rag_retrieval_source_scope.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:189 |
| `intergrax/contracts/workspace_tool_ids.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:190 |
| `intergrax/applications/_shared/application_host_orchestration_wiring.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:191 |
| `intergrax/applications/_shared/host_orchestration_environment_resolution.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:192 |
| `intergrax/tools/contracts/tool_availability.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:193 |
| `intergrax/runtime/execution/application_environment_context_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:198 |
| `intergrax/runtime/execution/application_graph_spec_to_plan.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:199 |
| `intergrax/runtime/execution/application_host_orchestration_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:200 |
| `intergrax/runtime/execution/environment_host_task_execution.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:201 |
| `intergrax/runtime/execution/harness_host_internal_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:202 |
| `intergrax/runtime/execution/host_orchestration_application_wiring_applier.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:203 |
| `intergrax/runtime/execution/host_orchestration_assembly_validation.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:204 |
| `intergrax/runtime/execution/host_orchestration_environment_spec_builder.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:205 |
| `intergrax/runtime/execution/host_orchestration_planner_classifier_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:206 |
| `intergrax/runtime/execution/host_orchestration_loop_init_spec.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:207 |
| `intergrax/applications/_shared/host_orchestration_backend_spec_builder.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:208 |
| `intergrax/applications/_shared/harness_host_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:209 |
| `intergrax/applications/_shared/harness_host_runtime.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:210 |
| `intergrax/applications/_shared/scenario_runtime_baseline.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:211 |
| `intergrax/applications/_shared/orchestration_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:212 |
| `intergrax/applications/_shared/context_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:213 |
| `intergrax/applications/_shared/delegation_budget_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:214 |
| `intergrax/applications/_shared/observability_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:215 |
| `intergrax/applications/contracts/graph_spec.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:216 |
| `intergrax/tools/providers/rag/source_scope_transport.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:217 |
| `intergrax/tools/providers/workspace/tool_ids.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:218 |
| `agents/local_search/steps/search_job.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:219 |
| `agents/tool_selection_qualifier/tool_selection.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:220 |
| `agents/tool_selection_qualifier/steps/tool_selection_job.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:221 |
| `agents/web_search_qualifier/steps/web_search_job.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:222 |
| `agents/web_search_qualifier/web_search_qualifier_agent.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:223 |
| `applications/local_workspace_application/host/agent_factories.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:224 |
| `intergrax/applications/contracts/environment_profile/root.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:225 |
| `applications/local_workspace_application/manifest.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:226 |
| `intergrax/skills/registry/tool_requirements.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:227 |
| `applications/local_workspace_application/host/environment_profile.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:232 |
| `intergrax/runtime/execution/host_orchestration_wiring_bundle.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:238 |
| `intergrax/runtime/execution/scenario_host_diagnostic_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:239 |
| `intergrax/runtime/architecture/obs_diag_x3_ast_gates.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:240 |
| `agents/web_search_qualifier/search_provider_resolver.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:241 |
| `applications/local_workspace_application/workspaces/document_store_factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:246 |
| `applications/local_workspace_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:247 |
| `testing_support/architecture/obs_diag_x3_ast_gates.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:253 |
| `tests/system/functional_diagnostics_q3/search_provider_resolver.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:254 |
| `tests/unit/runtime/architecture/test_wa_boundary_inversion_gate.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:255 |
| `tests/unit/applications/test_wa_wiring_port_inversion.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:256 |
| `tests/unit/agents/web_search_qualifier/test_wa_search_provider_injection.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:257 |
| `tests/unit/architecture/test_wa_contract_relocation_parity.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:258 |
| `tests/unit/agents/test_wa_agent_import_rule.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:259 |
| `tests/unit/architecture/test_wa_integrations_shared_leakage.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:260 |
| `tests/unit/architecture/test_wa_kr18_contract_root_no_shared_import.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:261 |
| `tests/unit/applications/test_wa_lkw_roster_search_provider.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:262 |
| `tests/qualification/harness_01/nexus_import_inventory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:267 |
| `tests/unit/architecture/test_ebh_4_r1_r3_b6_typing_gate.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:268 |
| `tests/unit/applications/test_orchestration_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:269 |
| `tests/unit/applications/test_harness_host_runtime_llm.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:270 |
| `testing_support/application_environment_test_support.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:271 |
| `tests/fixtures/harness_host_task_execution.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:272 |
| `tests/unit/applications/_shared/test_obs_diag_x3_universal_spine_adoption.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:273 |
| `tests/unit/applications/architecture/test_diag_platform_adoption_gate.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:274 |
| `tests/system/functional_diagnostics_q3/runner.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:275 |
| `tests/unit/applications/test_effective_profile_revision_adoption.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:277 |
| `tests/unit/architecture/test_ebh_2e_r4_agent_llm_composition_injection.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:277 |
| `tests/unit/architecture/test_ebh_2e_r4_r1_structural_replaceability_and_metadata_resolution.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:277 |

### PT-WB (T-IGX-043-H)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/contracts/profile_projection.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:149 |
| `intergrax/applications/_shared/import_allowlist.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:149 |
| `intergrax/runtime/wiring/llm_resolver.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:152 |
| `intergrax/applications/_shared/llm_resolver.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:153 |
| `intergrax/runtime/nexus/engine/runtime_state.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:154 |
| `intergrax/contracts/host_profile_slices.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:156 |
| `intergrax/applications/_shared/runtime_boundary_adapters.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:157 |
| `intergrax/applications/_shared/reliability_runtime_bridge.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:158 |
| `intergrax/applications/_shared/wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:160 |
| `intergrax/applications/_shared/agent_resolution.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:161 |
| `intergrax/applications/_shared/migration_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:162 |
| `intergrax/applications/_shared/mcp_import_guard.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:163 |
| `intergrax/agents/readiness/scoreboard.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:164 |
| `applications/local_workspace_application/host/main.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:165 |
| `applications/research_application/host/main.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:166 |
| `applications/research_application/host/settings.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:167 |
| `applications/research_application/.env.example` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:168 |
| `intergrax/runtime/execution/postgresql_decision_checkpoint_persistence.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:184 |
| `intergrax/runtime/long_running/postgresql_task_checkpoint_store.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:185 |
| `intergrax/runtime/migration/human_decision_legacy_disposition.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:186 |
| `applications/local_workspace_application/slack_companion/companion.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:187 |
| `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:193 |
| `applications/governed_contractor_application/hosting/__init__.py` | new | cond | NEW_IN_NEW_DIR | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:194 |
| `applications/governed_contractor_application/hosting/profile.py` | new | cond | NEW_IN_NEW_DIR | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:195 |
| `applications/governed_contractor_application/hosting/runtime.py` | new | cond | NEW_IN_NEW_DIR | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:196 |
| `applications/governed_contractor_application/hosting/boundary.py` | new | cond | NEW_IN_NEW_DIR | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:197 |
| `applications/governed_contractor_application/deploy/docker-compose-local-qualification/docker-compose.yml` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:198 |
| `applications/governed_contractor_application/deploy/docker-compose-local-qualification/RUNBOOK.md` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:199 |
| `applications/governed_contractor_application/deploy/linux/single-node-production-profile-v1/intergrax-gca-strict-production.env.example` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:200 |
| `intergrax/llm_adapters/USAGE.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:206 |
| `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:214 |
| `tests/unit/applications/test_wb_llm_resolver_canonical.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:220 |
| `tests/unit/applications/test_wb_llm_resolver_delegation.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:221 |
| `tests/unit/contracts/test_wb_profile_projection.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:222 |
| `tests/unit/applications/test_wb_profile_projection_registry.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:223 |
| `tests/unit/applications/test_wb_serving_bind_defaults.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:224 |
| `tests/unit/applications/test_wb_import_allowlist.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:225 |
| `tests/unit/applications/test_wb_provider_roots.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:231 |
| `tests/acceptance/llm_routing/test_concurrent_run_routing_isolation.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:237 |
| `tests/unit/llm_adapters/routing/test_evaluating_adapter.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:238 |
| `tests/unit/llm_adapters/routing/test_routing_metering.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:239 |
| `tests/unit/llm_adapters/routing/test_secondary_evaluating_wrap.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:240 |
| `tests/unit/architecture/test_ebh_2e_r6_r2_r3_routing_evaluator_end_to_end_injection.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:241 |
| `tests/unit/architecture/test_ebh_2e_r6_r1_r5_tier1_failover_chain_ordering_convergence.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:242 |
| `tests/unit/architecture/test_rw_bnd05_provider_switch_gate.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:248 |
| `tests/unit/architecture/test_rw_own05_duplicate_mechanisms.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:249 |
| `tests/unit/applications/test_wb_gca_hosted_strict_serving.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:255 |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_package_b_strict_serving_wiring_gate.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:256 |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_p1_strict_production_reachability.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:257 |
| `tests/qualification/prod_q/_prod_q_r1_r7_q_p1_strict_reachability.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:258 |

### PT-FC2 (T-IGX-043-B)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/runtime/capacity/production_adapters.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:110 |
| `intergrax/applications/_shared/production_capacity_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:111 |
| `intergrax/applications/_shared/production_queue_resolver.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:112 |
| `intergrax/applications/_shared/queue_worker_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:113 |
| `intergrax/applications/_shared/memory_provider_admission.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:114 |
| `intergrax/applications/_shared/memory_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:115 |
| `intergrax/applications/_shared/runtime_event_delivery_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:116 |
| `intergrax/applications/_shared/sandbox_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:117 |
| `intergrax/applications/_shared/wiring_posture.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:118 |
| `intergrax/runtime/integrations/observability_otlp.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:119 |
| `intergrax/runtime/observability/export_boundary.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:122 |
| `tests/unit/runtime/observability/test_fc2_kr19_import_order_independent.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:126 |
| `tests/unit/runtime/capacity/test_fc2_capacity_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:126 |
| `tests/unit/applications/test_fc2_queue_worker_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:126 |
| `tests/unit/applications/test_fc2_memory_org_session_admission.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:126 |
| `tests/unit/applications/test_fc2_otlp_composition_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:126 |
| `tests/unit/applications/test_fc2_sandbox_composition_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:126 |
| `tests/unit/applications/architecture/test_fc2_capacity_ci_posture_not_used_by_product_code.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:126 |
| `tests/unit/runtime/capacity/test_ecp_depth_gate.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:128 |
| `tests/unit/runtime/capacity/test_ecp_control_plane_governance.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:128 |
| `tests/unit/applications/architecture/test_diag_platform_adoption_gate.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:128 |
| `tests/unit/applications/test_environment_llm_wiring.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:128 |
| `tests/unit/applications/test_memory_plugin_discovery_unified.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:128 |
| `tests/unit/applications/test_package_wiring.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:128 |
| `tests/unit/applications/test_specialized_memory_composition.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:128 |
| `tests/unit/docs/test_platform_configuration_reference.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:128 |
| `docs/project/architecture/ELASTIC_CAPACITY_AND_SCALING.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:130 |
| `docs/project/architecture/BACKGROUND_TASKS.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:130 |
| `docs/project/architecture/MEMORY_ARCHITECTURE.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:130 |
| `docs/project/technical/guides/PLATFORM_CONFIGURATION.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-B/PLAN_T-IGX-043-B.md:130 |
| `tests/integration/memory/e2e/mongo_user_profile_real_vendor_support.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/memory/e2e/test_mem_final_audit_5d_qdrant_session_turn_index_real_vendor.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/memory/e2e/test_mem_final_audit_5e_pgvector_session_turn_index_real_vendor.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/memory/e2e/test_mem_final_audit_5f_chroma_session_turn_index_real_vendor.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/platform_plugins/test_plugin_engine_cross_flow.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/governance/gr12/test_gr12_a2_mandatory_cla04_composition.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/plug_03/test_plug_03_gates.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_environment_profile.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_mem_audit5a_production_provider_admission.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_mem_audit5a_r2_provider_identity_binding.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_mem_audit5a_r3_durability_evidence_admission.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_mem_final_audit_5b_sqlite_production_restart_e2e.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_mem_final_audit_5d_r_session_turn_index_production_admission.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_memory_wiring.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_plugin_qualification_authority_wiring.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_policy_wiring.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_security_plugin_adoption_wiring.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_tool_skill_bootstrap_evidence.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/memory/test_mem_ent11_platform_boundaries.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/memory/test_mem_ent12a_entity_projection_observability.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/memory/test_memory_store_resolver.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py` | file | cond | EXISTS | T-AF2-RW-PLAN |

### PT-FC4 (T-IGX-043-D)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/distributed/contracts/kv_store.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:103 |
| `intergrax/distributed/providers/sqlite_kv_store.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:104 |
| `intergrax/distributed/providers/redis_kv_store.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:105 |
| `intergrax/integrations/providers/key_value_cache/redis/bundle.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:106 |
| `intergrax/skills/execution_pinning_persistence.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:107 |
| `intergrax/skills/execution_pinning_durability.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:108 |
| `tests/unit/skills/test_skill_execution_pinning_durability_guard.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:114 |
| `tests/unit/distributed/contracts/test_kv_durability_attestation.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:115 |
| `tests/unit/distributed/test_sqlite_kv_store_durability.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:116 |
| `tests/unit/distributed/test_redis_kv_store_durability.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:117 |
| `tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:118 |
| `tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-D/PLAN_T-IGX-043-D.md:119 |

### T-AF2-DEPS-BATCH (T-IGX-051-H)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `pyproject.toml` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `uv.lock` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `scripts/maintenance/check_dependency_ownership.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_deps_core_ownership_declared.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-E05 (T-IGX-044-F)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `scripts/governance/verify_iiac_audit_manifest.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/044-F/PLAN_T-IGX-044-F.md:82 |
| `scripts/governance/verify_iiac_role_separation.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/044-F/PLAN_T-IGX-044-F.md:83 |
| `scripts/governance/iiac_activation_gate.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/044-F/PLAN_T-IGX-044-F.md:84 |
| `docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/044-F/PLAN_T-IGX-044-F.md:85 |
| `tests/unit/governance/test_e05_role_separation_identity.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/044-F/PLAN_T-IGX-044-F.md:86 |
| `scripts/governance/iiac_common.py` | file | firm | EXISTS | T-AF2-RW-PLAN |

### PT-D08 (T-IGX-046-D)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `agents/README.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/046-D/PLAN_T-IGX-046-D.md:42 |
| `intergrax/integrations/providers/graph_store/neo4j/USAGE.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/046-D/PLAN_T-IGX-046-D.md:43 |

### T-AF2-RW-RUNTIME (T-IGX-051-D)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/llm_adapters/base/base_llm_adapter.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/llm_adapters/_shared/provider_stream_admission.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/llm_adapters/_shared/openai_completion_mapping.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/llm_adapters/contracts/*` | glob | firm | GLOB_MATCHES_26 | T-AF2-RW-PLAN |
| `intergrax/llm_adapters/providers/*` | glob | firm | GLOB_MATCHES_40 | T-AF2-RW-PLAN |
| `intergrax/runtime/llm/*` | glob | firm | GLOB_MATCHES_3 | T-AF2-RW-PLAN |
| `intergrax/multimedia/images_loader.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/multimedia/image_smart_loader.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `agents/model_routing_qualifier/qualification_types.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/ebh_6/_findings_register.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/llm_adapters/test_rw_runtime_stream_open_breaker.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/llm_adapters/test_rw_runtime_timeout_enforced.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/llm_adapters/test_rw_runtime_typed_error_contract_v1.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/llm_adapters/test_rw_runtime_served_model_attribution.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/llm_adapters/test_rw_runtime_refusal_normalization_matrix.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/llm_adapters/test_rw_runtime_capability_flag_parity.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-WD (T-IGX-043-I)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/platform/enterprise_audit/impact_map.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:119 |
| `intergrax/platform/enterprise_audit/changed_since.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:120 |
| `intergrax/platform/enterprise_audit/runner.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:121 |
| `intergrax/platform/enterprise_audit/digests.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:122 |
| `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:123 |
| `intergrax/platform/enterprise_audit/data/schemas/frz_coverage.v1.schema.json` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:124 |
| `intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:125 |
| `intergrax/platform/enterprise_audit/manifest.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:126 |
| `intergrax/platform/enterprise_audit/evidence.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:127 |
| `scripts/enterprise_audit/run_enterprise_audit.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:135 |
| `scripts/enterprise_audit/build_impact_map.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:136 |
| `scripts/enterprise_audit/mechanical_readiness_check.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:137 |
| `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:143 |
| `tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:144 |
| `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:145 |
| `tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:146 |
| `tests/qualification/ent_audit_x/unit/test_wd_mapped_family_checks.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:147 |
| `tests/qualification/ent_audit_x/integration/test_adversarial_adv01_30.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:153 |
| `tests/qualification/ent_audit_x/integration/test_runner_full_smoke.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:154 |
| `tests/qualification/ent_audit_x/l3/test_l3_clean_checkout_full.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:155 |
| `tests/qualification/ent_audit_x/l3/test_l3_freeze_real_85.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:156 |
| `tests/qualification/ent_audit_x/l3/test_l3_full_real_85.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:157 |
| `tests/qualification/ent_audit_x/unit/test_baseline_manifest_digest_promotion.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:158 |
| `tests/qualification/ent_audit_x/unit/test_frz_aud_w402_neg.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:159 |
| `tests/qualification/ent_audit_x/unit/test_frz_coverage_matrix_complete.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:160 |
| `tests/qualification/ent_audit_x/unit/test_manifest_and_exits.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:161 |
| `tests/qualification/ent_audit_x/_manifest_fixtures.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:162 |
| `tests/qualification/pdc_x/p6_impact/test_p6_impact_map_coverage.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:163 |
| `tests/qualification/pdc_x/p6_impact/test_p6_impact_tier3_root_rule.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:164 |
| `tests/qualification/pdc_x/p8_repair/test_r4_impact_map_classification.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:165 |
| `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_aud_01.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:184 |
| `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_ctl_01.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:185 |
| `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_debt_01.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:186 |
| `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_frz_01.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:187 |
| `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_hrn_01.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:188 |
| `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_trc_01.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:189 |
| `scripts/enterprise_audit/qual_wave_lib.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:190 |
| `tests/qualification/ent_audit_x/unit/test_evidence_publication_strict.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-I/PLAN_T-IGX-043-I.md:191 |

### PT-FC1 (T-IGX-043-A)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/runtime/nexus/agent_router.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:87 |
| `intergrax/runtime/nexus/nexus_loop.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:88 |
| `intergrax/runtime/nexus/execution/graph_executor.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:89 |
| `intergrax/runtime/execution/nexus_host_execution.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:90 |
| `intergrax/runtime/execution/debug_lab_nexus_loop.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:91 |
| `intergrax/runtime/execution/host_orchestration_loop_init_spec.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:92 |
| `intergrax/runtime/execution/environment_orchestration_materialization.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:93 |
| `intergrax/runtime/execution/host_orchestration_environment_spec_builder.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:94 |
| `intergrax/runtime/execution/worker_host_task_execution_composition.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:95 |
| `intergrax/runtime/task_memory/store.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:96 |
| `intergrax/runtime/events/store.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:97 |
| `intergrax/runtime/human/persistence_errors.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:98 |
| `intergrax/applications/_shared/harness_host_runtime.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:99 |
| `intergrax/applications/_shared/task_memory_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:100 |
| `intergrax/applications/_shared/host_orchestration_backend_spec_builder.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:101 |
| `intergrax/agents/defaults.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:102 |
| `agents/organization_worker/organization_worker_agent.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:103 |
| `agents/external_contractor_adapter/external_contractor_adapter_agent.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:104 |
| `intergrax/agents/authoring/acp_stub_reflex.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:105 |
| `applications/USAGE.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:111 |
| `intergrax/applications/USAGE.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:112 |
| `agents/dispute_analyst/notebooks/01_dispute_analyst_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:113 |
| `agents/dispute_intake/notebooks/01_dispute_intake_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:114 |
| `agents/dispute_scenario/notebooks/01_dispute_scenario_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:115 |
| `agents/dispute_strategist/notebooks/01_dispute_strategist_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:116 |
| `agents/intergrax_assistant/notebooks/01_intergrax_assistant_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:117 |
| `agents/legal/notebooks/01_legal_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:118 |
| `agents/local_indexer/notebooks/01_local_indexer_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:119 |
| `agents/local_search/notebooks/01_local_search_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:120 |
| `agents/local_synthesizer/notebooks/01_local_synthesizer_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:121 |
| `agents/signoff_probe/notebooks/01_signoff_probe_experiment.ipynb` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:122 |
| `applications/governed_contractor_application/tests/host/test_governed_contractor_canonical_execution.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:128 |
| `applications/legal_application/tests/serving/test_legal_fastapi_chat.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:129 |
| `applications/legal_application/tests/serving/test_legal_serving_identity.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:130 |
| `applications/local_workspace_application/tests/host/test_lkw_canonical_execution.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:131 |
| `intergrax/runtime/task/nexus_worker_execution.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:137 |
| `intergrax/runtime/task/worker_bootstrap.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:138 |
| `intergrax/applications/_shared/tool_wiring.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:139 |
| `intergrax/applications/_shared/scenario_runtime_baseline.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:140 |
| `applications/attestation_demo/host/factory.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:141 |
| `applications/dispute_sim_application/host/factory.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:142 |
| `applications/governed_contractor_application/host/factory.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:143 |
| `applications/intergrax_assistant_application/host/factory.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:144 |
| `applications/lab_application/host/factory.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:145 |
| `applications/legal_application/host/factory.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:146 |
| `applications/poc_template_application/host/factory.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:147 |
| `applications/research_application/host/factory.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:148 |
| `applications/local_workspace_application/host/host_runtime_composition.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:149 |
| `applications/local_workspace_application/model_runtime_proof/runtime.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:150 |
| `intergrax/harness/app.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:151 |
| `intergrax/scaffold/new_application.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:152 |
| `intergrax/scaffold/new_application_product.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:153 |
| `tests/unit/runtime/nexus/test_fc1_agent_router_posture_and_fallback.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:161 |
| `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:162 |
| `tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:163 |
| `tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:164 |
| `tests/unit/architecture/test_fc1_router_and_loop_construction_gate.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:165 |
| `tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:166 |
| `tests/acceptance/agent_os/conftest.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:172 |
| `tests/acceptance/agent_os/test_acp_pattern_agents.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:173 |
| `tests/acceptance/agent_os/test_agent_os_scenarios.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:174 |
| `tests/integration/applications/architecture/test_harden_1d_durable_problem_store_failure_semantics.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:175 |
| `tests/integration/applications/test_unified_execution_entry_j2.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:176 |
| `tests/integration/applications/test_unified_execution_entry_j3.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:177 |
| `tests/integration/debug/test_debug_api_injected_trace_store_b09.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:178 |
| `tests/integration/eval/test_nexus_eval_runner.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:179 |
| `tests/integration/runtime/adaptive/test_nexus_adaptive_signal.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:180 |
| `tests/integration/runtime/long_running/test_long_running_scheduler_j4.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:181 |
| `tests/integration/runtime/test_declarative_policy_hitl_nexus_e2e.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:182 |
| `tests/integration/runtime/test_graph_cancellation.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:183 |
| `tests/integration/runtime/test_graph_failure_recovery.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:184 |
| `tests/integration/runtime/test_harness_hitl_pagerduty.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:185 |
| `tests/integration/runtime/test_nexus_loop_echo.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:186 |
| `tests/integration/runtime/test_nexus_loop_governance.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:187 |
| `tests/integration/runtime/test_nexus_loop_guardrail_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:188 |
| `tests/integration/runtime/test_nexus_loop_hitl_hooks.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:189 |
| `tests/integration/runtime/test_nexus_loop_human_request_v2.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:190 |
| `tests/integration/runtime/test_nexus_loop_long_running.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:191 |
| `tests/integration/runtime/test_nexus_loop_research.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:192 |
| `tests/integration/runtime/test_nexus_loop_runtime_event_persistence.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:193 |
| `tests/integration/runtime/test_nexus_loop_security_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:194 |
| `tests/integration/runtime/test_observability_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:195 |
| `tests/integration/runtime/test_planning_decision_record_gate.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:196 |
| `tests/integration/runtime/test_platform_wiring_observability.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:197 |
| `tests/integration/runtime/test_sandbox_uaep.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:198 |
| `tests/integration/runtime/test_shadow_workspace_uaep.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:199 |
| `tests/integration/runtime/test_terminal_diagnostic_production_e2e.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:200 |
| `tests/qualification/bg_01/test_bg_01_gates.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:201 |
| `tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_security_composition.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:202 |
| `tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_middleware_composition.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:203 |
| `tests/qualification/host_01/test_host_01_gates.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:204 |
| `tests/qualification/mcp_server_x/reference_app/http_serve.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:205 |
| `tests/qualification/mcp_server_x/test_mcp_server_x_p6_reference_l3_qualification.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:206 |
| `tests/qualification/mcp_server_x/test_mcp_server_x_p7_failure_injection_l2.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:207 |
| `tests/qualification/mcp_server_x/test_mcp_server_x_p7_l2_composition_exposure.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:208 |
| `tests/qualification/mcp_server_x/test_mcp_server_x_p7_l2_security_trace.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:209 |
| `tests/unit/agent_distribution/test_bounded_multi_agent_fanout.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:210 |
| `tests/unit/applications/test_application_security_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:211 |
| `tests/unit/applications/test_harness_guardrail_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:212 |
| `tests/unit/applications/test_http_canonical_execution.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:213 |
| `tests/unit/applications/test_mcp_canonical_execution.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:214 |
| `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:215 |
| `tests/unit/applications/test_mcp_host_mount.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:216 |
| `tests/unit/applications/test_mcp_server_engine.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:217 |
| `tests/unit/applications/test_mcp_server_p4_lifecycle.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:218 |
| `tests/unit/applications/test_reference_apps_mcp.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:219 |
| `tests/unit/architecture/test_ebh_2d_c_r4_canonical_runtime_agent_port.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:220 |
| `tests/unit/architecture/test_ebh_2d_c_r6_canonical_agent_routing_decision_closure.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:221 |
| `tests/unit/architecture/test_ebh_4_r1_r3_b6_r2_runtime_conformance.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:222 |
| `tests/unit/contracts/test_execution_identity.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:223 |
| `tests/unit/debug/test_g6_api.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:224 |
| `tests/unit/debug/test_h3_interaction_intake.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:225 |
| `tests/unit/debug/test_h5_teams_interaction_intake.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:226 |
| `tests/unit/runtime/architecture/test_agent_routing_policy.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:227 |
| `tests/unit/runtime/architecture/test_gr10_r12_orchestration_continuation_architecture_gates.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:228 |
| `tests/unit/runtime/architecture/test_gr10_r13_r2_production_reliability_composition_adoption.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:229 |
| `tests/unit/runtime/architecture/test_gr10_r9_r3_canonical_topology_slot_mse.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:230 |
| `tests/unit/runtime/architecture/test_gr10_r9_r4_topology_mse_composition_mandatory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:231 |
| `tests/unit/runtime/architecture/test_gr10_r9_r5_topology_factory_fail_closed.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:232 |
| `tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:233 |
| `tests/unit/runtime/architecture/test_npsc5d_r3_final_qualification.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:234 |
| `tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:235 |
| `tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:236 |
| `tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:237 |
| `tests/unit/runtime/architecture/test_obs_diag_port_1_gates.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:238 |
| `tests/unit/runtime/diagnostics/test_ue_11f_observability_diagnostic_correlation.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:239 |
| `tests/unit/runtime/events/test_observability_emission_coverage.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:240 |
| `tests/unit/runtime/execution/budget/test_ue_8b1r1_ledger_lifecycle.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:241 |
| `tests/unit/runtime/execution/budget/test_ue_8b1r2_preserve_run_budget_through_nexus_entry.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:242 |
| `tests/unit/runtime/execution/budget/test_ue_9ar1_preserve_run_budget_across_redelivery.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:243 |
| `tests/unit/runtime/execution/lineage/test_host_task_lineage_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:244 |
| `tests/unit/runtime/execution/lineage/test_host_task_resume_lineage_identity.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:245 |
| `tests/unit/runtime/execution/lineage/test_terminal_conflict_seal.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:246 |
| `tests/unit/runtime/execution/test_attempt_lifecycle_durability_gate.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:247 |
| `tests/unit/runtime/execution/test_decision_optionality.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:248 |
| `tests/unit/runtime/execution/test_decision_orchestration_recovery.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:249 |
| `tests/unit/runtime/execution/test_execution_runtime.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:250 |
| `tests/unit/runtime/execution/test_host_task_terminal_publisher.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:251 |
| `tests/unit/runtime/execution/test_orchestration.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:252 |
| `tests/unit/runtime/execution/test_orchestration_topology_e2e_proof.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:253 |
| `tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:254 |
| `tests/unit/runtime/execution/test_ue_11e_resume_recovery.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:255 |
| `tests/unit/runtime/governance/test_g3b_governance_coverage.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:256 |
| `tests/unit/runtime/governance/test_gov_x2_r3_post_run_tenant_contract.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:257 |
| `tests/unit/runtime/hooks/test_hook_parity_b06.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:258 |
| `tests/unit/runtime/human/test_human_decisions.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:259 |
| `tests/unit/runtime/long_running/test_pba_fix_a_checkpoint_port_consumption.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:260 |
| `tests/unit/runtime/long_running/test_runtime_checkpoint.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:261 |
| `tests/unit/runtime/nexus/test_handle_task_execution_identity.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:262 |
| `tests/unit/runtime/nexus/test_nexus_loop_finalization_hook_failure.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:263 |
| `tests/unit/runtime/nexus/test_nexus_loop_helpers.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:264 |
| `tests/unit/runtime/nexus/test_nexus_runtime_event_metric_scope_r3.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:265 |
| `tests/unit/runtime/nexus/test_nexus_runtime_event_metric_scope_r4.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:266 |
| `tests/unit/runtime/registry/test_agent_registry_read_boundary.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:267 |
| `tests/unit/runtime/registry/test_capability_routing_acp_con6.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:268 |
| `tests/unit/runtime/security/test_sec_ent.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:269 |
| `tests/unit/runtime/security/test_sec_planes.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:270 |
| `tests/unit/runtime/task/test_host_task_execution_run_adapter.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:271 |
| `testing_support/agent_distribution/dg001_canonical_multi_agent_diagnostic_harness.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:277 |
| `testing_support/agent_distribution/multi_agent_coordination_qualification_harness.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:278 |
| `testing_support/obs_universal_spine/diagnostic_execution_stack.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:279 |
| `tests/integration/agent_distribution/test_canonical_agent_lifecycle_architecture_gate.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:285 |
| `tests/qualification/pdc_x/d7/test_d7_recurrence.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:286 |
| `tests/qualification/sched_01/test_sched_01_gates.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:287 |
| `tests/unit/applications/_shared/test_obs_diag_x3_universal_spine_adoption.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:288 |
| `tests/unit/applications/architecture/test_diag_platform_adoption_gate.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:289 |
| `tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:290 |
| `tests/unit/scaffold/test_scaffold_harness_alignment.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:291 |
| `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:297 |
| `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:298 |
| `docs/project/architecture/MEMORY.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:299 |
| `docs/project/architecture/satellites/OBSERVABILITY_extended_depth.md` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-A/PLAN_T-IGX-043-A.md:300 |

### PT-FC2-E (T-IGX-043-E)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `scripts/maintenance/check_production_capacity_adapters.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-E/PLAN_T-IGX-043-E.md:76 |
| `scripts/maintenance/check_sandbox_policy_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-2/043-E/PLAN_T-IGX-043-E.md:77 |

### T-AF2-RW-RECOVERY (T-IGX-051-E)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/background_tasks/*` | glob | firm | GLOB_MATCHES_6 | T-AF2-RW-PLAN |
| `intergrax/queueing/providers/document_store/document_store_task_queue.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/queueing/providers/document_store/colocated_worker.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/hosting/engine/engine.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/hosting/shutdown.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/hosting/runner.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/long_running/scheduler.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/long_running/uncertain_reconciler.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/durable_registry_projection_input_store.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/queue_worker_wiring.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `applications/local_workspace_application/background_ingest/worker_handler.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/memory/retention_sweep.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/queueing/test_rw_recovery_no_double_execution.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/hosting/test_rw_recovery_drain_wait_for_idle.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/long_running/test_rw_recovery_uncertain_reconciler.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_rw_recovery_durable_cas_projection.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/queueing/test_rw_recovery_producer_identity.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### T-AF2-A4007-CALLERS (T-IGX-051-I)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/applications/_shared/import_policy.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:98 |
| `intergrax/applications/_shared/agent_resolution.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:99 |
| `intergrax/applications/_shared/wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:100 |
| `intergrax/applications/_shared/environment_conformance.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:101 |
| `intergrax/applications/_shared/roster_agent_contract_authority.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:102 |
| `scripts/maintenance/check_application_migrations.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:103 |
| `scripts/gates/check_application_production_gates.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:104 |
| `tests/unit/applications/test_a4007_caller_import_refusal.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:105 |
| `tests/unit/applications/test_a4007_import_policy_derivation.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:106 |
| `tests/unit/scripts/test_a4007_migration_gate_allowlist.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:107 |
| `intergrax/applications/_shared/package_wiring.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:109 |
| `intergrax/applications/_shared/environment_diff_wiring.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md:109 |

### PT-FC3 (T-IGX-043-C)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/integrations/contracts/base.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:125 |
| `intergrax/integrations/contracts/llm_guardrail.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:126 |
| `intergrax/core/security/security_configuration.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:127 |
| `intergrax/rag/embedding/registry/provider_authority.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:128 |
| `intergrax/rag/embedding/registry/profile.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:129 |
| `intergrax/rag/embedding/runtime/resolver.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:130 |
| `intergrax/rag/embedding/bootstrap/default_embedding_engine.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:131 |
| `intergrax/rag/vectorstore/bootstrap/integration_vectorstore.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:132 |
| `intergrax/runtime/notifications/backend_contract.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:133 |
| `intergrax/runtime/notifications/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:134 |
| `intergrax/model_inference/contracts.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:135 |
| `intergrax/model_inference/adapters/stub_vision.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:136 |
| `intergrax/model_inference/adapters/triton_vision.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:137 |
| `intergrax/model_inference/adapters/huggingface_inference_vision.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:138 |
| `intergrax/model_inference/workers/task_runner.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:139 |
| `intergrax/tools/providers/vision/inference_support.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:140 |
| `intergrax/tools/providers/vision/contracts.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:141 |
| `intergrax/tools/providers/vision/service.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:142 |
| `intergrax/runtime/interactions/verification/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:143 |
| `intergrax/runtime/interactions/verification/contract.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:144 |
| `intergrax/runtime/interactions/intake_service.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:145 |
| `intergrax/applications/_shared/interaction_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:146 |
| `intergrax/integrations/providers/llm_guardrail/_factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:147 |
| `intergrax/integrations/providers/llm_guardrail/_stub_backend.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:148 |
| `intergrax/integrations/providers/llm_guardrail/_pattern_scanner.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:149 |
| `intergrax/integrations/providers/llm_guardrail/bundles/llm_guard.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:150 |
| `intergrax/integrations/providers/llm_guardrail/bundles/presidio.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:150 |
| `intergrax/integrations/providers/llm_guardrail/bundles/guardrails_ai.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:150 |
| `intergrax/integrations/providers/llm_guardrail/bundles/llama_guard.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:150 |
| `intergrax/integrations/providers/llm_guardrail/bundles/nemo_guardrails.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:150 |
| `intergrax/integrations/providers/llm_guardrail/bundles/bedrock_guardrails.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:150 |
| `intergrax/integrations/providers/llm_guardrail/bundles/http_guardrail.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:150 |
| `intergrax/applications/_shared/environment_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:157 |
| `intergrax/applications/_shared/guardrail_runtime_bridge.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:158 |
| `intergrax/applications/_shared/harness_host_auxiliary_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:159 |
| `intergrax/applications/_shared/notification_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:160 |
| `intergrax/model_inference/registry/vision_profile.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:161 |
| `intergrax/rag/bootstrap/entry_point_load.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:162 |
| `intergrax/rag/bootstrap/rag_stack_bootstrap.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:162 |
| `intergrax/rag/bootstrap/hierarchical_bootstrap.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:162 |
| `intergrax/rag/document_splitters/bootstrap/default_chunking_engine.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:163 |
| `intergrax/rag/rerankers/bootstrap/reranker_bootstrap.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:164 |
| `intergrax/rag/retrievers/bootstrap/retriever_bootstrap.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:165 |
| `intergrax/tools/providers/rag/scope.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:166 |
| `intergrax/runtime/long_running/notification.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:167 |
| `intergrax/runtime/long_running/coordinator.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:167 |
| `applications/local_workspace_application/host/host_runtime_composition.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:168 |
| `intergrax/scaffold/new_application.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:169 |
| `intergrax/scaffold/new_application_product.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:169 |
| `intergrax/debug/app.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:170 |
| `intergrax/lab/organization_worker.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:171 |
| `applications/lab_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:172 |
| `applications/dispute_sim_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:173 |
| `applications/governed_contractor_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:173 |
| `applications/intergrax_assistant_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:173 |
| `applications/legal_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:173 |
| `applications/local_workspace_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:173 |
| `applications/poc_template_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:173 |
| `applications/research_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:173 |
| `applications/local_workspace_application/workspaces/indexed_vector_verifier.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:174 |
| `applications/local_workspace_application/scripts/lkw_ollama_embedding_bootstrap.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:175 |
| `applications/local_workspace_application/scripts/run-lkw-file-watcher-e2e-proof.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:175 |
| `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:183 |
| `tests/unit/rag/vectorstore/test_fc3_vectorstore_selector_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:183 |
| `tests/unit/runtime/notifications/test_fc3_notification_selector_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:183 |
| `tests/unit/model_inference/test_fc3_vision_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:183 |
| `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:183 |
| `tests/unit/integrations/providers/llm_guardrail/test_fc3_guardrail_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:183 |
| `tests/unit/model_inference/test_remote_vision_adapters.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:185 |
| `tests/unit/model_inference/test_model_inference_registry.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:185 |
| `tests/unit/rag/embedding/test_embedding_profile.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:185 |
| `tests/unit/rag/vectorstore/test_integration_vectorstore_bootstrap.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:185 |
| `tests/unit/rag/vectorstore/test_integration_vectorstore_scope.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:185 |
| `tests/unit/integrations/providers/notification_channel/test_slack.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:185 |
| `tests/unit/integrations/providers/notification_channel/test_teams.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:185 |
| `tests/unit/test_dep1_dependency_foundation.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:185 |
| `tests/qualification/pdc_x/reg_p/test_reg_p_ollama_single_row.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-C/PLAN_T-IGX-043-C.md:185 |
| `tests/acceptance/agent_os/test_lab_application.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/e2e/rag/test_hybrid_multi_source_rag_langgraph.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/e2e/rag/test_rag_full_runtime_e2e.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/applications/architecture/harden_4f_mongo_support.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/applications/test_unified_execution_entry_j1.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/rag/rerankers/test_embedding_cosine_reranker.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/runtime/diag_final_otel_support.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/runtime/test_terminal_diagnostic_production_e2e.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/prod_q/_p0_classification.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/prod_q/_prod_q_strict_http_invocation_identity_support.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_c_dim8_cross_tenant_integrated_gate.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_a_cross_tenant_strict.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_dur_02_gca_composition_binding.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_pis_02_provider_invocation_lifecycle_binding.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_r2_gca_startup_continuation.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_r3_startup_sqlite_isolation.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_r4_production_startup_path.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_r6_strict_gca_lifecycle.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_impl_01_r2_final_durability_and_qualification.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/local_workspace_application/test_lkw_ask_workspace_live_proof.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/local_workspace_application/test_lkw_file_watcher_e2e_proof.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/local_workspace_application/test_lkw_product_quickstart.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/task_control_product_host_test_support.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_application_startup_guards.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_guardrail_output_hooks.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_harness_guardrail_wiring.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_harness_task_control_wiring.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_interaction_wiring_public_surface.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_lab_harness_api_key_required.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_mcp_server_p4_lifecycle.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_mcp_surface_opt_in.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_mcp_surface_opt_in_runtime.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_poc_template_application.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_product_observability_host_composition.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_production_factory_contract_ac3.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_rag_authoritative_plugin_evidence.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_reference_apps_mcp.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_task_control_bypass_closure.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_task_control_host_composition.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/debug/test_h3_interaction_intake.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/debug/test_h5_teams_interaction_intake.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/integrations/providers/llm_guardrail/test_chained_backend.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/integrations/providers/llm_guardrail/test_http_guardrail_smoke.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/integrations/providers/llm_guardrail/test_pba_fix_c_guardrail_provider_config.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/integrations/providers/notification_channel/test_webhook.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/integrations/test_llm_guardrail_contract.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/integrations/test_p2_002_b3_runtime_binding_extension.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/model_inference/test_celery_modality_execution.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/model_inference/test_opencv_vision.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/model_inference/test_vision_profile.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/platform_proofs/scenarios/verified_product_identification/test_bootstrap_artifact_alignment.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/platform_proofs/scenarios/verified_product_identification/test_bootstrap_orchestrator.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/platform_proofs/scenarios/verified_product_identification/test_embedding_configuration.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/platform_proofs/scenarios/verified_product_identification/test_embedding_materialization.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/platform_proofs/scenarios/verified_product_identification/test_intergrax_embedding_adapter.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/platform_proofs/scenarios/verified_product_identification/test_platform_vector_search_adapter.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/platform_proofs/scenarios/verified_product_identification/test_qdrant_vector_candidate_search.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/rag/bootstrap/test_rag_entry_point_load.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/rag/document_splitters/test_optional_langchain_splitter.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/rag/embedding/test_llama_cpp_embedding_provider.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/rag/embedding/test_p2_002_b3_embedding_runtime_binding.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/rag/embedding/test_vllm_embedding_provider.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/architecture/test_gr10_r13_r3_production_host_reliability_adoption.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/architecture/test_npsc3c_d_canonical_execution_engine_conformance_gate.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/hooks/test_tool_and_selection_hooks.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/interactions/test_interaction_intake_service.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/interactions/test_teams_activity_adapter.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/notifications/test_log_notification_factory.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/notifications/test_webhook_delivery.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/tools/providers/rag/test_rag_scope.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/tools/providers/test_ml_batch_predict.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/tools/providers/test_modality_tools.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/tools/providers/test_vision_segment_ml_explain.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/tools/providers/vision/test_media_trust_boundary.py` | file | cond | EXISTS | T-AF2-RW-PLAN |

### T-AF2-RW-SUPPLY (T-IGX-051-B)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/agent_distribution/trust.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/agent_distribution/admin_service.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/deploy_activate_pipeline.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/production_agent_capability_runtime.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `applications/governed_contractor_application/host/single_node_profile_v1_strict_startup.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/core/plugins/*` | glob | firm | GLOB_MATCHES_9 | T-AF2-RW-PLAN |
| `intergrax/runtime/security/security_audit_trail.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/security/tenant_scope.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/security/defense_plugin_loader.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/core/security_bootstrap.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/security/test_rw_supply_defense_plugin_provenance_required.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `intergrax/contracts/agent_run.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/core/security/secret_safety.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/llm_adapters/governance/quota.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/security_assembly_resolver.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/tools/providers/rag/service.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/architecture/online_evaluation_registry.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/nexus/artifacts/retention.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/nexus/artifacts/store_base.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/nexus/artifacts/in_memory_artifact_store.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/security_runtime_bridge.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `scripts/maintenance/check_static_secret_literals.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `scripts/maintenance/check_sandbox_isolation_strength.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_rw_supply_trust_hook_before_persist.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/core/plugins/test_rw_supply_installed_file_digest.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/security/test_rw_supply_audit_trail_tamper_evidence.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/security/test_rw_supply_fail_open_set.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_rw_supply_strict_validator_refuses_weak.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/scripts/test_rw_supply_static_secret_gate.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/nexus/test_rw_supply_retention_and_legal_hold.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### T-AF2-RW-GATES (T-IGX-051-F)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `scripts/gates/check_mvp_promotion_gates.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/scenario_runtime_baseline.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `platform_proofs/scenarios/ai_incident_investigation/*` | glob | firm | GLOB_MATCHES_58 | T-AF2-RW-PLAN |
| `tests/unit/llm_adapters/test_fixture_strict_adapter_parity.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/events/payloads/canonical.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/observability/qualification_runtime_trace.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/architecture/evaluation_assets.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/business_agent_deploy_wiring.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/contracts/environment_profile/sub_profiles.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/plug_03/*` | glob | firm | GLOB_MATCHES_4 | T-AF2-RW-PLAN |
| `scripts/maintenance/check_not_composed_capabilities.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `intergrax/integrations/providers/document_store/mongodb/client.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/integrations/providers/document_store/mongodb/opens.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_kr17_vendor_sdk_inside_opens.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/nexus/agents/reference_harness_runtime.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agents/test_reference_harness_strict.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/policy/pre_model_principal.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agents/test_acp_token_budget_enforcement.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/integrations/providers/document_store/mongodb/integration.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/integrations/_shared/conformance.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_kr03_kr04_kr07_cause.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_not_composed.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_partial_inventory.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_gate_completeness.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_contract_census.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_transport_semantics.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_plugin_contracts.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_production_only_bypass.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_event_schema_reconciled.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_eval_digest_enforced.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_gates_scenario_get_steps_contract.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_rw_gates_registry_reuse.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/xd/test_rw_gates_model_provider_tenant_data.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/scripts/test_rw_gates_promotion_gate_binding.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### T-AF2-RW-TRUST (T-IGX-051-C)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/llm/messages.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/nexus/context/graph_assembly.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/context/contracts.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/context/policy/egress.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/context/provider_lifecycle.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/context/planner.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/context/formatter.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/runtime/test_context_provider_wiring.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agents/authoring/test_llm_router.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agents/test_runtime_request_bridge.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/context/budget/test_ce02_r1h_provider_base_identity.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/context/budget/test_ce02_r1i_mandatory_provider_hard_budget.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/context/test_ce_02_r1d_safe_degradation.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/context/test_context_formatter.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/context/test_legacy_bridge_providers.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/context/test_session_history_provider.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/llm/test_messages.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/nexus/context/test_context_compiler.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/nexus/context/test_uaep_assemble.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/rag/*` | glob | firm | GLOB_MATCHES_127 | T-AF2-RW-PLAN |
| `tests/integration/rag/*` | glob | firm | GLOB_MATCHES_35 | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/d2_repair/RAG_RED_CLASSIFICATION.md` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/context/test_rw_trust_untrusted_never_system_role.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/context/test_rw_trust_egress_default_most_restrictive.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/context/test_rw_trust_planner_ledger_full_message.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/llm/test_rw_trust_orphan_tool_call_id_strict.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-FC3-F (T-IGX-043-F)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `tests/qualification/config_x/_config_x_semantic_production_scan.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-F/PLAN_T-IGX-043-F.md:79 |
| `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-F/PLAN_T-IGX-043-F.md:80 |

### PT-WF (T-IGX-043-J)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/applications/_shared/evaluation_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:172 |
| `intergrax/applications/_shared/policy_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:173 |
| `intergrax/applications/_shared/adaptive_wiring.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:174 |
| `intergrax/applications/_shared/checkpoint_introspection_routes.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:175 |
| `intergrax/scaffold/new_application_product.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:176 |
| `applications/local_workspace_application/serving/schemas.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:177 |
| `applications/local_workspace_application/serving/fastapi_router.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:178 |
| `applications/local_workspace_application/serving/workspace_routes.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:179 |
| `applications/local_workspace_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:180 |
| `applications/dispute_sim_application/serving/schemas.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:181 |
| `applications/dispute_sim_application/serving/fastapi_router.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:182 |
| `applications/dispute_sim_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:183 |
| `applications/dispute_sim_application/host/settings.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:184 |
| `applications/attestation_demo/serving/fastapi_router.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:185 |
| `applications/attestation_demo/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:186 |
| `applications/governed_contractor_application/serving/schemas.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:187 |
| `applications/legal_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:188 |
| `applications/legal_application/host/settings.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:189 |
| `applications/research_application/host/factory.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:190 |
| `applications/research_application/host/settings.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:191 |
| `intergrax/contracts/decision_authoritative_exposure.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:197 |
| `intergrax/runtime/task/task.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:198 |
| `intergrax/runtime/nexus/nexus_loop.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:199 |
| `intergrax/runtime/nexus/orchestration/nexus_decision_exposure.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:200 |
| `intergrax/runtime/nexus/orchestration/task_finisher.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:201 |
| `intergrax/runtime/nexus/orchestration/graph_runner.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:202 |
| `intergrax/runtime/decision_exposure_mapping.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:203 |
| `intergrax/runtime/execution/decision_finalization_conformance.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:204 |
| `intergrax/agent_distribution/deployment.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:210 |
| `intergrax/applications/_shared/colocated_serving_runtime_composition.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:211 |
| `intergrax/applications/_shared/production_lifecycle.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:212 |
| `intergrax/applications/_shared/strict_production_serving_launcher.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:213 |
| `intergrax/runtime/execution/child.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:219 |
| `intergrax/runtime/execution/qualified_capability_execution_composition.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:220 |
| `intergrax/runtime/execution/runtime.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:221 |
| `intergrax/compat/langchain/documents.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:222 |
| `intergrax/rag/document_loaders/compat/legacy_runtime_document.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:223 |
| `intergrax/experiments/store.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:224 |
| `intergrax/debug/router.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:225 |
| `intergrax/debug/app.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:226 |
| `intergrax/applications/_shared/harness_auth.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:227 |
| `intergrax/runtime/enterprise_reliability/sqlite_provider_invocation_store.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:233 |
| `intergrax/contracts/provider_invocation.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:234 |
| `intergrax/contracts/provider_invocation_store.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:235 |
| `intergrax/runtime/enterprise_reliability/admission_boundary.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:236 |
| `intergrax/contracts/enterprise_reliability/admission_boundary.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:237 |
| `intergrax/runtime/adaptive/policy_learning_approval.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:238 |
| `tests/unit/applications/test_wf_tenant_evaluation_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:248 |
| `tests/unit/applications/test_wf_serving_tenant_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:249 |
| `tests/unit/architecture/test_wf_no_default_tenant_in_request_contracts.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:250 |
| `tests/unit/runtime/nexus/test_wf_exposure_payload_typed_seam.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:251 |
| `tests/unit/applications/test_wf_colocated_drain_protocol.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:252 |
| `tests/unit/compat/langchain/test_wf_trusted_scope_conversion.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:253 |
| `tests/unit/debug/test_wf_experiment_tenant_scope.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:254 |
| `tests/unit/runtime/execution/test_wf_decision_finalization_strict.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:255 |
| `tests/unit/runtime/execution/test_wf_child_admission_narrowing.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:256 |
| `tests/unit/applications/test_wf_debug_router_guard_idp.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:257 |
| `tests/unit/runtime/enterprise_reliability/test_wf_provider_invocation_tenant.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:263 |
| `tests/qualification/ebh_6/_q2d2_technical_qualification.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:269 |
| `tests/qualification/ebh_6/test_ebh_6_q2d2_dual_path.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:270 |
| `tests/qualification/ebh_6/_contracts_typing_qualification.py` | file | cond | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-J/PLAN_T-IGX-043-J.md:276 |

### T-AF2-RW-TENANT (T-IGX-051-A)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/runtime/sandbox/hosted_session.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/environment_wiring.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/tools/providers/storage/service.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/tools/providers/workspace/service.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/tools/registry/wiring.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/nexus/artifacts/governed_access.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/architecture/online_evaluation_models.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/policy_wiring.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/harness_host_runtime.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/agent_distribution/admin_service.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/mcp_harness_principal.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/rag/governance/embedding_version_policy.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/memory_wiring.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/memory_provider_admission.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/sandbox/test_rw_tenant_hosted_session_per_tenant.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/tools/providers/test_rw_tenant_storage_workspace_keys.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/nexus/test_rw_tenant_artifact_read_binding.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_rw_tenant_online_evaluation_observation.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_rw_tenant_install_cas.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_rw_tenant_mcp_client_asserted_tenant.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/rag/governance/test_rw_tenant_reindex_hook_scope.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_rw_tenant_strict_unscoped_memory.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/mcp_application_server.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/mcp_nexus_server.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/mcp_catalog_tools.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/rag/ingest/ingest_pipeline.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/rag/governance/source_freshness.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_evaluation_plane.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/tools/providers/storage/test_storage_tools.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/tools/providers/test_t10_catalog_tools.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/d8_repair/test_d8r_storage_tools_gate.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/p8_repair/test_r3_governed_keys.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/agent_distribution/admin_models.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/agent_distribution/stores.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/agent_distribution/in_memory_stores.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/agent_distribution/sqlite_stores.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/architecture/online_evaluation.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/nexus/nexus_loop.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/tools/providers/eval/service.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/architecture/online_evaluation_registry.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `intergrax/integrations/contracts/sandbox_host.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/control_planes/ctrl_x/catalog.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/rag/governance/test_embedding_version_policy.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/rag/ingest/test_ingest_embedding_version_policy.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/d2/test_d2_rag_freshness.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/d2/test_d2_rag_freshness_wiring.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/d2_repair/test_d2r_freshness_hook_isolation.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_memory_wiring.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_mem_audit5a_production_provider_admission.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/agent_distribution/test_ac6_trust_lifecycle_e2e.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/integration/agent_distribution/test_enterprise_projection_rehydration_e2e.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/d1/_d1_harness.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/d1/test_d1_adversarial_lifecycle.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/d1/test_d1_upgrade_retire.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/d1_repair/test_d1_repair_scope_and_evidence.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/pdc_x/xd/test_xd_11_observability.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_contract_authority.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_distribution_build_remediation.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_distribution_control_plane_governance.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_distribution_desired_state_idempotent_scope.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_distribution_desired_state_remediation.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_distribution_drain_recovery_remediation.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_distribution_tenant_denial_remediation.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_manager.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_platform_admin_service.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_platform_composite_scope_coexistence.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_agent_platform_cross_layer_acceptance.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_delegated_subtasks.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/agent_distribution/test_task_scoped_agents.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_ac3_phase5_production_lifecycle_e2e.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/applications/test_historical_projection_isolation_phase4e.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_ebh_2e_ar1_a_r1_r1_install_authority_closure.py` | file | cond | EXISTS | T-AF2-RW-PLAN |

### T-AF2-RW-TYP (T-IGX-051-G)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `scripts/maintenance/check_typ_ratchet.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/typ_ratchet_baseline.v1.json` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_typ_ratchet.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `intergrax/contracts/governed_execution_result.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/contracts/external_operation_redrive_trusted_effect_contracts.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_typ05_string_dispatch_reduction.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-T01 (T-IGX-043-L)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `scripts/maintenance/check_runtime_event_tenant_propagation.py` | file | firm | EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-L/PLAN_T-IGX-043-L.md:88 |
| `tests/unit/scripts/test_t01_runtime_event_tenant_propagation_emitters.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-L/PLAN_T-IGX-043-L.md:89 |
| `tests/unit/architecture/test_t01_nt*.py` | new_glob | firm | NEW_GLOB_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-L/PLAN_T-IGX-043-L.md:90 |
| `tests/qualification/tenant_x/test_t01_*.py` | new_glob | firm | NEW_GLOB_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-L/PLAN_T-IGX-043-L.md:91 |
| `tests/qualification/trace_x/test_t01_*.py` | new_glob | firm | NEW_GLOB_PARENT_EXISTS | .tmp/session/arch-freeze/w3-packet-4/043-L/PLAN_T-IGX-043-L.md:92 |

