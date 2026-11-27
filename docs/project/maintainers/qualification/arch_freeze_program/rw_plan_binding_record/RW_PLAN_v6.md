# T-AF2-RW-PLAN — RW product train plan v6 (ARCH-FREEZE, MEP-AF2 v2.1 W2)

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
- PT-WA: Part F removed (A4-007 -> PT-WB Part E); Part E and its test NT-RW-WA-01 are `cond` with the fallback to PT-G01 (Operator Q-WA-05 open); KR-18 added (`environment_profile/root.py` + new test).
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

**Counts.** 22 train tickets; 17 overlapping ticket pairs (all serialized); 9 lanes; 168 negative-test rows; 192 dispositions; coverage gaps reported by `coverage_check.py`: 0; 3 open questions.

**Scope rules.** Each scope entry is one of `file` (exists at HEAD), `new` (absent at HEAD, parent directory exists) or `glob` (matches >= 1 tracked file). Class `firm` is binding; `cond` is binding only under the stated condition. EQO-A tickets take their scope from the write set of their plan (overlap tables, forbidden lists and documents owned elsewhere excluded); the MEP-AF2 tickets have no prior plan and get their scope here. The full per-path list with its source line is appendix A1 and `rw_plan.v1.json` (`tickets[].scope`).

## (a) Ticket inventory

`Tnn` is the short id used in the overlap matrix. Closes = IDs this ticket owns (exactly-once ownership, section g). Contributes = IDs owned elsewhere that this ticket helps close.

| Id | Node | Ticket | Sens | Lane | Firm product / firm total / cond paths | Closes (owned) | Contributes to | Notes |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| T01 | PT-WA | T-IGX-043-G | PRODUCT | 0 | 41 / 59 / 3 | FRZ: FRZ-BND-04; finding: A2-009, A4-003, A4-004, X-A4-004; other: KR-18 | FRZ-EXE-02, FRZ-PLG-05 | W-A boundary inversion (12 runtime modules) and Tier-2 agent seams. v6: Part E (S8b, two LKW production modules through facades, DP-20) is firm, a separable last commit group (OPERATOR_PACKET_3, Q-WA-05 option A); Part F (A4-007, NT-WA-18..22) moves to PT-WB Part E (dedupe D-03, accepted v2) |
| T02 | PT-WB | T-IGX-043-H | PRODUCT | 1 | 19 / 33 / 15 | FRZ: FRZ-OWN-03, FRZ-OWN-05, FRZ-PRD-05; finding: A4-007, A4-020, A4-040, X-A4-016, X-A8-003 | FRZ-BND-05, FRZ-CFG-02, FRZ-CFG-06, FRZ-CTR-01, FRZ-CTR-02, FRZ-CTR-06, FRZ-SEC-04 | W-B one LLM resolver, profile projections, provider roots, serving, importlib allowlist |
| T03 | PT-FC2 | T-IGX-043-B | PRODUCT | 2 | 10 / 22 / 1 | FRZ: FRZ-PRD-03, FRZ-REL-07, FRZ-REL-09, FRZ-REL-10; finding: A4-030, A4-032, A4-035, A4-036; other: KR-19 | - | FC-2 production wiring: capacity, queue, memory, observability, sandbox. KR-19 (import cycle observability_otlp -> export_boundary) added in v2 (Q-RW-10) |
| T04 | PT-FC4 | T-IGX-043-D | PRODUCT | 2 | 6 / 12 / 0 | finding: A4-002 | - | FC-4 skill-pin durability and KV durability attestation |
| T05 | T-AF2-DEPS-BATCH | T-IGX-051-H | NONPRODUCT | 2 | 0 / 4 / 0 | FRZ: FRZ-DEBT-02, FRZ-DEBT-03; finding: A4-005 | - | Held dependency-bot / uv.lock batch; dependency-ownership gate (A4-005). v2 (Q-RW-09): batch = the dependency-bot PRs open when arch-freeze/rw-train is cut (at drafting: #631, aiohttp 3.13.3 -> 3.14.4) plus the A4-005 ownership declarations; re-listed at the train cut |
| T06 | PT-E05 | T-IGX-044-F | MATERIAL | 2 | 0 / 6 / 0 | finding: A2-038 | - | T-ENF-5 IIAC validator hardening (material governance paths) |
| T07 | PT-D08 | T-IGX-046-D | PRODUCT | 2 | 2 / 2 / 0 | finding: A2-012, A4-041 | - | DB-H product-path docs (2 files counted as product paths) |
| T08 | T-AF2-RW-RUNTIME | T-IGX-051-D | PRODUCT | 3 | 9 / 17 / 0 | EX: EX-D3-01, EX-D3-03, EX-L-07, EX-L-08; PARTIAL: PDC-D3-R005, PDC-D3-R019; FRZ: FRZ-CTR-01, FRZ-CTR-02, FRZ-CTR-06; finding: A4-018 | FRZ-CFG-02, FRZ-CFG-06, FRZ-SEC-04, PDC-D3-R018 | LLM runtime: stream-open errors through breaker/retry, timeout, typed error contract v1, served model, refusal matrix, SDK boundary |
| T09 | PT-WD | T-IGX-043-I | PRODUCT | 3 | 8 / 22 / 0 | finding: A6-003, A6-005, A6-007, A6-008, X-A6-003 | - | W-D ENT-AUDIT-X engine fixes (CHANGED-SINCE default-deny, FRZ attribution, FAST disclosure). v2 (Q-RW-04): the Q-WD-07 option B set is binding (PT-C01 mapped ENT checks before G2); tests/qualification/ent_audit_x/ENTERPRISE_AUDIT_BASELINE.json is NOT in the write set (the accepted baseline is T-AF2-ENT-BASELINE-PREFREEZE, W5); BASELINE:MANIFEST_DIGEST is expected-red with owner PT-WD. W-D changes what the audit checks (gate-meaning); approval route per OD-AF-22 (wave packet approved by the Operator, Approved line by the signing delegate with a Delegation pointer). |
| T10 | PT-FC1 | T-IGX-043-A | PRODUCT | 3 | 30 / 43 / 7 | FRZ: FRZ-BND-05, FRZ-EXE-02, FRZ-PLG-05; finding: A4-009, A4-015, A4-031, A4-037, A4-044, X-A4-024 | FRZ-OWN-03, FRZ-OWN-05, FRZ-PRD-05 | FC-1 execution routing and Nexus construction |
| T11 | PT-FC2-E | T-IGX-043-E | NONPRODUCT | 3 | 0 / 2 / 0 | - | FRZ-PRD-03, FRZ-REL-07, FRZ-REL-09, FRZ-REL-10 | FC-2 gate scripts (capacity adapters, sandbox policy wiring); gate-meaning change, Operator personal |
| T12 | T-AF2-RW-RECOVERY | T-IGX-051-E | PRODUCT | 3 | 12 / 17 / 0 | EX: EX-D1-07, EX-D4-01, EX-D7-01, EX-D7-03, EX-L-03, EX-L-06; PARTIAL: PDC-D7-R015; other: G-20 | - | Background and recovery: drain controller, wait_for_idle, CAS recovery, durable CAS, UNCERTAIN reconciler, capped retry, producer identity |
| T13 | PT-FC3 | T-IGX-043-C | PRODUCT | 4 | 62 / 68 / 0 | FRZ: FRZ-CFG-02, FRZ-CFG-06, FRZ-SEC-04; finding: A4-017, A4-019, A4-026 | A4-032, FRZ-PRD-03, FRZ-REL-07, FRZ-REL-09, FRZ-REL-10 | FC-3 vendor/provider selector fail-closed |
| T14 | T-AF2-RW-SUPPLY | T-IGX-051-B | PRODUCT | 4 | 20 / 31 / 2 | EX: EX-D1-01, EX-D1-02, EX-D1-04, EX-D6-01, EX-D6-02, EX-D6-03, EX-D8-01; PARTIAL: PDC-D1-R017, PDC-D3-R018, PDC-D6-R007, PDC-D6-R010; FRZ: FRZ-SEC-06; other: A-01, F-D3-12, F-D5-12, F-D6-02, F-D6-03, F-D6-04, F-D6-09, F-D6-10, F-D6-11, F-D6-14, F-D8-14 | A4-040, FRZ-EXE-02, FRZ-PLG-05 | Supply chain and security: attestation, revocation, trust hook order, digests, fail-open set, strict validator, secrets, evidence sink, retention |
| T15 | T-AF2-RW-GATES | T-IGX-051-F | PRODUCT | 4 | 8 / 28 / 7 | EX: EX-D1-03, EX-D1-05, EX-D1-06, EX-D2-03, EX-D3-02, EX-D3-04, EX-D4-05, EX-D5-02, EX-D5-03, EX-D7-02, EX-D8-02, EX-L-01, EX-L-04, EX-L-10, EX-L-11, EX-L-13, EX-L-14; PARTIAL: PDC-D1-R015, PDC-D5-R002, PDC-D6-R015; FRZ: FRZ-BND-06, FRZ-DEBT-04, FRZ-OWN-02, FRZ-PLG-01, FRZ-PLG-02, FRZ-PLG-06, FRZ-PLG-07, FRZ-PLG-08, FRZ-PRD-08, FRZ-REG-09; other: KR-03, KR-04, KR-07, KR-17 | FRZ-OWN-03, FRZ-OWN-05, FRZ-PRD-05 | Gates and contracts: promotion gate binding, canonical get_steps, event schema, eval digest, NOT_COMPOSED gates, inventory gates |
| T16 | PT-WE | T-IGX-043-K | PRODUCT | 4 | 0 / 0 / 2 | - | - | W-E (conditional) TOOL-LIFE-X residual hardening PW-3. only if DP-43 sends residuals to a fix and the audit reproduces one |
| T17 | T-AF2-RW-TRUST | T-IGX-051-C | PRODUCT | 5 | 6 / 13 / 0 | EX: EX-D2-01, EX-D2-02, EX-D2-05, EX-D2-07, EX-L-05; PARTIAL: PDC-D2-R004 | - | Trust boundary: trust classes, no untrusted role=system, egress default, orphan tool_call_id, planner ledger, RAG reds |
| T18 | PT-FC3-F | T-IGX-043-F | NONPRODUCT | 5 | 0 / 2 / 0 | FRZ: FRZ-CFG-08; finding: A4-025 | - | CONFIG-X vendor-slug default scan extension (gate-meaning change, Operator personal) |
| T19 | PT-WF | T-IGX-043-J | PRODUCT | 6 | 41 / 51 / 9 | FRZ: FRZ-RPL-04, FRZ-TEN-01, FRZ-TEN-10, FRZ-TYP-01, FRZ-TYP-03; finding: A4-027, A8-020, X-A5-011, X-ESC-E1, X-ESC-E2, X-ESC-E4, X-ESC-E5 | EX-L-09 | W-F tenant G-09 fail-closed, typed seams, B-1 confirmed items. Part JF (A4-018) moves to T-AF2-RW-RUNTIME (dedupe D-05); Part JD stays held and is not scope (D-09) |
| T20 | T-AF2-RW-TENANT | T-IGX-051-A | PRODUCT | 7 | 14 / 22 / 0 | EX: EX-D2-06, EX-D4-04, EX-D5-05, EX-D8-03, EX-D8-04, EX-L-02, EX-L-09; other: F-D2-11, G-12, R-P8-01 | FRZ-CFG-02, FRZ-CFG-06, FRZ-SEC-04, FRZ-TEN-01, FRZ-TEN-10 | Tenant binding: sandbox sessions, object keys, tool context, artifact read, observation tenant, install CAS, MCP tenant, unscoped memory |
| T21 | T-AF2-RW-TYP | T-IGX-051-G | PRODUCT | 7 | 4 / 8 / 0 | FRZ: FRZ-TYP-02, FRZ-TYP-04, FRZ-TYP-05 | FRZ-RPL-04, FRZ-TYP-01, FRZ-TYP-03 | Typing ratchet on frozen contract surfaces (TYP-05; TYP-02/04 per OD-AF-11). v6 (OD-AF-11, OPERATOR_PACKET_3): ratchet gate on the frozen contract surfaces (intergrax/contracts/**: no new Any and no new object annotation above the baseline taken from closed-world/typ_measure.json: 138 Any, 357 object) for TYP-02/04/05, plus the TYP-05 reduction target: the 4 string-keyed dispatch-table candidates measured in intergrax/contracts/** go to 0 (each removed or replaced by a typed mapping); the 4 files are the only product write scope. The rest of the platform is outside the freeze scope, stated in the freeze documents (T-AF2-MANIFESTS) |
| T22 | PT-T01 | T-IGX-043-L | NONPRODUCT | 8 | 0 / 5 / 0 | finding: A4-006 | - | Campaign B tracked negative/adversarial tests NT-01..NT-16 + tenant-propagation emitter repair (A4-006) |

Ticket letters of PT-WE, PT-T01, PT-D08 and PT-E05 are reserved by the EQO-A answer (v2); the tickets are created when their nodes start. T-IGX-051-A..H follow OD-AF-01a.

**Approval route (OD-AF-22 (OPERATOR_PACKET_3, 2026-11-17)).** Applies to every train ticket: T-IGX-051-A..H and the EQO-A T-IGX-043-*, T-IGX-044-F, T-IGX-046-D tickets, including gate-meaning PT-WD and IIAC-material PT-E05. The orchestrator prepares one approval packet per train lane or wave (each ticket, plan file and plan digest); the Operator approves the packet once; the signing delegate then writes each Approved: line with a Delegation: pointer to the packet record. Train PRs merge into arch-freeze/rw-train after independent review (OD-AF-02). The merge of the train into development (FCS) stays the Operator's decision. The the train branch may be cut once the first wave packet (PT-WA) is approved.

## (b) File-scope overlap matrix

Cell = number of shared paths (firm and conditional scope, globs expanded over the tree at HEAD; a new path of one ticket under a glob of the other counts). Blank = disjoint. The shared paths follow the matrix.

| | T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08 | T09 | T10 | T11 | T12 | T13 | T14 | T15 | T16 | T17 | T18 | T19 | T20 | T21 | T22 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **T01** PT-WA | · |  |  |  |  |  |  |  |  | 3 |  |  | 1 |  | 1 |  |  |  | 1 | 1 |  |  |
| **T02** PT-WB |  | · |  |  |  |  |  |  |  |  |  |  |  | 1 |  |  |  |  | 1 |  |  |  |
| **T03** PT-FC2 |  |  | · |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  |  | 2 |  |  |
| **T04** PT-FC4 |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T05** T-AF2-DEPS-BATCH |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T06** PT-E05 |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T07** PT-D08 |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T08** T-AF2-RW-RUNTIME |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T09** PT-WD |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **T10** PT-FC1 | 3 |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  | 1 | 1 |  |  |
| **T11** PT-FC2-E |  |  |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |  |
| **T12** T-AF2-RW-RECOVERY |  |  | 1 |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |  |  |  |
| **T13** PT-FC3 | 1 |  |  |  |  |  |  |  |  |  |  |  | · |  |  |  | 2 |  | 6 | 1 |  |  |
| **T14** T-AF2-RW-SUPPLY |  | 1 |  |  |  |  |  |  |  |  |  |  |  | · |  |  |  |  |  | 1 |  |  |
| **T15** T-AF2-RW-GATES | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |  |
| **T16** PT-WE |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | · |  |  |  |  |  |  |
| **T17** T-AF2-RW-TRUST |  |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |  | · |  |  | 1 |  |  |
| **T18** PT-FC3-F |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | · |  |  |  |  |
| **T19** PT-WF | 1 | 1 |  |  |  |  |  |  |  | 1 |  |  | 6 |  |  |  |  |  | · | 1 |  |  |
| **T20** T-AF2-RW-TENANT | 1 |  | 2 |  |  |  |  |  |  | 1 |  |  | 1 | 1 |  |  | 1 |  | 1 | · |  |  |
| **T21** T-AF2-RW-TYP |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | · |  |
| **T22** PT-T01 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | · |

| Pair | Shared paths |
| --- | --- |
| PT-FC1 × PT-WF | intergrax/runtime/nexus/nexus_loop.py |
| PT-FC1 × T-AF2-RW-TENANT | intergrax/applications/_shared/harness_host_runtime.py |
| PT-FC2 × T-AF2-RW-RECOVERY | intergrax/applications/_shared/queue_worker_wiring.py |
| PT-FC2 × T-AF2-RW-TENANT | intergrax/applications/_shared/memory_provider_admission.py; intergrax/applications/_shared/memory_wiring.py |
| PT-FC3 × PT-WF | applications/dispute_sim_application/host/factory.py; applications/legal_application/host/factory.py; applications/local_workspace_application/host/factory.py; applications/research_application/host/factory.py; intergrax/debug/app.py; intergrax/scaffold/new_application_product.py |
| PT-FC3 × T-AF2-RW-TENANT | intergrax/applications/_shared/environment_wiring.py |
| PT-FC3 × T-AF2-RW-TRUST | tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py (new; under tests/unit/rag/*); tests/unit/rag/vectorstore/test_fc3_vectorstore_selector_fail_closed.py (new; under tests/unit/rag/*) |
| PT-WA × PT-FC1 | intergrax/applications/_shared/harness_host_runtime.py; intergrax/runtime/execution/host_orchestration_environment_spec_builder.py; intergrax/runtime/execution/host_orchestration_loop_init_spec.py |
| PT-WA × PT-FC3 | applications/local_workspace_application/host/factory.py |
| PT-WA × PT-WF | applications/local_workspace_application/host/factory.py |
| PT-WA × T-AF2-RW-GATES | intergrax/applications/_shared/scenario_runtime_baseline.py |
| PT-WA × T-AF2-RW-TENANT | intergrax/applications/_shared/harness_host_runtime.py |
| PT-WB × PT-WF | applications/research_application/host/settings.py |
| PT-WB × T-AF2-RW-SUPPLY | applications/governed_contractor_application/host/package_b_strict_production_serving_command.py |
| PT-WF × T-AF2-RW-TENANT | intergrax/applications/_shared/policy_wiring.py |
| T-AF2-RW-TENANT × T-AF2-RW-SUPPLY | intergrax/agent_distribution/admin_service.py |
| T-AF2-RW-TENANT × T-AF2-RW-TRUST | tests/unit/rag/governance/test_rw_tenant_reindex_hook_scope.py (new; under tests/unit/rag/*) |

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
| 4 | PT-FC3, T-AF2-RW-SUPPLY, T-AF2-RW-GATES, PT-WE | parallel |
| 5 | T-AF2-RW-TRUST, PT-FC3-F | parallel |
| 6 | PT-WF | parallel |
| 7 | T-AF2-RW-TENANT, T-AF2-RW-TYP | parallel |
| 8 | PT-T01 | parallel |

Serialization reasons (every overlap and precedence that sets a ticket's lane):

| Ticket | Lane | After (reason) |
| --- | ---: | --- |
| PT-WA | 0 | - |
| PT-WB | 1 | PT-WA (plan G step 1) |
| PT-FC2 | 2 | PT-WA, PT-WB (plan G steps 1-2) |
| PT-FC4 | 2 | PT-WA, PT-WB (plan G steps 1-2) |
| T-AF2-DEPS-BATCH | 2 | PT-WA, PT-WB (plan G steps 1-2) |
| PT-E05 | 2 | PT-WA, PT-WB (plan G steps 1-2) |
| PT-D08 | 2 | PT-WA, PT-WB (plan G steps 1-2) |
| T-AF2-RW-RUNTIME | 3 | PT-WA, PT-WB (plan G steps 1-2); lane 2 full (max 5 parallel tickets); placed in lane 3 |
| PT-WD | 3 | PT-WA, PT-WB (plan G steps 1-2); lane 2 full (max 5 parallel tickets); placed in lane 3 |
| PT-FC1 | 3 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: Operator merge order B, A, C (FC/OPERATOR_ANSWERS_TRANSCRIPTION.md item 6); after PT-WA: file overlap (3 shared: intergrax/applications/_shared/harness_host_runtime.py, intergrax/runtime/execution/host_orchestration_environment_spec_builder.py, intergrax/runtime/execution/host_orchestration_loop_init_spec.py) |
| PT-FC2-E | 3 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: gate scripts merge after the FC-2 fix (gate would be red before the fix) |
| T-AF2-RW-RECOVERY | 3 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: plan Z: serialized with PT-FC2 on queue wiring; after PT-FC2: file overlap (1 shared: intergrax/applications/_shared/queue_worker_wiring.py) |
| PT-FC3 | 4 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC1: Operator merge order B, A, C; after PT-FC2: PT-FC3 imports wiring_posture.py created by PT-FC2; after PT-WA: file overlap (1 shared: applications/local_workspace_application/host/factory.py) |
| T-AF2-RW-SUPPLY | 4 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: plan Z: serialized with PT-FC2/PT-WB on applications/_shared wiring; after PT-WB: file overlap (1 shared: applications/governed_contractor_application/host/package_b_strict_production_serving_command.py); lane 3 full (max 5 parallel tickets); placed in lane 4 |
| T-AF2-RW-GATES | 4 | PT-WA, PT-WB (plan G steps 1-2); after PT-WA: file overlap (1 shared: intergrax/applications/_shared/scenario_runtime_baseline.py); lane 2 full (max 5 parallel tickets); placed in lane 4 |
| PT-WE | 4 | PT-WA, PT-WB (plan G steps 1-2); lane 2 full (max 5 parallel tickets); placed in lane 4 |
| T-AF2-RW-TRUST | 5 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC3: file overlap (2 shared: tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py (new; under tests/unit/rag/*), tests/unit/rag/vectorstore/test_fc3_vectorstore_selector_fail_closed.py (new; under tests/unit/rag/*)) |
| PT-FC3-F | 5 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC3: CONFIG-X extension merges after FC-3 (gate would be red before the fix) |
| PT-WF | 6 | PT-WA, PT-WB (plan G steps 1-2); after PT-FC2: PT-WF imports wiring_posture.py created by PT-FC2; after PT-FC3-F: proposed order W-A/W-B, B, A, C, F, then J (WF plan 7.3, Q-WF-12); after PT-WB: file overlap (1 shared: applications/research_application/host/settings.py); after PT-FC3: file overlap (6 shared: applications/dispute_sim_application/host/factory.py, applications/legal_application/host/factory.py, applications/local_workspace_application/host/factory.py ...); after PT-WA: file overlap (1 shared: applications/local_workspace_application/host/factory.py); after PT-FC1: file overlap (1 shared: intergrax/runtime/nexus/nexus_loop.py) |
| T-AF2-RW-TENANT | 7 | PT-WA, PT-WB (plan G steps 1-2); after PT-WF: plan Z: RW-TENANT serialized with PT-FC1/PT-WF on tenant wiring; after PT-FC1: plan Z: harness_host_runtime.py; after PT-WF: file overlap (1 shared: intergrax/applications/_shared/policy_wiring.py); after PT-FC3: file overlap (1 shared: intergrax/applications/_shared/environment_wiring.py); after T-AF2-RW-SUPPLY: file overlap (1 shared: intergrax/agent_distribution/admin_service.py); after T-AF2-RW-TRUST: file overlap (1 shared: tests/unit/rag/governance/test_rw_tenant_reindex_hook_scope.py (new; under tests/unit/rag/*)); after PT-WA: file overlap (1 shared: intergrax/applications/_shared/harness_host_runtime.py); after PT-FC1: file overlap (1 shared: intergrax/applications/_shared/harness_host_runtime.py); after PT-FC2: file overlap (2 shared: intergrax/applications/_shared/memory_provider_admission.py, intergrax/applications/_shared/memory_wiring.py) |
| T-AF2-RW-TYP | 7 | PT-WA, PT-WB (plan G steps 1-2); after PT-WF: RW-TYP ratchet baseline is taken after the typed seams of W-F JB; after T-AF2-RW-SUPPLY: the ratchet baseline is taken after the last train ticket that edits intergrax/contracts/** (RW-SUPPLY edits agent_run.py) |
| PT-T01 | 8 | PT-WA, PT-WB (plan G steps 1-2); after T-AF2-RW-TENANT: NT-01..16 must be green when merged (no expected-red test without a ruling); after PT-WF: NT-01..16 cover W-F defects; after PT-WD: NT-08 covers A6-007 |

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
- T-AF2-RW-TENANT -> PT-T01: NT-01..16 must be green when merged (no expected-red test without a ruling)
- PT-WF -> PT-T01: NT-01..16 cover W-F defects
- PT-WD -> PT-T01: NT-08 covers A6-007
- PT-WF -> T-AF2-RW-TYP: RW-TYP ratchet baseline is taken after the typed seams of W-F JB
- T-AF2-RW-SUPPLY -> T-AF2-RW-TYP: the ratchet baseline is taken after the last train ticket that edits intergrax/contracts/** (RW-SUPPLY edits agent_run.py)

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
| NT-ROW-019 | A4-044 | CODE_FIX_29, CAMPAIGN_RW | PT-FC1 | B | `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-01..04, 12 | production NexusLoop + empty agent_id + non-production agent is rejected; production-eligible agent without owner/runbook rejected; production agent routes | B-1 R03 v1 CONFIRMED at router level at B; files unchanged B->H |
| NT-ROW-020 | A4-009 | CODE_FIX_29 | PT-FC1 | B | `tests/unit/runtime/nexus/test_fc1_agent_router_posture_and_fallback.py` (NEW_PARENT_EXISTS) | NT-FC1-07..11 | STRICT + no capability + no best match raises a typed error; LAB opt-in keeps fallback_used | AgentRouter falls back silently at HEAD (finding A4-009) |
| NT-ROW-021 | A4-037 | CODE_FIX_29, CAMPAIGN_RW | PT-FC1 | B | `tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-13..15, 20..23, 26 | STRICT NexusLoop without a human decision store raises; reject and escalate verdicts are persisted durably | B-1 R03 v5 CONFIRMED at B |
| NT-ROW-022 | A4-017 | CODE_FIX_29 | PT-FC3 | B | `tests/unit/model_inference/test_fc3_vision_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-VIS1..11 | unconfigured Triton/HF raises a typed unconfigured error; zero detections return an empty list, never object.stub | the stub detection is fabricated at HEAD (finding A4-017) |
| NT-ROW-023 | A4-019 | CODE_FIX_29 | PT-FC3 | B | `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-E1..6, V1..4, N1..13 | unset embedding provider, vector store and notification backend raise (LAB opt-in excepted) | silent defaults to ollama, in-memory and log at HEAD (finding A4-019) |
| NT-ROW-024 | A4-026 | CODE_FIX_29, CAMPAIGN_RW | PT-FC3 | B | `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-INT1..12 | unknown verification mode raises in STRICT; Slack/Teams mode requires a secret; NONE only with explicit acknowledgement | B-1 R03 v3 CONFIRMED at B |
| NT-ROW-025 | A4-002 | CODE_FIX_29 | PT-FC4 | B | `tests/unit/skills/test_skill_execution_pinning_durability_guard.py` (NEW_PARENT_EXISTS) | FC-4 NT-1..14 | in-memory KV or doc store + production_mode raises; KV without durability attestation raises; SQLite passes | is_durable=True is reported unconditionally at HEAD (finding A4-002) |
| NT-ROW-026 | A4-035 | CODE_FIX_29 | PT-FC2 | B | `tests/unit/runtime/capacity/test_fc2_capacity_fail_closed.py` (NEW_PARENT_EXISTS) | CAP-1..7 | PRODUCT + adapters enabled + Kubernetes URL unset is UNQUALIFIED and builds no adapters; in_memory never yields probe_passed | in-memory scaler substituted at HEAD (finding A4-035) |
| NT-ROW-027 | A4-036 | CODE_FIX_29 | PT-FC2 | B | `tests/unit/applications/test_fc2_queue_worker_fail_closed.py` (NEW_PARENT_EXISTS) | FC-2 queue tests | STRICT + enabled without a real broker raises; KAFKA/RABBITMQ are not rewritten to CELERY | rewrite to eager in-process Celery at HEAD (finding A4-036) |
| NT-ROW-028 | A4-030 | CODE_FIX_29 | PT-FC2 | B | `tests/unit/applications/test_fc2_memory_org_session_admission.py` (NEW_PARENT_EXISTS) | FC-2 memory tests | PRODUCT + org memory + Mongo raises; PRODUCT + no backend + persistent memory raises | in-memory org/session stores built at HEAD (finding A4-030) |
| NT-ROW-029 | A4-032 | CODE_FIX_29 | PT-FC2 | B | `tests/unit/applications/test_fc2_otlp_composition_fail_closed.py` (NEW_PARENT_EXISTS) | FC-2 OTLP/sandbox tests; NT-FC3-GR1..5 (part c) | enabled OTLP + empty endpoint raises in STRICT; STRICT + side-effect tools + no sandbox manager raises; guardrail downgrade raises unless allowed | silent degraded defaults at HEAD (finding A4-032) |
| NT-ROW-030 | FRZ-BND-04 | FRZ_CODE_FIX_35 | PT-WA | S | `tests/unit/architecture/test_wa_integrations_shared_leakage.py` (NEW_PARENT_EXISTS) | NT-BND-04-1 | as A2-009 | as A2-009 (8 files) |
| NT-ROW-031 | FRZ-BND-05 | FRZ_CODE_FIX_35 | PT-FC1 | S | `tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py` (NEW_PARENT_EXISTS) | NT-BND-05-1 (+NT-FC1-05/06/12/24); NT-BND-05-2 in tests/unit/architecture/test_rw_bnd05_provider_switch_gate.py (PT-WB) | a Tier-2 agent that constructs RuntimeContext outside the sanctioned composition root fails the construction gate; an enum/string provider switch in Tier-1/Tier-3 fails the provider-switch gate | B-1 R03 v2 CONFIRMED one STRICT-reachable relaxed build_context site at B; the switch half is C (B-2 S01 heuristic candidates, ruling DP-18) |
| NT-ROW-032 | FRZ-BND-06 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_transport_semantics.py` (NEW_PARENT_EXISTS) | NT-BND-06-1 | a hosting, queue, MCP or HTTP transport adapter that constructs or interprets a RuntimeEvent/envelope fails the transport-semantics gate | red only if the step-0 inventory at HEAD finds an adapter that owns runtime semantics; otherwise the row becomes evidence-only under PT-G02 |
| NT-ROW-033 | FRZ-OWN-02 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | G | `tests/unit/architecture/test_rw_gates_contract_census.py` (NEW_PARENT_EXISTS) | NT-OWN-02-1 | a second contract with the same field set and responsibility as a canonical one fails the census unless it has a ratification entry | the duplicate task_artifacts contracts and the three ArtifactRef types exist at HEAD without ratification (PDC-D8-R017/R018) |
| NT-ROW-034 | FRZ-OWN-03 | FRZ_CODE_FIX_35 | PT-WB | B | `tests/unit/applications/test_wb_llm_resolver_canonical.py` (NEW_PARENT_EXISTS) | NT-WB-01..03; NT-OWN-03-1/2 | one host LLM resolver (Tier-1 canonical, Tier-3 delegates); GR-12 product-host composition passes under STRICT | two resolvers with divergent routing at HEAD (A4-016); NT-WB-01/02 are red at HEAD per the W-B plan |
| NT-ROW-035 | FRZ-OWN-05 | FRZ_CODE_FIX_35 | PT-WB | S | `tests/unit/architecture/test_rw_own05_duplicate_mechanisms.py` (NEW_PARENT_EXISTS) | NT-OWN-05-1 | for each of the five named responsibilities (plugin discovery, context-window tables, tool-input guard, evaluator contract, host LLM resolver) exactly one mechanism exists or the duplicate is ratified | two host LLM resolvers exist at HEAD; the other four are C until the step-0 scan |
| NT-ROW-036 | FRZ-CTR-01 | FRZ_CODE_FIX_35 | T-AF2-RW-RUNTIME | C | `tests/unit/llm_adapters/test_rw_runtime_capability_flag_parity.py` (NEW_PARENT_EXISTS) | NT-CTR-01-1 | parametrized over every shipped adapter: a declared streaming/tool-streaming capability without the method fails, and an implemented method with the flag false fails | PDC records misreporting flags (D3-R022) and Ollama without stream_with_tools (D3-R006); confirmed per adapter at step 0 |
| NT-ROW-037 | FRZ-CTR-02 | FRZ_CODE_FIX_35 | T-AF2-RW-RUNTIME | S | `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py` (NEW_PARENT_EXISTS) | NT-CTR-02-1 | as A4-018 | as A4-018 |
| NT-ROW-038 | FRZ-CTR-04 | FRZ_CODE_FIX_35 | PT-Q12 | N | `tests/qualification/pdc_x/d4/test_d4_public_api_manifest.py` (EXISTS) | NT-CTR-04-1 (replay) | no new test: the public-API manifest test is replayed at M3 | not applicable (RESOLVED_AT_BASE) |
| NT-ROW-039 | FRZ-CTR-06 | FRZ_CODE_FIX_35 | T-AF2-RW-RUNTIME | S | `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py` (NEW_PARENT_EXISTS) | NT-CTR-06-1 | as A4-018 | as A4-018 |
| NT-ROW-040 | FRZ-TYP-05 | FRZ_CODE_FIX_35 | T-AF2-RW-TYP | S | `tests/unit/architecture/test_rw_typ05_string_dispatch_reduction.py` (NEW_PARENT_EXISTS) | NT-TYP-05-2 (reduction, red-first); NT-TYP-05-1 in tests/unit/architecture/test_rw_typ_ratchet.py (ratchet, seeded new Any) | AST over intergrax/contracts/** finds 0 string-keyed dispatch tables (target of OD-AF-11); and the ratchet fails on a seeded new Any or object annotation in a temporary copy of a real frozen contract file (sensitivity, Q-RW-07 conditions 1-3) | 4 dispatch-table candidates in intergrax/contracts/** at HEAD (closed-world typ_measure.json: capability_metering/usage_event.py, governed_execution_result.py, runtime_execution_context.py, sanctioned_external_work_operation.py), above the target 0; this meets audit condition (4) of Q-RW-07 for TYP-05 |
| NT-ROW-041 | FRZ-PLG-01 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_plugin_contracts.py` (NEW_PARENT_EXISTS) | NT-PLG-01-1 | an evaluator registered outside the single evaluator contract fails registration | PDC-D5-R005 PARTIAL (two evaluator contracts); confirm at step 0 |
| NT-ROW-042 | FRZ-PLG-02 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_plugin_contracts.py` (NEW_PARENT_EXISTS) | NT-PLG-02-1 (new narrow test), NT-PLG-02-2 | an external chunker registered through the public plugin entry point is used without a core patch; no provider-plugin registration path exists outside LLMAdapterRegistry | the existing red plug_03 flow does NOT count; the new narrow test must be shown red at HEAD for the defect reason |
| NT-ROW-043 | FRZ-PLG-05 | FRZ_CODE_FIX_35 | PT-FC1 | B | `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-01..04; NT-PLG-05-1 in tests/unit/core/plugins/test_rw_supply_installed_file_digest.py | as A4-044; an unsigned or digest-mismatched plugin entry point is refused at admission | A4-044 half: B-1 R03 v1 CONFIRMED at B (behavioural red). Plugin half: entry-point provenance admission exists at HEAD (PluginProvenanceAdmission, F-58), so 'unsigned plugin refused' is not a red; the remaining plugin red is the installed-file digest mismatch without re-hash (EX-D6-03, NT-RW-SUP-06) and the defense-loader gap of FRZ-SEC-06 (basis C) |
| NT-ROW-044 | FRZ-PLG-06 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | G | `tests/unit/architecture/test_rw_gates_plugin_contracts.py` (NEW_PARENT_EXISTS) | NT-PLG-06-1 | every tool, skill, MCP and integration package is in the package inventory; a missing one fails | GR-11 does not cover these packages at HEAD; the new inventory gate reports them |
| NT-ROW-045 | FRZ-PLG-07 | FRZ_CODE_FIX_35 | T-AF2-RW-GATES | C | `tests/unit/architecture/test_rw_gates_plugin_contracts.py` (NEW_PARENT_EXISTS) | NT-PLG-07-1 | a package that adds a capability beyond its manifest is refused | confirm at step 0 with the adversarial fixture against HEAD |
| NT-ROW-046 | FRZ-EXE-02 | FRZ_CODE_FIX_35 | PT-FC1 | B | `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-01..04; NT-EXE-02-1 = NT-WA-13 | as A4-044; a Tier-2 agent constructed outside the registry fails the registry-bypass rule | B-1 R03 v1 and R01b at B |
| NT-ROW-047 | FRZ-CFG-02 | FRZ_CODE_FIX_35 | PT-FC3 | B | `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-E*, V*, N*, VIS*, INT* | as A4-017, A4-019, A4-026 | as those findings |
| NT-ROW-048 | FRZ-CFG-06 | FRZ_CODE_FIX_35 | PT-FC3 | B | `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-E*, VIS*; NT-CFG-06-1 = NT-RW-RUN-07 | no alternate factory, fallback or direct SDK path reaches an unconfigured provider | as A4-017, A4-018, A4-019 |
| NT-ROW-049 | FRZ-CFG-08 | FRZ_CODE_FIX_35 | PT-FC3-F | G | `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py` (NEW_PARENT_EXISTS) | NT-FC3-CX1..12; NT-CFG-08-1 | a vendor-slug default or an ambient os.environ read outside the inventoried surfaces turns the CONFIG-X gate red | over HEAD the extended scan reports the vendor-slug defaults FC-3 removes and the uninventoried reads B-2 S01 counted; it turns green only after PT-FC3 (hence F after C) |
| NT-ROW-050 | FRZ-SEC-04 | FRZ_CODE_FIX_35 | PT-FC3 | B | `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py` (NEW_PARENT_EXISTS) | NT-FC3-INT1..12; NT-SEC-04-1 | a STRICT or PRODUCT profile with an unset or invalid verification mode refuses to start | B-1 R03 v3 at B |
| NT-ROW-051 | FRZ-REL-07 | FRZ_CODE_FIX_35 | PT-FC2 | B | `tests/unit/runtime/capacity/test_fc2_capacity_fail_closed.py` (NEW_PARENT_EXISTS) | CAP-1..7; FC-2 queue, OTLP, sandbox tests | no silent backend substitution in STRICT/PRODUCT | as A4-032, A4-035, A4-036 |
| NT-ROW-052 | FRZ-REL-09 | FRZ_CODE_FIX_35 | PT-FC2 | B | `tests/unit/applications/test_fc2_queue_worker_fail_closed.py` (NEW_PARENT_EXISTS) | FC-2 queue tests | as A4-036 | as A4-036 |
| NT-ROW-053 | FRZ-REL-10 | FRZ_CODE_FIX_35 | PT-FC2 | B | `tests/unit/runtime/capacity/test_fc2_capacity_fail_closed.py` (NEW_PARENT_EXISTS) | CAP-1..7 | as A4-035 | as A4-035 |
| NT-ROW-054 | FRZ-PRD-03 | FRZ_CODE_FIX_35 | PT-FC2 | B | `tests/unit/applications/test_fc2_queue_worker_fail_closed.py` (NEW_PARENT_EXISTS) | FC-2 tests; NT-FC3-INT*; NT-PRD-03-1 | an unsupported production configuration in STRICT raises instead of substituting an in-memory or in-process backend | as A4-026, A4-035, A4-036 |
| NT-ROW-055 | FRZ-PRD-05 | FRZ_CODE_FIX_35 | PT-WB | S | `tests/unit/applications/test_wb_serving_bind_defaults.py` (NEW_PARENT_EXISTS) | NT-WB-14..17, 25, 26; NT-PRD-05-1 | no Tier-3 entrypoint starts serving outside the hosting lifecycle engine, except the declared set | 7 application files call uvicorn.run at B (B-2 S01), one a strict production serving command |
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
| NT-ROW-139 | A4-007 | CAMPAIGN_RW | PT-WB | B | `tests/unit/applications/test_wb_import_allowlist.py` (NEW_PARENT_EXISTS) | NT-WB-18..24 (+ E9 scoreboard test) | the three importlib seams refuse a module outside the prefix allowlist before any import; registered names still resolve | B-1 R02 b1, b2, b4 CONFIRMED at B |
| NT-ROW-140 | A4-015 | CAMPAIGN_RW | PT-FC1 | B | `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-05, 06, 12, 24 | STRICT host: Tier-2 build_context yields production_mode True; NexusLoop without explicit posture raises | B-1 R03 v2 at B |
| NT-ROW-141 | A4-027 | CAMPAIGN_RW | PT-WF | B | `tests/unit/applications/test_wf_serving_tenant_fail_closed.py` (NEW_PARENT_EXISTS) | NT-WF-A07..A14, A16 | as FRZ-TEN-10 | B-1 R03 v7 at B |
| NT-ROW-142 | A4-031 | CAMPAIGN_RW | PT-FC1 | B | `tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py` (NEW_PARENT_EXISTS) | NT-FC1-25 (part C2) | a STRICT host composition hands UAEP both a context engine and an LLM adapter, or refuses | B-1 R03 v6 CONFIRMED at B (composition level) |
| NT-ROW-143 | X-A4-024 | CAMPAIGN_RW | PT-FC1 | B | `tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py` (NEW_PARENT_EXISTS) | NT-FC1-16..19 | a STRICT profile with all memory kinds off does not open the task-memory store when INTERGRAX_TASK_MEMORY_DB is set | B-1 R03 v4 at B |
| NT-ROW-144 | X-ESC-E1 | CAMPAIGN_RW | PT-WF | B | `tests/unit/runtime/execution/test_wf_child_admission_narrowing.py` (NEW_PARENT_EXISTS) | NT-WF-G1-01..03 | child admission refuses scopes outside the parent even with an injected wide policy | B-1 R03 v8 at B |
| NT-ROW-145 | A8-020 | CAMPAIGN_RW | PT-WF | B | `tests/unit/compat/langchain/test_wf_trusted_scope_conversion.py` (NEW_PARENT_EXISTS) | NT-WF-G3-01..04 | compat conversion takes tenant from the trusted scope, never from foreign document metadata | B-1 R04 nt11a/nt11b at B |
| NT-ROW-146 | X-A4-016 | CAMPAIGN_RW | PT-WB | B | `tests/unit/applications/test_wb_llm_resolver_canonical.py` (NEW_PARENT_EXISTS) | NT-WB-02, 03, 27, 28 | a routing record is consumable only by the adapter and tenant that produced it | B-1 R04 nt12a/nt12b at B |
| NT-ROW-147 | X-A8-003 | CAMPAIGN_RW | PT-WB | B | `tests/unit/contracts/test_wb_profile_projection.py` (NEW_PARENT_EXISTS) | NT-WB-10..13, 29 | each configured Tier-3-only value is carried by the projection or the projection refuses | B-1 R04 nt13a at B (criterion open: Q-WB-03) |
| NT-ROW-148 | X-ESC-E4 | CAMPAIGN_RW | PT-WF | B | `tests/unit/debug/test_wf_experiment_tenant_scope.py` (NEW_PARENT_EXISTS) | NT-WF-G4-01..04 | tenant B cannot list or show an experiment registered by tenant A | B-1 R04 nt15b at B |
| NT-ROW-149 | X-ESC-E5 | CAMPAIGN_RW | PT-WF | B | `tests/unit/applications/test_wf_debug_router_guard_idp.py` (NEW_PARENT_EXISTS) | NT-WF-G5-01..02 (5b held: G5-03..06) | the harness guard denies an anonymous caller when an identity provider is configured | B-1 R04 nt15d at B |
| NT-ROW-150 | X-A5-011 | CAMPAIGN_RW | PT-WF | B | `tests/unit/applications/test_wf_tenant_evaluation_fail_closed.py` (NEW_PARENT_EXISTS) | NT-WF-A01..A04 | as FRZ-TEN-01 | B-1 R08 g09a at B |
| NT-ROW-151 | X-ESC-E2 | CAMPAIGN_RW | PT-WF | B | `tests/unit/runtime/execution/test_wf_decision_finalization_strict.py` (NEW_PARENT_EXISTS) | NT-WF-G2-01..03 | a shipped STRICT host binds a durable decision finalization store or refuses | B-1 R12 nt09b at B |
| NT-ROW-152 | X-A6-003 | CAMPAIGN_RW | PT-WD | B | `tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py` (NEW_PARENT_EXISTS) | WD B01, B02 | as A6-003 | B-2 R25c at B |
| NT-ROW-153 | A4-040 | CAMPAIGN_RW | PT-WB | S | `tests/unit/applications/test_wb_serving_bind_defaults.py` (NEW_PARENT_EXISTS) | NT-WB-14..17, 25, 26 | as FRZ-PRD-05 | B-2 S01/S03 at B |
| NT-ROW-154 | A4-020 | CAMPAIGN_RW | PT-WB | C | `tests/unit/architecture/test_rw_bnd05_provider_switch_gate.py` (NEW_PARENT_EXISTS) | NT-BND-05-2 | a provider-selection switch outside IntegrationProfile in Tier-1/Tier-3 fails, except files on the DP-18 named list | red only for a file ruled a bypass (B-2: heuristic candidates) |
| NT-ROW-155 | A4-025 | CAMPAIGN_RW | PT-FC3-F | G | `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py` (NEW_PARENT_EXISTS) | NT-CFG-08-1 | an ambient environment read outside the inventoried CONFIG-X surfaces turns the gate red | B-2 S01: 22 runtime files / 27 sites read the environment, 9 of 14 named by no CONFIG-X gate |
| NT-ROW-156 | KR-18 | KNOWN_RED_RW | PT-WA | S | `tests/unit/architecture/test_wa_kr18_contract_root_no_shared_import.py` (NEW_PARENT_EXISTS) | NT-WA-KR18 | AST over intergrax/applications/contracts/** (function-level imports included) finds no import of intergrax.applications._shared | root.py imports strict_host_dependency_concurrency_admission from applications._shared at HEAD (FACT_CHECK F-44). The known-red node of KR-18 does not count |
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
| NT-ROW-167 | FRZ-TYP-02 | FRZ_CODE_FIX_35 | T-AF2-RW-TYP | M | `tests/unit/architecture/test_rw_typ_ratchet.py` (NEW_PARENT_EXISTS) | NT-TYP-02-1 | a seeded new Any or object annotation in a temporary copy of a real frozen contract file under intergrax/contracts/** fails the ratchet; the counts may only fall | none by construction (a ratchet passes on its baseline: 138 Any, 357 object at HEAD per typ_measure.json); prevention control under the Q-RW-07 ruling |
| NT-ROW-168 | FRZ-TYP-04 | FRZ_CODE_FIX_35 | T-AF2-RW-TYP | M | `tests/unit/architecture/test_rw_typ_ratchet.py` (NEW_PARENT_EXISTS) | NT-TYP-04-1 | a seeded new Any or object annotation in a temporary copy of a real frozen contract file under intergrax/contracts/** fails the ratchet; the counts may only fall | none by construction (a ratchet passes on its baseline: 138 Any, 357 object at HEAD per typ_measure.json); prevention control under the Q-RW-07 ruling |

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
| PT-WA | `tests/unit/runtime/architecture/test_faudit_remediation.py`; `tests/unit/architecture/test_ebh_4_r1_r3_b6_typing_gate.py`; `tests/unit/applications/test_orchestration_wiring.py`; `tests/unit/applications/test_harness_host_runtime_llm.py`; `tests/qualification/harness_01/nexus_import_inventory.py` | FC/WA/PLAN_T-IGX-043-G.md section 11 steps S1-S7; .tmp/session/arch-freeze/closed-world/bnd_tests_importing_t3_targets.txt (S7 collect-only, authorization WA-A12) |
| PT-WB | `tests/unit/llm_adapters/routing/test_evaluating_adapter.py`; `tests/acceptance/llm_routing/test_concurrent_run_routing_isolation.py` | FC/WB/REGRESSION_NODES_VERIFIED.txt (209 node ids) |
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
| PT-FC3 | `tests/unit/model_inference/test_remote_vision_adapters.py`; `tests/unit/rag/vectorstore/test_integration_vectorstore_bootstrap.py`; `tests/qualification/config_x` | FC/FC-3/regression_nodes.txt |
| T-AF2-RW-SUPPLY | `tests/unit/agent_distribution`; `tests/unit/core/plugins`; `tests/unit/runtime/security`; `tests/qualification/pdc_x/xd`; `scripts/maintenance/check_immutable_security_audit_trail.py`; `scripts/maintenance/check_sandbox_policy_wiring.py` | agent distribution, plugins, runtime security, PDC D1/D6 qualification, audit-trail gate |
| T-AF2-RW-GATES | `tests/unit/scripts`; `tests/unit/applications/test_scenario_runtime_baseline.py`; `tests/unit/llm_adapters/test_fixture_strict_adapter_parity.py`; `tests/qualification/plug_03`; `tests/qualification/trace_x` | gate scripts, scenario baseline, strict adapter parity, plug_03, TRACE-X |
| PT-WE | `tests/unit/runtime/nexus/tools` | TOOL-LIFE-X C-1..C-3 nodes (PT-V02) |
| T-AF2-RW-TRUST | `tests/unit/context`; `tests/unit/llm`; `tests/unit/rag`; `tests/integration/rag`; `tests/qualification/pdc_x/d2_repair` | context, llm, rag (extras profile for integration) |
| PT-FC3-F | `tests/qualification/config_x` | FC/FC-3/regression_nodes_F.txt |
| PT-WF | `tests/qualification/tenant_x`; `tests/unit/compat/langchain/test_documents.py`; `tests/integration/decision_system/test_concurrent_finalization.py` | FC/WF/regression_nodes.txt |
| T-AF2-RW-TENANT | `tests/qualification/tenant_x`; `tests/unit/runtime/sandbox`; `tests/unit/tools/providers`; `tests/unit/agent_distribution`; `tests/qualification/mcp_server_x`; `tests/unit/applications/test_mcp_identity_mode_fail_closed.py`; `tests/unit/rag/governance` | tenant suites, sandbox, storage tools, agent distribution, MCP identity |
| T-AF2-RW-TYP | `tests/unit/architecture/test_ebh_4_r1_r3_b6_typing_gate.py` | typing gates |
| PT-T01 | `tests/qualification/tenant_x`; `tests/qualification/trace_x` | NT-01..NT-16 at the train head (red-first at B recorded by PT-B01) |

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
| D-03 | three importlib seams accept any module path | B-1 A4-007; W-A Part F (Q-WA-08); W-B Part E (DP-21) | PT-WB | W-B Part E creates the single allowlist module import_allowlist.py and covers all three seams plus wiring.py and mcp_import_guard.py (same loader class); W-A Part F is separable. Both plans write agent_resolution.py, migration_wiring.py and scoreboard.py, so one owner removes the file overlap. |
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
| A4-007 | PT-WB | RW | - | W-B Part E (DP-21); W-A Part F dropped (D-03) |
| A4-009 | PT-FC1 | RW | - | - |
| A4-011 | PT-G01 | NON-RW: W2 gate/seam-ledger node | - | - |
| A4-015 | PT-FC1 | RW | - | - |
| A4-017 | PT-FC3 | RW | - | - |
| A4-018 | T-AF2-RW-RUNTIME | RW | - | plan E WP12-Q02; PT-WF Part JF moved here |
| A4-019 | PT-FC3 | RW | - | - |
| A4-020 | PT-WB | RW-CLASSIFY-FIRST (Part C; code only for a file ruled a bypass) | - | - |
| A4-025 | PT-FC3-F | RW-GATE-SCOPE (CONFIG-X ambient read inventory; product only if a read is ruled fail-open) | - | - |
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
| KR-18 | PT-WA | RW (known-red FIX row, Q-RW-10) | - | environment_profile/root.py imports applications._shared |
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
negative-test rows: 168; items needing a row: 166; basis counts: {'B': 92, 'C': 19, 'G': 9, 'M': 16, 'N': 3, 'S': 29}
phase-3 rows with a red-first basis (B/S/G): 18 of 18; with >= 1 regression path existing at head_sha: 18 of 18
red-first rows (B/S/G): 130; prevention controls (M, reported apart, Q-RW-07 conditions attached): 16; conditional (C): 19; no code change (N): 3
audit-flag bindings (6 conflicts): 6 of 6 recorded with owner and auditing node
scope entries: 513; not verified at head_sha: 0
expected-red (registered, from expected_red_A01.json): 6 entries; planned, to be registered: 7 (all with owner and registration moment: True)
known-red FIX rows assigned (Q-RW-10): 6; with owner and negative-test row: 6
regression paths: 76; missing at head_sha: 0; tickets without a regression scope: 0
tickets: 22; lanes: 9; overlapping pairs: 17 (all serialized: True)
fact checks: 74; not verified: none

COVERAGE GAPS: 0
```

The check is mechanical: it proves exactly-once ownership, owner validity, row completeness, scope verification and lane rules over the data; it does not judge whether a negative test is the right one (that is T-AF2-RW-PLAN-AUDIT).

## (h) Open questions for the orchestrator

- **Q-RW-16** (Operator) Remaining Operator items after OPERATOR_PACKET_3: (b) the A4-040 scope ruling (Q4), which lands in W-B Part G or RW-SUPPLY on package_b_strict_production_serving_command.py; (d) authorization of the W-B Part G PROD-Q node runs. Items (a) and (c) are settled by OD-AF-22.
- **Q-RW-17** (orchestrator) Text amendments the EQO-A answer assigns to the orchestrator: the W-D plan (option B binding, baseline JSON removed) and the MEP-AF2 section 0(e) wording. KR-19 has no enumerated node ids; the PT-FC2 implementer enumerates them at step 0 for the known-red register PR.
- **Q-RW-20** (orchestrator) New in v6: the W-A plan (FC/WA/PLAN_T-IGX-043-G.md) was amended by EQO-A on 2026-11-17 06:26, after the scope extraction of 2026-11-16 (work/fc_writesets.json). The PT-WA scope and its source line numbers refer to the earlier text (Part F already dropped here by drop_paths). Re-extract the PT-WA write set from the amended plan and re-run the overlap matrix before the PT-WA wave packet is approved? The other nine EQO-A plans are unchanged since the extraction.

## Limits of this record

- No test or probe was run: none was strictly needed, because every red-first condition is stated for the implementer and checked by the audit; no lock was taken.
- Write sets of the EQO-A tickets are extracted mechanically from their plans; the extraction excludes the plan line ranges listed in `rw_plan_data.py` (`FC_EXTRACT_EXCLUDE`, `FC_EXTRACT_DROP`). A path a plan names only in prose outside its write-set section is not in scope.
- Conditional parts (W-A Part E, W-B Parts E-conditional and G, W-D option B, W-F JG-5b, RW-TYP contracts, RW-SUPPLY security_runtime_bridge) are in the overlap matrix; they are binding only if their condition is decided.
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

## Appendix A1. Binding scope per ticket

### PT-WA (T-IGX-043-G)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/contracts/delegation_budget_policy.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:160 |
| `intergrax/contracts/host_orchestration_wiring_port.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:161 |
| `intergrax/contracts/host_orchestration_environment.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:162 |
| `intergrax/contracts/rag_retrieval_source_scope.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:163 |
| `intergrax/contracts/workspace_tool_ids.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:164 |
| `intergrax/applications/_shared/application_host_orchestration_wiring.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:165 |
| `intergrax/applications/_shared/host_orchestration_environment_resolution.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:166 |
| `intergrax/runtime/execution/application_environment_context_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:171 |
| `intergrax/runtime/execution/application_graph_spec_to_plan.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:172 |
| `intergrax/runtime/execution/application_host_orchestration_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:173 |
| `intergrax/runtime/execution/environment_host_task_execution.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:174 |
| `intergrax/runtime/execution/harness_host_internal_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:175 |
| `intergrax/runtime/execution/host_orchestration_application_wiring_applier.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:176 |
| `intergrax/runtime/execution/host_orchestration_assembly_validation.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:177 |
| `intergrax/runtime/execution/host_orchestration_environment_spec_builder.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:178 |
| `intergrax/runtime/execution/host_orchestration_planner_classifier_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:179 |
| `intergrax/runtime/execution/host_orchestration_loop_init_spec.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:180 |
| `intergrax/applications/_shared/host_orchestration_backend_spec_builder.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:181 |
| `intergrax/applications/_shared/harness_host_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:182 |
| `intergrax/applications/_shared/harness_host_runtime.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:183 |
| `intergrax/applications/_shared/scenario_runtime_baseline.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:184 |
| `intergrax/applications/_shared/orchestration_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:185 |
| `intergrax/applications/_shared/context_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:186 |
| `intergrax/applications/_shared/delegation_budget_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:187 |
| `intergrax/applications/_shared/observability_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:188 |
| `intergrax/applications/contracts/graph_spec.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:189 |
| `intergrax/tools/providers/rag/source_scope_transport.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:190 |
| `intergrax/tools/providers/workspace/tool_ids.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:191 |
| `agents/local_search/steps/search_job.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:192 |
| `agents/tool_selection_qualifier/tool_selection.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:193 |
| `agents/tool_selection_qualifier/steps/tool_selection_job.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:194 |
| `agents/web_search_qualifier/steps/web_search_job.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:195 |
| `agents/web_search_qualifier/web_search_qualifier_agent.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:196 |
| `applications/local_workspace_application/host/agent_factories.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:197 |
| `intergrax/runtime/execution/host_orchestration_wiring_bundle.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:202 |
| `intergrax/runtime/execution/scenario_host_diagnostic_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:203 |
| `intergrax/runtime/architecture/obs_diag_x3_ast_gates.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:204 |
| `agents/web_search_qualifier/search_provider_resolver.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:205 |
| `applications/local_workspace_application/workspaces/document_store_factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:210 |
| `applications/local_workspace_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:211 |
| `testing_support/architecture/obs_diag_x3_ast_gates.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:225 |
| `tests/system/functional_diagnostics_q3/search_provider_resolver.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:226 |
| `tests/unit/runtime/architecture/test_wa_boundary_inversion_gate.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:227 |
| `tests/unit/applications/test_wa_wiring_port_inversion.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:228 |
| `tests/unit/agents/web_search_qualifier/test_wa_search_provider_injection.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:229 |
| `tests/unit/architecture/test_wa_contract_relocation_parity.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:230 |
| `tests/unit/agents/test_wa_agent_import_rule.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:231 |
| `tests/qualification/harness_01/nexus_import_inventory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:237 |
| `tests/unit/architecture/test_ebh_4_r1_r3_b6_typing_gate.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:238 |
| `tests/unit/applications/test_orchestration_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:239 |
| `tests/unit/applications/test_harness_host_runtime_llm.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:240 |
| `testing_support/application_environment_test_support.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:241 |
| `tests/fixtures/harness_host_task_execution.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:242 |
| `tests/unit/applications/_shared/test_obs_diag_x3_universal_spine_adoption.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:243 |
| `tests/unit/applications/architecture/test_diag_platform_adoption_gate.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:244 |
| `tests/system/functional_diagnostics_q3/runner.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:245 |
| `tests/unit/applications/test_effective_profile_revision_adoption.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:248 |
| `tests/unit/architecture/test_ebh_2e_r4_agent_llm_composition_injection.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:248 |
| `tests/unit/architecture/test_ebh_2e_r4_r1_structural_replaceability_and_metadata_resolution.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md:248 |
| `tests/unit/architecture/test_wa_integrations_shared_leakage.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/contracts/environment_profile/root.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_wa_kr18_contract_root_no_shared_import.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-WB (T-IGX-043-H)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/contracts/profile_projection.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:132 |
| `intergrax/applications/_shared/import_allowlist.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:132 |
| `intergrax/runtime/wiring/llm_resolver.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:135 |
| `intergrax/applications/_shared/llm_resolver.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:136 |
| `intergrax/runtime/nexus/engine/runtime_state.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:137 |
| `intergrax/contracts/host_profile_slices.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:139 |
| `intergrax/applications/_shared/runtime_boundary_adapters.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:140 |
| `intergrax/applications/_shared/reliability_runtime_bridge.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:141 |
| `intergrax/applications/_shared/wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:143 |
| `intergrax/applications/_shared/agent_resolution.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:144 |
| `intergrax/applications/_shared/migration_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:145 |
| `intergrax/applications/_shared/mcp_import_guard.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:146 |
| `intergrax/agents/readiness/scoreboard.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:147 |
| `applications/local_workspace_application/host/main.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:148 |
| `applications/research_application/host/main.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:149 |
| `applications/research_application/host/settings.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:150 |
| `applications/research_application/.env.example` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:151 |
| `intergrax/applications/_shared/environment_conformance.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:157 |
| `intergrax/applications/_shared/environment_diff_wiring.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:158 |
| `intergrax/applications/_shared/roster_agent_contract_authority.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:159 |
| `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:165 |
| `applications/governed_contractor_application/hosting/__init__.py` | new | cond | NEW_IN_NEW_DIR | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:166 |
| `applications/governed_contractor_application/hosting/profile.py` | new | cond | NEW_IN_NEW_DIR | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:167 |
| `applications/governed_contractor_application/hosting/runtime.py` | new | cond | NEW_IN_NEW_DIR | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:168 |
| `applications/governed_contractor_application/hosting/boundary.py` | new | cond | NEW_IN_NEW_DIR | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:169 |
| `applications/governed_contractor_application/deploy/docker-compose-local-qualification/docker-compose.yml` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:170 |
| `applications/governed_contractor_application/deploy/docker-compose-local-qualification/RUNBOOK.md` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:171 |
| `applications/governed_contractor_application/deploy/linux/single-node-production-profile-v1/intergrax-gca-strict-production.env.example` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:172 |
| `intergrax/llm_adapters/USAGE.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:178 |
| `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:186 |
| `tests/unit/applications/test_wb_llm_resolver_canonical.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:192 |
| `tests/unit/applications/test_wb_llm_resolver_delegation.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:193 |
| `tests/unit/contracts/test_wb_profile_projection.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:194 |
| `tests/unit/applications/test_wb_profile_projection_registry.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:195 |
| `tests/unit/applications/test_wb_serving_bind_defaults.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:196 |
| `tests/unit/applications/test_wb_import_allowlist.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:197 |
| `tests/acceptance/llm_routing/test_concurrent_run_routing_isolation.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:203 |
| `tests/unit/llm_adapters/routing/test_evaluating_adapter.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:204 |
| `tests/unit/llm_adapters/routing/test_routing_metering.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:205 |
| `tests/unit/llm_adapters/routing/test_secondary_evaluating_wrap.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:206 |
| `tests/unit/architecture/test_ebh_2e_r6_r2_r3_routing_evaluator_end_to_end_injection.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:207 |
| `tests/unit/architecture/test_ebh_2e_r6_r1_r5_tier1_failover_chain_ordering_convergence.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:208 |
| `tests/unit/applications/test_wb_gca_hosted_strict_serving.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:214 |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_package_b_strict_serving_wiring_gate.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:215 |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_p1_strict_production_reachability.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:216 |
| `tests/qualification/prod_q/_prod_q_r1_r7_q_p1_strict_reachability.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WB/PLAN_T-IGX-043-H.md:217 |
| `tests/unit/architecture/test_rw_bnd05_provider_switch_gate.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_own05_duplicate_mechanisms.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-FC2 (T-IGX-043-B)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/runtime/capacity/production_adapters.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:85 |
| `intergrax/applications/_shared/production_capacity_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:86 |
| `intergrax/applications/_shared/production_queue_resolver.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:87 |
| `intergrax/applications/_shared/queue_worker_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:88 |
| `intergrax/applications/_shared/memory_provider_admission.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:89 |
| `intergrax/applications/_shared/memory_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:90 |
| `intergrax/applications/_shared/runtime_event_delivery_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:91 |
| `intergrax/applications/_shared/sandbox_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:92 |
| `intergrax/applications/_shared/wiring_posture.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:93 |
| `tests/unit/runtime/capacity/test_fc2_capacity_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:98 |
| `tests/unit/applications/test_fc2_queue_worker_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:98 |
| `tests/unit/applications/test_fc2_memory_org_session_admission.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:98 |
| `tests/unit/applications/test_fc2_otlp_composition_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:98 |
| `tests/unit/applications/test_fc2_sandbox_composition_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:98 |
| `tests/unit/applications/architecture/test_fc2_capacity_ci_posture_not_used_by_product_code.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:98 |
| `docs/project/architecture/ELASTIC_CAPACITY_AND_SCALING.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:102 |
| `docs/project/architecture/BACKGROUND_TASKS.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:102 |
| `docs/project/architecture/MEMORY_ARCHITECTURE.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:102 |
| `docs/project/technical/guides/PLATFORM_CONFIGURATION.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:102 |
| `tests/unit/docs/test_platform_configuration_reference.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-B.md:102 |
| `intergrax/runtime/integrations/observability_otlp.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/observability/export_boundary.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/observability/test_fc2_kr19_import_order_independent.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-FC4 (T-IGX-043-D)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/distributed/contracts/kv_store.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:103 |
| `intergrax/distributed/providers/sqlite_kv_store.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:104 |
| `intergrax/distributed/providers/redis_kv_store.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:105 |
| `intergrax/integrations/providers/key_value_cache/redis/bundle.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:106 |
| `intergrax/skills/execution_pinning_persistence.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:107 |
| `intergrax/skills/execution_pinning_durability.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:108 |
| `tests/unit/skills/test_skill_execution_pinning_durability_guard.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:114 |
| `tests/unit/distributed/contracts/test_kv_durability_attestation.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:115 |
| `tests/unit/distributed/test_sqlite_kv_store_durability.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:116 |
| `tests/unit/distributed/test_redis_kv_store_durability.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:117 |
| `tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:118 |
| `tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-4/PLAN_T-IGX-043-D.md:119 |

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
| `scripts/governance/iiac_common.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `scripts/governance/verify_iiac_audit_manifest.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `scripts/governance/verify_iiac_role_separation.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `scripts/governance/iiac_activation_gate.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/governance/test_e05_role_separation_identity.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-D08 (T-IGX-046-D)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `agents/README.md` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/integrations/providers/graph_store/neo4j/USAGE.md` | file | firm | EXISTS | T-AF2-RW-PLAN |

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
| `intergrax/platform/enterprise_audit/impact_map.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:99 |
| `intergrax/platform/enterprise_audit/changed_since.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:100 |
| `intergrax/platform/enterprise_audit/runner.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:101 |
| `intergrax/platform/enterprise_audit/digests.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:102 |
| `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:103 |
| `intergrax/platform/enterprise_audit/data/schemas/frz_coverage.v1.schema.json` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:104 |
| `scripts/enterprise_audit/run_enterprise_audit.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:112 |
| `scripts/enterprise_audit/build_impact_map.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:113 |
| `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:119 |
| `tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:120 |
| `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:121 |
| `tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:122 |
| `intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:139 |
| `intergrax/platform/enterprise_audit/manifest.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:140 |
| `scripts/enterprise_audit/mechanical_readiness_check.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:142 |
| `tests/qualification/ent_audit_x/unit/test_frz_coverage_matrix_complete.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:143 |
| `tests/qualification/ent_audit_x/unit/test_frz_aud_w402_neg.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:144 |
| `tests/qualification/ent_audit_x/unit/test_manifest_and_exits.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:145 |
| `tests/qualification/ent_audit_x/integration/test_adversarial_adv01_30.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:146 |
| `tests/qualification/ent_audit_x/integration/test_runner_full_smoke.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:147 |
| `tests/qualification/pdc_x/p6_impact/test_p6_impact_map_coverage.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:148 |
| `tests/qualification/pdc_x/p8_repair/test_r4_impact_map_classification.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WD/PLAN_T-IGX-043-I.md:149 |

### PT-FC1 (T-IGX-043-A)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/runtime/nexus/agent_router.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:82 |
| `intergrax/runtime/nexus/nexus_loop.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:83 |
| `intergrax/runtime/nexus/execution/graph_executor.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:84 |
| `intergrax/runtime/execution/nexus_host_execution.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:85 |
| `intergrax/runtime/execution/debug_lab_nexus_loop.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:86 |
| `intergrax/runtime/execution/host_orchestration_loop_init_spec.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:87 |
| `intergrax/runtime/execution/environment_orchestration_materialization.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:88 |
| `intergrax/runtime/execution/host_orchestration_environment_spec_builder.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:89 |
| `intergrax/runtime/execution/worker_host_task_execution_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:90 |
| `intergrax/runtime/task_memory/store.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:91 |
| `intergrax/runtime/events/store.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:92 |
| `intergrax/runtime/human/persistence_errors.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:93 |
| `intergrax/applications/_shared/harness_host_runtime.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:94 |
| `intergrax/applications/_shared/task_memory_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:95 |
| `applications/USAGE.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `intergrax/applications/USAGE.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/dispute_analyst/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/dispute_intake/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/dispute_scenario/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/dispute_strategist/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/intergrax_assistant/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/legal/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/local_indexer/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/local_search/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/local_synthesizer/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `agents/signoff_probe/notebooks/01_*_experiment.ipynb` | glob | firm | GLOB_MATCHES_1 | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:98 |
| `applications/governed_contractor_application/tests/host/test_governed_contractor_canonical_execution.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:100 |
| `applications/legal_application/tests/serving/test_legal_fastapi_chat.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:100 |
| `applications/legal_application/tests/serving/test_legal_serving_identity.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:100 |
| `applications/local_workspace_application/tests/host/test_lkw_canonical_execution.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:100 |
| `intergrax/runtime/task/nexus_worker_execution.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:102 |
| `intergrax/runtime/task/worker_bootstrap.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:102 |
| `agents/defaults.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:102 |
| `agents/organization_worker/organization_worker_agent.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:102 |
| `agents/external_contractor_adapter/external_contractor_adapter_agent.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:102 |
| `intergrax/agents/authoring/acp_stub_reflex.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:102 |
| `intergrax/applications/_shared/tool_wiring.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:102 |
| `testing_support/agent_distribution/dg001_canonical_multi_agent_diagnostic_harness.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:104 |
| `testing_support/agent_distribution/multi_agent_coordination_qualification_harness.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:104 |
| `testing_support/obs_universal_spine/diagnostic_execution_stack.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:104 |
| `tests/unit/runtime/nexus/test_fc1_agent_router_posture_and_fallback.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:107 |
| `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:108 |
| `tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:109 |
| `tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:110 |
| `tests/unit/architecture/test_fc1_router_and_loop_construction_gate.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:111 |
| `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:114 |
| `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:114 |
| `docs/project/architecture/MEMORY.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:114 |
| `docs/project/architecture/satellites/OBSERVABILITY_extended_depth.md` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md:114 |
| `tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-FC2-E (T-IGX-043-E)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `scripts/maintenance/check_production_capacity_adapters.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-E.md:55 |
| `scripts/maintenance/check_sandbox_policy_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-2/PLAN_T-IGX-043-E.md:56 |

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

### PT-FC3 (T-IGX-043-C)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/integrations/contracts/base.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:122 |
| `intergrax/integrations/contracts/llm_guardrail.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:123 |
| `intergrax/core/security/security_configuration.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:124 |
| `intergrax/rag/embedding/registry/provider_authority.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:125 |
| `intergrax/rag/embedding/registry/profile.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:126 |
| `intergrax/rag/embedding/runtime/resolver.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:127 |
| `intergrax/rag/embedding/bootstrap/default_embedding_engine.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:128 |
| `intergrax/rag/vectorstore/bootstrap/integration_vectorstore.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:129 |
| `intergrax/runtime/notifications/backend_contract.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:130 |
| `intergrax/runtime/notifications/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:131 |
| `intergrax/model_inference/contracts.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:132 |
| `intergrax/model_inference/adapters/stub_vision.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:133 |
| `intergrax/model_inference/adapters/triton_vision.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:134 |
| `intergrax/model_inference/adapters/huggingface_inference_vision.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:135 |
| `intergrax/model_inference/workers/task_runner.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:136 |
| `intergrax/tools/providers/vision/inference_support.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:137 |
| `intergrax/tools/providers/vision/contracts.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:138 |
| `intergrax/tools/providers/vision/service.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:139 |
| `intergrax/runtime/interactions/verification/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:140 |
| `intergrax/runtime/interactions/verification/contract.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:141 |
| `intergrax/runtime/interactions/intake_service.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:142 |
| `intergrax/applications/_shared/interaction_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:143 |
| `intergrax/integrations/providers/llm_guardrail/_factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:144 |
| `intergrax/integrations/providers/llm_guardrail/_stub_backend.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:145 |
| `intergrax/integrations/providers/llm_guardrail/_pattern_scanner.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:146 |
| `intergrax/integrations/providers/llm_guardrail/bundles/llm_guard.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:147 |
| `intergrax/integrations/providers/llm_guardrail/bundles/presidio.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:147 |
| `intergrax/integrations/providers/llm_guardrail/bundles/guardrails_ai.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:147 |
| `intergrax/integrations/providers/llm_guardrail/bundles/llama_guard.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:147 |
| `intergrax/integrations/providers/llm_guardrail/bundles/nemo_guardrails.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:147 |
| `intergrax/integrations/providers/llm_guardrail/bundles/bedrock_guardrails.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:147 |
| `intergrax/integrations/providers/llm_guardrail/bundles/http_guardrail.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:147 |
| `intergrax/applications/_shared/environment_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:154 |
| `intergrax/applications/_shared/guardrail_runtime_bridge.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:155 |
| `intergrax/applications/_shared/harness_host_auxiliary_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:156 |
| `intergrax/applications/_shared/notification_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:157 |
| `intergrax/model_inference/registry/vision_profile.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:158 |
| `intergrax/rag/bootstrap/entry_point_load.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:159 |
| `intergrax/rag/bootstrap/rag_stack_bootstrap.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:159 |
| `intergrax/rag/bootstrap/hierarchical_bootstrap.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:159 |
| `intergrax/rag/document_splitters/bootstrap/default_chunking_engine.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:160 |
| `intergrax/rag/rerankers/bootstrap/reranker_bootstrap.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:161 |
| `intergrax/rag/retrievers/bootstrap/retriever_bootstrap.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:162 |
| `intergrax/tools/providers/rag/scope.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:163 |
| `intergrax/runtime/long_running/notification.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:164 |
| `intergrax/runtime/long_running/coordinator.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:164 |
| `applications/local_workspace_application/host/host_runtime_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:165 |
| `intergrax/scaffold/new_application.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:166 |
| `intergrax/scaffold/new_application_product.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:166 |
| `intergrax/debug/app.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:167 |
| `intergrax/lab/organization_worker.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:168 |
| `applications/lab_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:169 |
| `applications/dispute_sim_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:170 |
| `applications/governed_contractor_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:170 |
| `applications/intergrax_assistant_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:170 |
| `applications/legal_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:170 |
| `applications/local_workspace_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:170 |
| `applications/poc_template_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:170 |
| `applications/research_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:170 |
| `applications/local_workspace_application/workspaces/indexed_vector_verifier.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:171 |
| `applications/local_workspace_application/scripts/lkw_ollama_embedding_bootstrap.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:172 |
| `applications/local_workspace_application/scripts/run-lkw-file-watcher-e2e-proof.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:172 |
| `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:180 |
| `tests/unit/rag/vectorstore/test_fc3_vectorstore_selector_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:180 |
| `tests/unit/runtime/notifications/test_fc3_notification_selector_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:180 |
| `tests/unit/model_inference/test_fc3_vision_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:180 |
| `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:180 |
| `tests/unit/integrations/providers/llm_guardrail/test_fc3_guardrail_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-C.md:180 |

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

### PT-WE (T-IGX-043-K)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/runtime/nexus/tools/invoker.py` | file | cond | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/runtime/nexus/tools/test_we_invoker_residual_hardening.py` | new | cond | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### T-AF2-RW-TRUST (T-IGX-051-C)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/llm/messages.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/runtime/nexus/context/graph_assembly.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/context/contracts.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/context/policy/egress.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/context/provider_lifecycle.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/context/planner.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
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
| `tests/qualification/config_x/_config_x_semantic_production_scan.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-F.md:53 |
| `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/FC-3/PLAN_T-IGX-043-F.md:54 |

### PT-WF (T-IGX-043-J)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `intergrax/applications/_shared/evaluation_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:181 |
| `intergrax/applications/_shared/policy_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:182 |
| `intergrax/applications/_shared/adaptive_wiring.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:183 |
| `intergrax/applications/_shared/checkpoint_introspection_routes.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:184 |
| `intergrax/scaffold/new_application_product.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:185 |
| `applications/local_workspace_application/serving/schemas.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:186 |
| `applications/local_workspace_application/serving/fastapi_router.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:187 |
| `applications/local_workspace_application/serving/workspace_routes.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:188 |
| `applications/local_workspace_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:189 |
| `applications/dispute_sim_application/serving/schemas.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:190 |
| `applications/dispute_sim_application/serving/fastapi_router.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:191 |
| `applications/dispute_sim_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:192 |
| `applications/dispute_sim_application/host/settings.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:193 |
| `applications/attestation_demo/serving/fastapi_router.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:194 |
| `applications/attestation_demo/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:195 |
| `applications/governed_contractor_application/serving/schemas.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:196 |
| `applications/legal_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:197 |
| `applications/legal_application/host/settings.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:198 |
| `applications/research_application/host/factory.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:199 |
| `applications/research_application/host/settings.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:200 |
| `intergrax/contracts/decision_authoritative_exposure.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:206 |
| `intergrax/runtime/task/task.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:207 |
| `intergrax/runtime/nexus/nexus_loop.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:208 |
| `intergrax/runtime/nexus/orchestration/nexus_decision_exposure.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:209 |
| `intergrax/runtime/nexus/orchestration/task_finisher.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:210 |
| `intergrax/runtime/nexus/orchestration/graph_runner.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:211 |
| `intergrax/runtime/decision_exposure_mapping.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:212 |
| `intergrax/runtime/execution/decision_finalization_conformance.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:213 |
| `intergrax/agent_distribution/deployment.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:219 |
| `intergrax/applications/_shared/colocated_serving_runtime_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:220 |
| `intergrax/runtime/execution/child.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:247 |
| `intergrax/runtime/execution/qualified_capability_execution_composition.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:248 |
| `intergrax/compat/langchain/documents.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:249 |
| `intergrax/rag/document_loaders/compat/legacy_runtime_document.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:250 |
| `intergrax/experiments/store.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:251 |
| `intergrax/debug/router.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:252 |
| `intergrax/applications/_shared/harness_auth.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:253 |
| `intergrax/runtime/execution/runtime.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:256 |
| `intergrax/debug/app.py` | file | firm | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:256 |
| `intergrax/runtime/enterprise_reliability/sqlite_provider_invocation_store.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:261 |
| `intergrax/contracts/provider_invocation.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:262 |
| `intergrax/contracts/provider_invocation_store.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:263 |
| `intergrax/runtime/enterprise_reliability/admission_boundary.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:264 |
| `intergrax/contracts/enterprise_reliability/admission_boundary.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:265 |
| `intergrax/runtime/adaptive/policy_learning_approval.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:266 |
| `tests/unit/applications/test_wf_tenant_evaluation_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:276 |
| `tests/unit/applications/test_wf_serving_tenant_fail_closed.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:277 |
| `tests/unit/architecture/test_wf_no_default_tenant_in_request_contracts.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:278 |
| `tests/unit/runtime/nexus/test_wf_exposure_payload_typed_seam.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:279 |
| `tests/unit/applications/test_wf_colocated_drain_protocol.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:280 |
| `tests/unit/compat/langchain/test_wf_trusted_scope_conversion.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:291 |
| `tests/unit/debug/test_wf_experiment_tenant_scope.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:292 |
| `tests/unit/runtime/execution/test_wf_decision_finalization_strict.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:293 |
| `tests/unit/runtime/execution/test_wf_child_admission_narrowing.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:294 |
| `tests/unit/applications/test_wf_debug_router_guard_idp.py` | new | firm | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:295 |
| `tests/unit/runtime/enterprise_reliability/test_wf_provider_invocation_tenant.py` | new | cond | NEW_PARENT_EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:296 |
| `tests/qualification/ebh_6/_q2d2_technical_qualification.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:302 |
| `tests/qualification/ebh_6/test_ebh_6_q2d2_dual_path.py` | file | cond | EXISTS | .tmp/session/eqo-a-remediation/FC/WF/PLAN_T-IGX-043-J.md:303 |
| `intergrax/applications/_shared/production_lifecycle.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/applications/_shared/strict_production_serving_launcher.py` | file | firm | EXISTS | T-AF2-RW-PLAN |

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

### T-AF2-RW-TYP (T-IGX-051-G)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `scripts/maintenance/check_typ_ratchet.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/typ_ratchet_baseline.v1.json` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_typ_ratchet.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `intergrax/contracts/capability_metering/usage_event.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/contracts/governed_execution_result.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/contracts/runtime_execution_context.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `intergrax/contracts/sanctioned_external_work_operation.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_rw_typ05_string_dispatch_reduction.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |

### PT-T01 (T-IGX-043-L)

| Path | Kind | Class | Verified at HEAD | Source |
| --- | --- | --- | --- | --- |
| `scripts/maintenance/check_runtime_event_tenant_propagation.py` | file | firm | EXISTS | T-AF2-RW-PLAN |
| `tests/unit/scripts/test_t01_runtime_event_tenant_propagation_emitters.py` | new | firm | NEW_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/unit/architecture/test_t01_nt*.py` | new_glob | firm | NEW_GLOB_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/tenant_x/test_t01_*.py` | new_glob | firm | NEW_GLOB_PARENT_EXISTS | T-AF2-RW-PLAN |
| `tests/qualification/trace_x/test_t01_*.py` | new_glob | firm | NEW_GLOB_PARENT_EXISTS | T-AF2-RW-PLAN |

