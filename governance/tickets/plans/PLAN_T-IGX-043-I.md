# PLAN T-IGX-043-I — EQO-A remediation W-D / T-ENF-6: ENT-AUDIT-X engine fixes (CHANGED-SINCE default-deny, FRZ coverage attribution, FAST promotion disclosure)

Status: DRAFT for Operator review. Drafted 2026-11-16 by a planning session (read-only: no branch, no worktree, no commit, no test run, no tracked file changed). Nothing here is approved, applied or audited. Master-plan node `PT-WD` (`WP-12/TICKETS_DRAFT.md`, `WP-12/DAG.md`), authorization row `AR-42`, WP-7 plan B section V6 and ticket row `T-ENF-6`. The Operator accepted DP-66 as recommended (decision session 6, `OPERATOR_DECISIONS_REMEDIATION_WAVE1.md`); the other design choices below are drafted on the recommended option and listed in `OPEN_QUESTIONS.md` with what each would change. **Amended 2026-11-17** for the ARCH-FREEZE ruling Q-RW-04 (section 1, 3.3a, 9.3 and the write set) and for the Campaign B-2 alpha item X-A6-003 (section 1); the amendment is state-neutral: it states what the draft changes, not what any gate, audit or stage has concluded. A second amendment of 2026-11-17 reconciles the draft with RW-PLAN v13 appendix A1 (section 2) and records the train-head facts (3.6, 9.3).

Notation. `B` = `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (Campaign A baseline, PR #589). `H` = `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (`origin/development` after one `git fetch origin development` on 2026-11-16). `path@B:line` and `path@H:line` mean the full SHAs above; `FACT_CHECK.md` spells them out in every row. Head status of every file named below: `UNCHANGED` at H (`git diff --name-only B H -- intergrax agents applications` lists only three `intergrax/rag/graph/` files, none of them here; `git diff --stat B H` over `tests/qualification/{ent_audit_x,pdc_x,evol_x,roadmap_replay_x}` and `scripts/enterprise_audit` is empty).

## 1. Purpose, findings, decisions

Make the ENT-AUDIT-X CHANGED-SINCE selector and the FRZ coverage artefact stop failing open or silent, and make the FAST-run status honest, without rewriting the accepted baseline or touching any IIAC validator; under ruling Q-RW-04 the check manifest does change (six mapped ENT checks, section 3.3a) and so does its digest (section 9.3).

| Finding | Severity (dossier) | Defect at B | ARCH-FREEZE impact (WP-7 / master plan) |
|---|---|---|---|
| EQOA-A6-007 | MEDIUM | CHANGED-SINCE is fail-closed only for two prefixes (`impact_map.py@B:15-18`); an unmapped changed product path selects nothing beyond the invariant floor (`impact_map.py@B:335-343`, `changed_since.py@B:126-127`). Static census at B: 2043 of 8852 tracked product files (23 %) match no prefix and no entrypoint | CONDITIONS (FRZ-EVOL-06 / FRZ-AUD-06) |
| EQOA-A6-003 | LOW | `unmapped_evol_changed_paths` accepts only exact entrypoint files (`changed_since.py@B:72-100`); 77 of the 80 files under `tests/qualification/evol_x` are unmapped, among them the eight named by the dossier (`_constants.py`, `conftest.py`, six `adapters/*`); any changed one makes CHANGED-SINCE raise | NONE |
| EQOA-A6-004 | MEDIUM | 60 FRZ criteria (42 PASS) in six families AUD, CTL, DEBT, FRZ, HRN, TRC are attributed to no check (`digests.py@B:25-49`) and the artefact says nothing about it | CONDITIONS |
| EQOA-A6-005 | MEDIUM | 30 checks (CMP, DUP, EXE, GOV, OBS, PLG, REC, REL, SEC, STA, three each) fall through to the aggregator `FRZ-AUD-03` (`digests.py@B:49`); root cause is code (`_frz_for_check`), not data | NONE (dossier) / CONDITIONS (master-plan effective) |
| EQOA-A6-008 | LOW | non-certifying FAST: PARTIAL becomes PASS and exit 0 when only TRACKED_DEBT drift remains (`runner.py@B:266-274`); nothing in the report says so; the condition also lacks the incomplete-floor guard (section 3.4) | NONE |

**Confirmed item (Campaign B-1 alpha, 2026-11-16).** A6-007 was run at B as probe pack R13 (NT-08) and is CONFIRMED: `intergrax/zz_new_pkg/mod.py` through the production selector `select_checks_for_changed_since` selects only the invariant floor (test `test_pp_nt08_new_package_selection` FAILED), and of 153 uncovered packages (L1 sub-packages of `intergrax/ agents/ applications/` plus L2 sub-packages of `intergrax/`) 152 select FLOOR_ONLY and one is REFUSED (`test_pp_nt08_census_uncovered_packages` FAILED); the dossier's "101 uncovered" matches none of the probe's granularities (59 / 94 / 153). Class CONDITIONS, product change required in the RW train, routed to W-D (T-IGX-043-I) with `impact_map.py` as the product path (`.tmp/session/arch-freeze/b1-alpha/RESULTS_B1_ALPHA.md` section 3 and the A6-007 row; `.tmp/session/eqo-a-remediation/B1_ALPHA_ROUTING.md`; run files `.tmp/session/arch-freeze/b1-alpha/runs/R13/`, probe `probes/pp_nt08.py`). Closing negative tests in the section Negative tests: A01 (new package path selects FULL) and A12 (no uncovered package selects floor only; same unit as the probe); the alpha probe is the untracked red-first evidence, A01/A12 are the tracked tests. Reported, not decided here: the alpha probe used the production selector, so the NT-08 retargeting worry of C-07 is already met on the alpha side.

**Confirmed item (Campaign B-2 alpha, 2026-11-17): X-A6-003.** The CHANGED-SINCE selection for the 12 stage certification SHAs was run at B as run R25c (`.tmp/session/arch-freeze/b2-alpha/RESULTS_B2_ALPHA.md`, run manifest and row X-A6-003; `.tmp/session/arch-freeze/b2-alpha/runs/R25c/stdout.log`): CONFIRMED, 12 children, 3 exit 0 and 9 exit 1 with `CHANGED-SINCE fail-closed: unmapped evol_x paths` (TENANT-X, CONFIG-X, COMPAT-X, STATE-X, TRACE-X, CTRL-X, GOV-X2, EBH-5, DUP-X; EVOL-X, ENT-AUDIT-X and MCP-SERVER-X exit 0), the same pattern as Campaign A. Class NONE to NONE, product change required: `impact_map.py` and its data file are product paths. It is in scope as A6-003 (Part B, section 3.2), next to A6-007. Closing negative tests: B01 (every tracked file under `tests/qualification/evol_x` is mapped) and B02 (the eight dossier paths select the five EVOL checks without a raise), both RED at B. Replay: the 12-SHA diagnostic (A-05) must exit 0 for 12 of 12 instead of 3 of 12 (computability only: after Part A most long ranges escalate to FULL). The alpha run is untracked red-first evidence; B01 and B02 are the tracked tests.

**Binding ruling Q-RW-04 (ARCH-FREEZE orchestrator, relayed 2026-11-17).** (i) Q-WD-07 option B is binding: the mapped ENT checks of `PT-C01` for the six families land in this ticket before G2, so the check manifest, `manifest.py` and the pinned tests are in the write set (section 3.3a, section Write set). (ii) `tests/qualification/ent_audit_x/ENTERPRISE_AUDIT_BASELINE.json` is not rewritten in the train: the accepted baseline comes from `T-AF2-ENT-BASELINE-PREFREEZE`, so the file is out of the write set and stays in the forbidden list. (iii) Every ENT run expected to go red because of the manifest digest change (`BASELINE:MANIFEST_DIGEST`) is listed with its owner in section 9.3, for the known-red register. (iv) The ticket is a gate-meaning ticket (next paragraph). The earlier drafted default for Q-WD-07 (option A, declaration only) is superseded by this ruling; the stage-gate-only declaration of Part C stays and now lists only what the new checks do not exercise.

**Gate-meaning ticket.** This ticket changes what the enterprise audit selects (CHANGED-SINCE default-deny), attributes (FRZ family map), reports (FAST status) and checks (six new manifest checks, a new manifest digest, edited expected values in 13 pinned test files). The approval line is therefore not written by the signing delegate alone: it goes through the wave packet (OD-AF-22, RW-PLAN v13): the Operator approves the packet that lists this ticket, its plan file and its plan digest, and the signing delegate then writes the Approved line with a Delegation pointer to that packet record; the delegate does not write it alone. This supersedes the open delegability question of `AUTHORIZATION_REQUESTS.md` A-01.

`A6-001` / `A6-006` (semantic wrappers skip silently, debt gate is existence-only) are NOT in this ticket: they belong to `PT-E03` (`T-ENF-3`, non-product), see section 9.1.

Decisions applied. **DP-66** (accepted as recommended): an unmapped changed product path in CHANGED-SINCE selection selects FULL; the six FRZ families with no mapped check are declared "stage-gate-only" in the coverage artefact now, new mapped ENT checks are assessed in Campaign C; semantic wrappers run in automation (that part is `PT-E03`); ruling Q-RW-04 supersedes the clause "new mapped ENT checks are assessed in Campaign C" for the manifest part (the checks land here). **DP-11** (one integrated wave), **DP-58** (single RB-2), **DP-59/60/61** (EBH-7 pin, ROADMAP-REPLAY-X head, last pin), **C3** (fail-closed; negative tests; regression; independent audit at the exact SHA), **C5**, **C6**, **OD-REM-07**. DP-66 says nothing about the engine tree itself, the exemption of non-code files, the FRZ family of DUP-X or the wave's own touched modules: those are `OPEN_QUESTIONS.md` Q-WD-02..05 with the drafted default.

## 2. Prerequisites and current PDC state (reported, not decided here)

- Wave node `PT-WD`; hard prerequisites `SCOPE` (= M1 + DS-1..DS-4 + DS-EXC + S1 + Campaign B-1/B-2 alpha + REV-MASTER + `T-AF2-RW-PLAN` + its audit) and `PT-E03` (`WP-12/DAG.md` row `PT-WD`); authorization `AR-42` with `AR-32` (`PT-E03`) merged; decision DP-66. The ENT engine and manifest must land in the RW train before G2 (`MEP-AF2_v2.1_final.md` rule PD2-03, line 462-463).
- Regenerated DAG (message of the ARCH-FREEZE session, 2026-11-16): the mapped ENT checks for the six unmapped FRZ families (former `PT-C01`) now sit under `PT-WD`, and `PT-E05` and `PT-D08` now depend on `SCOPE`. The first point is binding under ruling Q-RW-04 (section 1): the six mapped checks are part of this ticket (3.3a), which supersedes the earlier DP-66-only reading (Q-WD-07 option A); the second needs no change here (`PT-E05`/`PT-D08` share no file with this ticket). Product commits of this ticket go only to branch `arch-freeze/rw-train`.
- RW-PLAN v13 reconciliation (2026-11-17; `arch-freeze/rw-plan/RW_PLAN.md`, appendix A1 block `PT-WD (T-IGX-043-I)` and sections (b), (c), (d)): the A1 write set equals the Write set below (9 product paths, 3 scripts, 5 new and 13 pinned test files = 30 firm, 8 conditional; no path added or dropped; the A1 source line references 118 to 190 match this file); section (b) lists no file overlap for `PT-WD`; section (c) places `PT-WD` in lane 3 after `PT-WA` (merged into `origin/arch-freeze/rw-train` as PR #642, head `05699b633cdb1318c215f6529e449420efc1382c`) and `PT-WB` (not merged when this was written); section (d) cites A01..A16, C01..C03, B01, B02 and D01..D06 as NT-ROW-008, 014, 017, 018 and 152 and has no row for C04..C09 and E01..E06 (additional tests of this ticket). The enterprise-audit tree, the pinned tests and the scripts of the Write set are identical at that train head and at H (`git diff --name-only` empty).
- Ticket approval line (through the wave packet, OD-AF-22; gate-meaning ticket, section 1; AR-38), one worktree and branch, the single-process pytest lock, and the authorizations of `AUTHORIZATION_REQUESTS.md` (A-01..A-12). The product-path go AR-39..AR-44 was granted by the Operator on 2026-11-16 but is effective only when M1 holds (`OPERATOR_ANSWERS_TRANSCRIPTION.md`).
- PDC status at H, as written in the files: `PDC_INDEX.md@H` rows 44-49 still read "Answers recorded in chat; awaiting the Operator's final confirmation" for the P9 documents; `PDC_SHA_LADDER.md@H:39` records `FINAL_CLOSURE_PUBLICATION_SHA = feaa2a87d41c3c8ae95724590291f360d0232016` (PR #603). The master plan states M1 is reached. The index text and the ladder are not consistent with each other; this plan does not decide whether P9 is closed.

## 3. Design

### 3.0 Invariants kept

`contracts.py`, `modes.py`, `baseline.py`, `validators.py`, `report.py` and `ENTERPRISE_AUDIT_BASELINE.json` are not edited. The check manifest (`enterprise_audit_check_manifest.v1.json`, 85 checks at B and H) and `manifest.py` are edited by Part E (ruling Q-RW-04), so the manifest digest changes: at B and H the sha256 of the canonical manifest bytes equals `check_manifest_digest` of the baseline (`0fe966e4…691b8b`, reproduced statically from the two blobs); after Part E it does not, and every certifying FULL or FREEZE run against the unchanged baseline reports `BASELINE:MANIFEST_DIGEST` until `T-AF2-ENT-BASELINE-PREFREEZE` re-promotes the baseline. That state is expected and disclosed (section 9.3); no baseline is promoted here (section 9.2). Parts A to D do not change the manifest. The public signatures `select_checks_for_changed_since(...) -> frozenset[str]`, `check_ids_for_changed_path`, `unmapped_fail_closed_paths`, `selected_check_ids_from_impact`, `fail_closed_prefixes`, `load_default_impact_map`, `build_impact_map_from_manifest`, `frz_coverage_artifact(checks)` keep working for existing callers (8 test files and two scripts call them: FACT_CHECK F-21). The impact JSON keeps its shape (`schema_version`, `path_prefix_to_check_ids`) because `test_shipped_ssot_equals_generator_output` pins the exact object and bytes (`tests/qualification/pdc_x/p6_impact/test_p6_impact_map_coverage.py@B:125-133`); new data therefore goes into Python rule tables and the JSON is regenerated, never hand-edited (it is byte-identical to the generator output at B, FACT_CHECK F-14).

### 3.1 Part A — unmapped product path selects FULL (A6-007, DP-66)

Files: `intergrax/platform/enterprise_audit/impact_map.py`, `.../changed_since.py`.

1. **Definitions** (new, in `impact_map.py`): `PRODUCT_ESCALATION_PREFIXES = ("intergrax/", "agents/", "applications/")`, a copy of `PRODUCT_PATH_PREFIXES` (`scripts/governance/iiac_common.py@H:22`; Tier-0 cannot import `scripts/`), tied by a parity test. A changed path (normalised `\` to `/`) *requires classification* when it starts with one of them or with an existing fail-closed prefix (`scripts/enterprise_audit/`, the audit tooling itself). It is *covered* when (a) a manifest entrypoint covers it (`_check_touched`, unchanged), or (b) `check_ids_for_changed_path(path, prefixes)` intersected with the manifest check ids is non-empty, or (c) a rule of the new table `NO_CHECK_REQUIRED_RULES` matches (empty by default, Q-WD-03). New pure function `unmapped_product_paths(changed, *, prefixes, manifest_check_ids, entrypoint_covered) -> frozenset[str]`.
2. **Legacy ids are not coverage.** 12 prefix entries of the shipped map carry legacy ids that are not manifest ids (`C002`, `C004`, `C024`, …); for `runner.py`, `baseline.py`, `modes.py`, `validators.py`, `evidence.py`, `sandbox.py`, `report.py` and others every id is legacy (static check, FACT_CHECK F-12; the dossier's own note counts "86 ids" against 85 checks, `A6/notes_parent.md`). Today such a path counts as mapped and selects nothing real. Rule (b) intersects with the manifest, so these paths become unmapped and select FULL. The legacy ids stay in the generator (an existing test pins `C024`/`C029`, `test_changed_since_impact_map.py@B:14-18`); `select_checks_for_changed_since` additionally drops ids that are not manifest ids from its result.
3. **Planner** (new, in `changed_since.py`): `ChangedSinceSelection(check_ids, escalated_to_full, unmapped_paths, reasons)` and `plan_changed_since_selection(checks, *, repo_root, since_sha)`. Order: `assert_git_ancestor` (unchanged); `git_changed_files`; `unmapped_evol_changed_paths` (raises ValueError, now prefix-aware, Part B); `unmapped_fail_closed_paths` restricted to the paths that are not product paths (today only `scripts/enterprise_audit/`, which the parent-directory key already maps entirely, so it cannot fire; it stays as a guard); then `unmapped_product_paths`, which also covers the two `scripts/enterprise_audit/` files whose own key holds legacy ids only (`build_impact_map.py`, `run_qual_wave.py`; longest prefix wins, so the parent key `ENT-A-DOC-02` is not consulted). If the result is non-empty the selection is `FULL_SET = {c.check_id for c in checks_for_mode(checks, AuditMode.FULL)}` (all 85 at B: every manifest check lists FULL, FACT_CHECK F-16) and `reasons` carries the count and the first eight sorted paths. Otherwise: invariant floor, union of mapped manifest ids, entrypoint-touched checks (unchanged logic). `select_checks_for_changed_since` returns `plan_...().check_ids` (still a `frozenset`).
4. **Engine tree.** `intergrax/platform/enterprise_audit/` is itself a product path: with the planner above an engine file that has no real mapping selects FULL instead of raising (the raise of `changed_since.py@B:118-125` applied to it). DP-66 reads as covering it; whether the old raise is kept is Q-WD-02 (drafted default: FULL). Consequence for this wave: `digests.py`, the frz schema, `contracts.py` and 17 other engine files have no mapping at B (static census, F-13); under a kept raise, W-D's own `digests.py` and schema change would make every CHANGED-SINCE run over a range containing the wave raise.
5. **Fail-closed data loading.** `load_default_impact_map` and `load_impact_map_from_path` share one parser that raises `ValueError` when `path_prefix_to_check_ids` is missing or not an object, when a value is not a list of strings, or when `schema_version != 1` (today `raw.get(..., {})` silently yields an empty map, so every path would look unmapped without any signal; `impact_map.py@B:234-251`).
6. **Renames.** `git_changed_files` adds `--no-renames`: with git 2.46 the default output lists only the new name of a rename, so a move out of an unmapped path is invisible (verified in a scratch repository, F-17).
7. **Disclosure.** `EnterpriseAuditRunner.run` gets `selection_notes: tuple[str, ...] = ()` appended to `limitations`; the CLI passes the planner's `reasons` (section 3.5). The report states the escalation; coverage stays `PARTIAL` for CHANGED-SINCE (`coverage.py@B:37-38`), never FULL.

Non-goals of Part A: no change to what FULL runs, to FAST/FULL/FREEZE planning, or to the invariant floor; governance prefixes outside the product tree (`scripts/`, `tests/qualification/`, `.github/workflows/`) do not escalate (Q-WD-01).

### 3.2 Part B — evol_x test-support tree mapped (A6-003)

Files: `impact_map.py` (rule table), `changed_since.py`, regenerated JSON.

- New table `EVOL_X_TREE_IMPACT_RULES` (kept apart from `PDC_DOMAIN_IMPACT_RULES`, whose test requires `intergrax/` prefixes, `test_p6_impact_map_coverage.py@B:108-116`): `tests/qualification/evol_x/{_constants.py, conftest.py, adapters, data, experiments, frz, metrics}` each to the five ids already attached to `tests/qualification/evol_x/gates` in the shipped map (`ENT-A-EVOL-01..03`, `ENT-A-INV-01`, `ENT-A-REG-02`), added by the generator loop of `build_impact_map_from_manifest` (`impact_map.py@B:296-302`). Together with the existing `gates` key all 80 tracked files under the tree are covered.
- `unmapped_evol_changed_paths(checks, changed, *, prefixes=None)`: with `prefixes` given, a path whose manifest-intersected ids are non-empty is mapped. Without it the old entrypoint-only behaviour is kept (three existing callers, two tests and one script, use two arguments). `select_checks_for_changed_since` passes the prefixes. A new, unmapped directory under `evol_x` still raises (fail closed kept).
- JSON regenerated with `scripts/enterprise_audit/build_impact_map.py --emit-ssot` (the existing generator command; it writes the product JSON, so it runs only in the implementation worktree).
- Closure evidence for the dossier's condition (the 12 stage certification SHAs compute): bounded diagnostic `build_impact_map.py --since-sha <sha>` per SHA (AR-15d, `AUTHORIZATION_REQUESTS.md` A-05). Honest reading: after Part A most long historic ranges will escalate to FULL; exit 0 then proves that the selection is computable, and the per-SHA payload (`escalated_to_full`, reasons, selected count) is what the auditor reads.

### 3.3 Part C — FRZ family attribution and the stage-gate-only declaration (A6-005, A6-004)

Files: `digests.py`, `data/schemas/frz_coverage.v1.schema.json`, `runner.py` (guard and call).

- **Attribution table.** `_frz_for_check` keeps its 11 existing prefix rules and gains explicit rows for the nine categories that currently fall through, by check-id prefix: `ENT-A-CMP-` → `FRZ-CMP-*`, `-EXE-` → `FRZ-EXE-*`, `-GOV-` → `FRZ-GOV-*`, `-OBS-` → `FRZ-OBS-*`, `-PLG-` → `FRZ-PLG-*`, `-REC-` → `FRZ-REC-*`, `-REL-` → `FRZ-REL-*`, `-SEC-` → `FRZ-SEC-*`, `-STA-` → `FRZ-STA-*`; every one of these families exists in the checklist (`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md@H`, F-09). `ENT-A-DUP-` has no FRZ family in the checklist: it gets an explicit entry `("FRZ-AUD-03",)` (aggregator only, no new coverage claim) pending Q-WD-05. Family-level wildcard attribution is the existing convention (`FRZ-TEN-*` …) and the closure wording of the master plan ("family-level mapping for all 85 checks").
- **No silent fall-through.** New `frz_default_fallback_check_ids(checks) -> tuple[str, ...]` returns checks that match no explicit row. `frz_coverage_artifact(checks, *, strict=False)`: with `strict=True` it raises `ValueError` naming them; the runner passes `strict=effective_certifying` and, before planning, returns a BLOCKED report with finding `FRZ:COVERAGE_DEFAULT_FALLBACK` (same pattern as `CERTIFYING:CHECK_FILTER_FORBIDDEN`, `runner.py@B:89-97`) when a certifying run meets such a check. With the shipped manifest the set is empty, so certifying runs are unaffected; a future check added without an attribution row is refused.
- **Stage-gate-only declaration.** `STAGE_GATE_ONLY_FRZ_FAMILIES` in `digests.py` (family, owner stage as written in the checklist owner column, one-sentence basis) for AUD (FRZ-AUD-01..10 except the aggregator FRZ-AUD-03), CTL (CTRL-X), DEBT (ARCH-FREEZE), FRZ (ARCH-FREEZE), HRN (HARNESS-FINAL), TRC (TRACE-X, GOV-X*). `frz_coverage_artifact` adds a top-level block `stage_gate_only` (array of those objects). The block carries no criteria counts, so it does not rot when the checklist changes. Under ruling Q-RW-04 each of the six families also gains one mapped check (Part E); the block then names, per family, only the criteria that check does not exercise (content chosen with the check design, Q-WD-11), and a family whose criteria are all exercised drops out of it. Rows, `row_count == 85` and the row keys are unchanged (the mechanical readiness check requires 85 rows, `mechanical_readiness_check.py@B:391-405`).
- **Schema.** `frz_coverage.v1.schema.json` is `additionalProperties: false` at the top level and per row (`@B:6`, `@B:15`); the block is added as an optional property (not in `required`), same `v1` and `schema_version: 1`, so published bundles without it still validate (`validate_frz_coverage` and `validate_b5_handoff_bundle`, `validators.py@B:119-120,194-204`). The alternative of a separate artefact (needs the evidence catalog, validators and a new schema) is listed in Q-WD-09.
- **Completeness test, not a number.** A test reads the families present in the checklist and asserts that each is either attributed by at least one check or listed in `stage_gate_only`; a family added to the checklist later forces a decision (`test_wd_c05`).
- **Mapped ENT checks.** Delivered by Part E (3.3a), ruling Q-RW-04. The DP-66 clause "assessed in Campaign C" is superseded for the manifest part; Campaign C (`PT-C01`) keeps the independent re-judging sample only (`DAG.md` row `PT-C01`).

### 3.3a Part E — six mapped ENT checks for the families AUD, CTL, DEBT, FRZ, HRN, TRC (A6-004, ruling Q-RW-04; former `PT-C01` content)

Files. Product: `data/enterprise_audit_check_manifest.v1.json`, `manifest.py`, `evidence.py` (plus the rows of `digests.py` from Part C and the generated rules of `impact_map.py` and its JSON). Scripts: `scripts/enterprise_audit/mechanical_readiness_check.py`. Tests: five new files (the fifth holds E01..E06) and the 13 pinned files of the write set.

- **Design, drafted (owner decisions in Q-WD-11 to Q-WD-13).** One new check per family, named by the existing id convention and by the family prefix that Part C attributes: `ENT-A-AUD-01`, `ENT-A-CTL-01`, `ENT-A-DEBT-01`, `ENT-A-FRZ-01`, `ENT-A-HRN-01`, `ENT-A-TRC-01`; the manifest grows from 85 to 91 checks. Category `architecture` (the enum `AuditCategory` in `contracts.py` is not extended), `severity_default` ERROR, `mutating` false, `depends_on` `ENT-A-INV-01` (as `VBN-REACHABILITY` does), modes FULL, FREEZE and CHANGED-SINCE (not FAST: the FAST floor and the required PR check `ent-audit-fast` keep their composition, Q-WD-12), `pg_id` as the owner assigns.
- **Entrypoints are not chosen by this plan.** No check design for the six families exists in the records read (UNVERIFIED that none exists elsewhere), and the criteria differ in kind: the checklist at H lists FRZ-TRC (12 rows), FRZ-CTL (12), FRZ-HRN (8), FRZ-AUD (10) with PASS evidence by stage record, while all six FRZ-DEBT rows and twelve of the thirteen FRZ-FRZ rows read OPEN. A check for an OPEN criterion that passes before the state exists would assert a state that does not exist; one that fails turns every FULL run red. The ENT-AUDIT-X owner names, per family, one pytest entrypoint that asserts a machine-checkable property, and which criteria it does and does not exercise (the latter feed the `stage_gate_only` block of Part C). Existing suites found by listing at H, offered as candidates only (suitability UNVERIFIED): TRC `tests/qualification/trace_x/` (40 test files); HRN `tests/qualification/harness_01/` (14); DEBT `tests/unit/runtime/architecture/test_debt_governance.py` and `tests/unit/governance/test_arch_freeze_b1_closed_world_inventory.py`; AUD and FRZ `tests/qualification/ent_audit_x/unit/test_frz_*`; CTL `tests/qualification/control_planes/`. Whether a new entrypoint file is needed for DEBT, FRZ or CTL is open and would add test files to the write set.
- **Count constants.** `_validate_manifest` pins `85` (`manifest.py@B:61-62`) and `evidence.py@B:70-75` pins `85` three times for closed-world FULL and FREEZE (expected checks, `checks_planned`, outcomes). Both move to one shared constant `EXPECTED_CHECK_COUNT = 91` defined in `manifest.py` and imported by `evidence.py`; the closed-world purpose stays (adding or dropping a check without editing the constant is refused in both directions, test E03). `evidence.py` is not named by the ruling; the design adds it (C-10). `scripts/enterprise_audit/mechanical_readiness_check.py@H:401,453` pins 85 rows and 85 planned and executed (the 85 coverage rows become 91 by Part C and E).
- **Attribution and coverage.** Six explicit rows `ENT-A-<FAM>-` to `FRZ-<FAM>-*` in the Part C table; the coverage artefact then has 91 rows (`row_count` 91), no fall-through (C01, E04). Adding a check does not change the status of any FRZ row: the rows keep the stage-gate evidence they cite; the artefact states which criteria the new checks exercise and which they do not.
- **Impact map.** The generator derives the rules for the new entrypoints; the JSON is regenerated and pinned by the existing generator-equality test (B04); each new entrypoint path selects its check (E05).
- **Pinned tests.** The 13 files of the write set pin the count 85, the row count or the old digest literal. They are edited in the same commit as the manifest so that no commit leaves them red; only the expected value is updated, no assertion is removed or loosened. Changing an expected value of an ENT or PDC gate is part of why the ticket is a gate-meaning ticket.
- **Acceptance wrappers.** `tests/qualification/ent_audit_x/acceptance/` holds one generated wrapper per check (85 wrapper files and three helper files at H, `tools/generate_acceptance_wrappers.py`) and belongs to `PT-E03`; whether each new check gets a wrapper (and so whether `scripts/enterprise_audit/qual_wave_lib.py` `_SEMANTIC_CHECK_COUNT` and the env-gated "full 85 gate matrix" in `test_semantic_gate_execution.py` change) is Q-WD-13; the files are listed in the conditional write set.
- **Not part of Part E.** `ENTERPRISE_AUDIT_BASELINE.json`, `--bootstrap-candidate`, promotion and any IIAC record (`T-AF2-ENT-BASELINE-PREFREEZE`); `contracts.py`; `.github/`; FAST membership; FRZ checklist edits.

### 3.4 Part D — FAST promotion disclosure (A6-008)

File: `runner.py` (lines 266-274 at B).

- The condition moves into a pure helper `_non_certifying_fast_promotion(...)` returning whether to promote and the TRACKED_DEBT count. On promotion the report gets a limitation line `status_promoted_from=PARTIAL: non-certifying FAST run; only TRACKED_DEBT drift remains (tracked_debt_count=N); aggregate status raised to PASS, exit code to 0`. `limitations` is a string array already rendered into the JSON and Markdown report and validated as strings (`report.py@B:38,118-119`, report schema `limitations`), so no schema or contract change; the dossier's "report header" wish is met by the limitations block that precedes the findings. A dedicated header field would need `AuditReport`, `report.py` and the report schema (Q-WD-06 option B2).
- Guard added (Q-WD-06, drafted default): no promotion when `incomplete` is true. At B `aggregate_status` returns PARTIAL for "incomplete and any finding" (`modes.py@B:28-29`), and the promotion cannot tell that case from "PARTIAL because of active debt". A FAST run whose mandatory floor was not fully executed (`fast_floor_incomplete`, `modes.py@B:56-72`) with only TRACKED_DEBT findings would therefore exit 0. Static reading only: the case needs a mandatory check without execution or with a TRACKED_DEBT finding plus a debt finding, and the baseline is empty at B (`ENTERPRISE_AUDIT_BASELINE.json@B`, `"findings": {}`), so it is latent and not shown by a run (UNVERIFIED). It matters because the required PR check `ent-audit-fast` runs the real FAST non-certifying CLI (`enterprise-audit.yml@H:47-50`). Exit-code behaviour for complete runs is unchanged.
- FULL, FREEZE, CHANGED-SINCE and certifying FAST are never promoted (guard tests).

### 3.5 Scripts (non-product, same ticket)

`scripts/enterprise_audit/run_enterprise_audit.py` calls `plan_changed_since_selection`, passes `selection_notes` to the runner and prints a stderr notice on escalation; `scripts/enterprise_audit/build_impact_map.py` writes `escalated_to_full`, `unmapped_product_paths` and the planner `reasons` into its payload and passes `prefixes` to `unmapped_evol_changed_paths`. They are not ACT-N11 paths (`scripts/` is outside `PRODUCT_PATH_PREFIXES`) and not on the EBH-7 identity allowlist, so they merge in the wave before the pin. The commit trailer `Ref: governance/tickets/T-IGX-043-I.md` is required (`scripts/` is a ticket-governed path).

### 3.6 Wave-diff mapping (last commit of the ticket)

Two pinned PDC tests assert that every product module changed since the PDC baseline `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` maps to at least one check: `test_p6_impact_map_coverage.py::test_every_pdc_touched_product_module_maps_to_at_least_one_check` (`@B:143-154`) and `test_xd_12_evol_ent_audit.py::test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check`. The range now holds 192 product files (static count at H) and will contain the whole wave. Files already named in the FC-1..FC-4 plans alone include at least 30 modules in unmapped groups (`intergrax/runtime` 13, `intergrax/model_inference` 7, `intergrax/distributed` 3, `agents/*` 3, others; static scan of the plans, F-20), so these two tests would turn red at the wave head regardless of Part A. W-D owns `impact_map.py` and the JSON and is the only ticket that may edit them. Last step: at the integrated wave head compute `git diff --name-only a695c9c8…<head> -- intergrax agents applications` minus the covered files and add one rule per touched package group to a new table, mapped to every manifest id (explicit FULL until the owner supplies narrower ids), with a comment per group; regenerate the JSON (it grows by about 85 lines per group). Alternatives (nearest-owner ids; editing the two PDC tests, which is a gate-meaning change and the Operator's personal ruling) are Q-WD-04. At the train head `05699b633cdb1318c215f6529e449420efc1382c` a static replication of the mapping rule (prefix ids of the shipped map, or a manifest entrypoint) finds 228 touched `.py` modules since the PDC baseline, of which 5 are unmapped (`agents/tool_selection_qualifier` 2, `agents/web_search_qualifier` 2, `agents/local_search` 1), all touched by `PT-WA`; at H it finds 189 and 0. The replication is not the test itself and no run was made (UNVERIFIED).

### 3.7 Not done by this ticket

Rewrite of `ENTERPRISE_AUDIT_BASELINE.json`, any baseline promotion or `--bootstrap-candidate` run (`T-AF2-ENT-BASELINE-PREFREEZE`, ruling Q-RW-04); a new `AuditCategory`; `PT-E03` work (acceptance conftest, debt gate, nightly env); `.github/` and GitHub settings; IIAC validators (`PT-E05`); docs and the FRZ checklist (the human-readable stage-gate-only statement and any FRZ-AUD-03 annotation under DP-06 belong to doc batches); NT-08 of Campaign B (section 9.1).

## Write set

Branch: product commits of this ticket go only to branch `arch-freeze/rw-train` (the integration branch of the ARCH-FREEZE session, stated by that session on 2026-11-16); the worktree and branch authorization is A-02. One path per line below; product paths are separated from scripts, tests and docs.

### Product paths (ACT-N11 sensitive: every one breaks the IIAC activation until the single RB-2 re-pin)

```
intergrax/platform/enterprise_audit/impact_map.py
intergrax/platform/enterprise_audit/changed_since.py
intergrax/platform/enterprise_audit/runner.py
intergrax/platform/enterprise_audit/digests.py
intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json
intergrax/platform/enterprise_audit/data/schemas/frz_coverage.v1.schema.json
intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json
intergrax/platform/enterprise_audit/manifest.py
intergrax/platform/enterprise_audit/evidence.py
```

Parts: `impact_map.py` A, B and the wave-diff mapping (3.6); `changed_since.py` A, B (not named by the master-plan row); `runner.py` A (notes), C (guard), D; `digests.py` C; the JSON is regenerated by the existing generator, never hand-edited; the schema is C (not named by the master-plan row); the manifest JSON and `manifest.py` are E (named by ruling Q-RW-04); `evidence.py` is E (not named by the ruling, added by the design, C-10); all nine are named in the PRE-ENT L3 production delta list (`tests/qualification/roadmap_replay_x/_pre_ent_l3_prod_delta.py@H:97-110` for the manifest JSON, `digests.py`, `evidence.py`, `impact_map.py`, `manifest.py` and the impact JSON). No new product file; none of the six `MATERIAL_GOVERNANCE_PATHS` is touched (section 9).

### Scripts (not product paths; `Ref:` trailer required)

```
scripts/enterprise_audit/run_enterprise_audit.py
scripts/enterprise_audit/build_impact_map.py
scripts/enterprise_audit/mechanical_readiness_check.py
```

### Tests (not ACT-N11 paths; not on the EBH-7 identity allowlist `scripts/qualification/verify_ebh7_p03_immutable_evidence.py@H:20-37`, so they merge before the pin)

```
tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py
tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py
tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py
tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py
tests/qualification/ent_audit_x/unit/test_wd_mapped_family_checks.py
```

Pinned tests edited by Part E (they pin the count 85, the coverage row count or the old digest literal; static grep at H over `tests/qualification`, `scripts/enterprise_audit`, `intergrax/platform/enterprise_audit`; the list is a lower bound, completeness UNVERIFIED; the ruling names eight, the grep finds twelve test files and one fixture; the two `*_real_85` files in the L3 directory are skipped unless an environment variable is set but still pin 85):

```
tests/qualification/ent_audit_x/integration/test_adversarial_adv01_30.py
tests/qualification/ent_audit_x/integration/test_runner_full_smoke.py
tests/qualification/ent_audit_x/l3/test_l3_clean_checkout_full.py
tests/qualification/ent_audit_x/l3/test_l3_freeze_real_85.py
tests/qualification/ent_audit_x/l3/test_l3_full_real_85.py
tests/qualification/ent_audit_x/unit/test_baseline_manifest_digest_promotion.py
tests/qualification/ent_audit_x/unit/test_frz_aud_w402_neg.py
tests/qualification/ent_audit_x/unit/test_frz_coverage_matrix_complete.py
tests/qualification/ent_audit_x/unit/test_manifest_and_exits.py
tests/qualification/ent_audit_x/_manifest_fixtures.py
tests/qualification/pdc_x/p6_impact/test_p6_impact_map_coverage.py
tests/qualification/pdc_x/p6_impact/test_p6_impact_tier3_root_rule.py
tests/qualification/pdc_x/p8_repair/test_r4_impact_map_classification.py
```

(`l3/test_l3_freeze_real_85.py` and `l3/test_l3_full_real_85.py` are gated by `ENT_AUDIT_RUN_L3_FREEZE` and `ENT_AUDIT_RUN_L3_FULL`; `test_baseline_manifest_digest_promotion.py` holds the literal digest `0fe966e4…691b8b` at line 18; `_manifest_fixtures.py` is a fixture module with `planned = 85`, `executed = 85`.)

The five `test_wd_*` files are new; the directory is part of the required PR job (`enterprise-audit.yml@H:42`). In the pinned files only the expected value is updated; no assertion is removed or loosened. If any other pinned test turns red the ticket stops and asks.

### Docs and governance files

```
governance/tickets/T-IGX-043-I.md
governance/tickets/plans/PLAN_T-IGX-043-I.md
```

No other document. Totals: 9 product + 3 scripts + 18 tests (5 new, 13 pinned and edited) + 2 governance files = 32 files (master-plan row: 4 product files; before ruling Q-RW-04 the draft listed 14).

### Conditional (decided by Q-WD-11 to Q-WD-13; not counted in the totals above)

```
tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_aud_01.py
tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_ctl_01.py
tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_debt_01.py
tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_frz_01.py
tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_hrn_01.py
tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_trc_01.py
scripts/enterprise_audit/qual_wave_lib.py
tests/qualification/ent_audit_x/unit/test_evidence_publication_strict.py
```

The six wrapper files are generated by the existing tool and belong to the `PT-E03` area (Q-WD-13; file naming follows the 85 existing wrappers, UNVERIFIED for these ids); `qual_wave_lib.py` holds `_SEMANTIC_CHECK_COUNT = 85` and `85/85` strings (whether they count manifest checks or wrappers is UNVERIFIED); `test_evidence_publication_strict.py` builds manifests with `checks_planned=85` against the closed-world pin of `evidence.py` and is a candidate for the red-first run. New entrypoint test files for the DEBT, FRZ or CTL checks, if the owner chooses to add them (Q-WD-11), are not listed because their names are not decided.

### Forbidden paths

```
intergrax/platform/enterprise_audit/contracts.py
intergrax/platform/enterprise_audit/modes.py
intergrax/platform/enterprise_audit/baseline.py
intergrax/platform/enterprise_audit/validators.py
intergrax/platform/enterprise_audit/report.py
intergrax/platform/enterprise_audit/report_summary.py
intergrax/platform/enterprise_audit/data/evidence_artifact_catalog.v1.json
tests/qualification/ent_audit_x/ENTERPRISE_AUDIT_BASELINE.json
tests/qualification/ent_audit_x/acceptance/ (except the conditional wrapper files above)
scripts/maintenance/
scripts/governance/
.github/
pyproject.toml
uv.lock
```

`ENTERPRISE_AUDIT_BASELINE.json` is forbidden by ruling Q-RW-04 (the accepted baseline comes from `T-AF2-ENT-BASELINE-PREFREEZE`). Also forbidden: every other file of the engine tree and of `tests/`, docs, the roadmap, the FRZ checklist, IIAC and PDC-X records, `governance/` other than the two own files, and the files of FC-1..FC-4, W-A, W-B and W-F.


## Negative tests

Each test names the configuration and the expected refusal. "RED at B" (listed after the table) means the test fails at the baseline tree (tests of new symbols fail by import error, which the auditor's mutation check complements by re-introducing the old behaviour); "GUARD" means green at B and after, protecting against over-correction. Selection tests monkeypatch `changed_since.git_changed_files` and `assert_git_ancestor`, so at B they fail by assertion, not by import. 41 test functions (A 16, B 4, C 9, D 6, E 6; A01 and D03 are parametrized), of which 26 are RED at B and 15 are GUARDs.

| ID | Configuration | Expected refusal | Node id |
|---|---|---|---|
| A01 | changed = `<prefix>/zz_wd_new_pkg/mod.py`, shipped manifest and map | selection == FULL_SET (every manifest check), `escalated_to_full` | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a01_new_product_package_selects_full` (parametrized: base id) |
| A02 | `intergrax/rag/retrieval/retrieval_service.py` | floor ∪ mapped ids, not FULL, ⊆ manifest | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a02_mapped_product_path_stays_narrow` |
| A03 | `intergrax/platform/enterprise_audit/runner.py`, `scripts/enterprise_audit/build_impact_map.py` | FULL_SET; no id outside the manifest | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a03_engine_file_with_only_legacy_ids_selects_full` |
| A04 | `intergrax/platform/enterprise_audit/zz_new.py` | FULL_SET, no raise (inverts if Q-WD-02 keeps the raise) | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a04_unknown_engine_tree_file_selects_full` |
| A05 | `intergrax\zz_wd_new_pkg\mod.py` | FULL_SET | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a05_backslash_path_is_normalised` |
| A06 | scratch git repo, `git mv` out of an unmapped path | `git_changed_files` returns both paths | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a06_rename_reports_old_and_new_path` |
| A07 | scratch git repo, delete `intergrax/zz_old/a.py` | FULL_SET | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a07_deleted_unmapped_product_path_escalates` |
| A08 | `docs/x.md`, `tests/unit/zz.py`, `scripts/maintenance/zz.py`, `.github/workflows/zz.yml`, `governance/tickets/T.md` | floor only | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a08_non_product_paths_do_not_escalate` |
| A09 | changed = empty | floor, no escalation (`test_adv26` shape) | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a09_empty_diff_is_floor_not_full` |
| A10 | synthetic map `{"intergrax/zz_pkg": ["C999"]}` | path reported unmapped | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a10_prefix_with_only_unknown_ids_counts_unmapped` |
| A11 | JSON without `path_prefix_to_check_ids`; `schema_version` 2; non-list ids | `ValueError` | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a11_malformed_impact_map_fails_closed` |
| A12 | one probe path in every package directory (L1 sub-packages of `intergrax/ agents/ applications/` plus L2 sub-packages of `intergrax/`, discovered at test time; the unit of the alpha probe `pp_nt08_census_uncovered_packages`, 153 uncovered at B) | each probe is mapped (real ids) or FULL; none floor-only | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a12_no_floor_only_outcome_for_any_product_package` |
| A13 | dry-run runner with `selection_notes` | `limitations` contains the escalation line and the first unmapped path | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a13_escalation_is_disclosed` |
| A14 | shipped manifest | FULL_SET == `checks_for_mode(FULL)`, size equal to the manifest size | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a14_full_set_is_the_full_mode_plan` |
| A15 | AST read of `scripts/governance/iiac_common.py` | tuples equal | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a15_product_prefix_parity_with_iiac_common` |
| A16 | every tracked path (`git ls-files`) | new selection ⊇ floor ∪ (prefix ids ∩ manifest) ∪ entrypoint-touched | `tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py::test_wd_a16_selection_never_shrinks_versus_prefix_union` |
| B01 | all files under `tests/qualification/evol_x`, `prefixes=` shipped map | `unmapped_evol_changed_paths == ∅` | `tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py::test_wd_b01_every_tracked_evol_x_file_is_mapped` |
| B02 | `_constants.py`, `conftest.py`, six `adapters/*` | selection ⊇ the five evol ids, no raise | `tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py::test_wd_b02_the_eight_dossier_paths_select_the_evol_checks` |
| B03 | `tests/qualification/evol_x/zz_new/x.py` | `ValueError` | `tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py::test_wd_b03_unknown_new_evol_x_directory_still_raises` |
| B04 | regenerated JSON | equals the generator output, bytes included | `tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py::test_wd_b04_shipped_map_still_equals_generator` |
| C01 | shipped manifest | `frz_default_fallback_check_ids == ()`; the nine families and the six new check ids map to `FRZ-<FAM>-*` | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py::test_wd_c01_no_check_uses_the_default_fallthrough` |
| C02 | synthetic `ENT-A-ZZZ-01` | `strict=True` raises naming it; `strict=False` keeps the old default | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py::test_wd_c02_strict_artifact_refuses_unattributed_check` |
| C03 | runner with one renamed check, certifying | BLOCKED report, finding `FRZ:COVERAGE_DEFAULT_FALLBACK`, exit VIOLATION; dry-run not blocked | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py::test_wd_c03_certifying_run_is_blocked_on_fallthrough` |
| C04 | shipped manifest | every one of the six families has a mapped check; the block lists, per family, only the criteria the check does not exercise, with owner stage and basis (content per the check design, Q-WD-11) | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py::test_wd_c04_artifact_declares_only_what_the_new_checks_do_not_exercise` |
| C05 | families parsed from the checklist | every family is attributed or declared (fails on an undeclared new family) | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py::test_wd_c05_checklist_families_are_attributed_or_declared` |
| C06 | artefact with block; block missing owner; legacy artefact without block | valid; invalid; valid | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py::test_wd_c06_schema_accepts_block_rejects_malformed_and_accepts_legacy` |
| C07 | shipped manifest | rows == manifest size (91 after Part E), same row keys | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py::test_wd_c07_rows_and_row_count_unchanged` |
| C08 | synthetic baseline objects, one holding the digest of the shipped manifest and one an older digest | `assert_baseline_manifest_digest_current` accepts the first and raises on the second (the refusal behind `BASELINE:MANIFEST_DIGEST`; independent of the shipped baseline file, so it holds before and after the baseline re-promotion) | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py::test_wd_c08_manifest_digest_drift_is_refused` |
| C09 | `ENT-A-DUP-01..03` | `("FRZ-AUD-03",)` through an explicit row (follows Q-WD-05) | `tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py::test_wd_c09_dup_checks_are_explicit_aggregator_only` |
| D01 | non-certifying FAST, complete floor, only TRACKED_DEBT | promoted; limitation line with `status_promoted_from=PARTIAL` and the count | `tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py::test_wd_d01_promoted_run_discloses_status_and_debt_count` |
| D02 | as D01 with an incomplete floor | stays PARTIAL, exit INCOMPLETE (follows Q-WD-06) | `tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py::test_wd_d02_incomplete_floor_is_not_promoted` |
| D03 | only TRACKED_DEBT, non-certifying | not promoted | `tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py::test_wd_d03_other_modes_are_never_promoted` (parametrized: base id) |
| D04 | certifying FAST, only TRACKED_DEBT | not promoted | `tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py::test_wd_d04_certifying_fast_is_never_promoted` |
| D05 | TRACKED_DEBT plus ERROR | not promoted | `tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py::test_wd_d05_mixed_severities_are_not_promoted` |
| D06 | promoted report JSON | `validate_report_semantic_consistency` empty | `tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py::test_wd_d06_promoted_report_passes_semantic_validation` |
| E01 | shipped manifest | exactly 91 checks, ids unique; `ENT-A-AUD-01`, `ENT-A-CTL-01`, `ENT-A-DEBT-01`, `ENT-A-FRZ-01`, `ENT-A-HRN-01`, `ENT-A-TRC-01` present with category `architecture`, `depends_on` `ENT-A-INV-01`, FAST not among the modes | `tests/qualification/ent_audit_x/unit/test_wd_mapped_family_checks.py::test_wd_e01_manifest_holds_the_six_family_checks` |
| E02 | entrypoints of the six new checks | each entrypoint file exists in the tree, the kind is valid, the pytest node id is collectable | `tests/qualification/ent_audit_x/unit/test_wd_mapped_family_checks.py::test_wd_e02_new_check_entrypoints_resolve` |
| E03 | synthetic copy of the manifest with one check more, and with one less | `load_manifest_from_path` raises both times (the count pin refuses both directions) | `tests/qualification/ent_audit_x/unit/test_wd_mapped_family_checks.py::test_wd_e03_count_pin_refuses_one_more_and_one_less` |
| E04 | the six new ids through `_frz_for_check` and the coverage artefact | each maps to `FRZ-<FAM>-*`, none uses the fall-through, `row_count` equals the manifest size | `tests/qualification/ent_audit_x/unit/test_wd_mapped_family_checks.py::test_wd_e04_new_checks_are_attributed_to_their_family` |
| E05 | the entrypoint paths of the six new checks, shipped impact map | each path selects its own check id; the JSON equals the generator output | `tests/qualification/ent_audit_x/unit/test_wd_mapped_family_checks.py::test_wd_e05_new_checks_are_reachable_through_the_impact_map` |
| E06 | certifying FULL and FREEZE plan with a synthetic accepted baseline holding the pre-change digest, and one holding the current digest | the first yields finding `BASELINE:MANIFEST_DIGEST` (BLOCKER); the second does not | `tests/qualification/ent_audit_x/unit/test_wd_mapped_family_checks.py::test_wd_e06_stale_baseline_digest_blocks_certifying_runs` |

RED at B (26): A01, A03, A04, A05, A06, A07, A10, A11, A12, A13, A15, B01, B02, C01, C02, C03, C04, C05, C06, C09, D01, D02, E01, E02, E04, E05. GUARD, green before and after (15): A02, A08, A09, A14, A16, B03, B04, C07, C08, D03, D04, D05, D06, E03, E06.

A12 and B01 are the completeness tests the dossier asks for; A01 and A12 close the alpha-confirmed A6-007 (alpha at B: new package floor-only, 152 of 153 uncovered packages floor-only) (A6-007: census completeness; A6-003: the evol_x tree); B01 and B02 close the B-2-alpha-confirmed X-A6-003 (9 of 12 stage ranges exit 1 at B, run R25c); E01 to E06 are the checks of Part E (E03 and E06 are guards of behaviour that exists today and must not weaken). A16 is the monotonicity property that the PDC-X addendum-4 impact-map probes checked by hand ("0 of 19472 tracked files lose impact ids", FACT_CHECK F-22): selection may only grow.

## 6. Regression scope

`regression_nodes.txt` in this directory lists 98 Tier 1 node ids in 27 files and 47 Tier 2 node ids in 7 files; each id was read with `git show` at B and at H, and every file is byte-identical at both (FACT_CHECK F-23). Tier 1 covers the impact-map and selection tests (ENT unit/integration, `pdc_x/p6_impact`, `p8_repair`, `xd`), FRZ coverage and schema validation, report and status semantics, FAST floor and modes, baseline promotion refusals, runner dry-run and the CLI exit codes. Tier 2 are the checks the map itself names for `impact_map.py` and its JSON (`ENT-A-EVOL-01..03` entrypoints in `tests/qualification/evol_x/gates`) and the L3 directory, which the required PR job already runs. Six ids are parametrized: pass the base id. The 13 pinned files of the write set are not in the unedited Tier 1 list; they run in their edited form with it (section 7, command 3). The two PDC tests of 3.6 are in Tier 1 (they are the reason for the last commit).

Not in the scope, deliberately: `tests/qualification/ent_audit_x/acceptance` (semantic wrappers, `PT-E03`); the EBH-7 and ROADMAP-REPLAY-X suites (`PT-Q04/Q05`).

## 7. Test commands (one process at a time; no diagnostic or broad suite; no Docker, no network)

Take the lock first (`mkdir D:/Projekty/_wt/_locks/pytest.lock`, `rmdir` immediately after); logs under `.tmp/session/eqo-a-remediation/IMPL/WD/`.

1. Red-first at B (in a scratch worktree at B, authorization A-04): the five new files with `-p no:cacheprovider`; the 26 RED tests must fail, the 15 GUARD tests must pass.
2. Same command at the fix commit: all green.
3. Regression: one `uv run pytest <Tier 1 ids> <the 13 edited pinned files> -p no:cacheprovider`.
4. `uv run python -m scripts.governance.igx_gov ticket` before push.
5. Not part of the approvable contract: Tier 2 (once, by the auditor at the wave SHA), the 12-SHA diagnostic (A-05), one dry FULL coverage artefact diff (A-06).

## 8. Order of work inside the ticket

1. Preflight: approval line (wave packet, OD-AF-22); `PT-WA` and `PT-WB` merged into the train (RW-PLAN lane 3); the exception ruling for `BASELINE:MANIFEST_DIGEST` recorded (9.3); worktree outside the repository (`<parent>/_wt/intergrax/t-igx-043-i`); M1; `PT-E03` merged; read the Campaign B alpha results R13/R25 (red baseline of A6-007/A6-003).
2. Write the five test files; run them red at B (RED/GUARD split as in the section Negative tests). Commit tests first.
3. Part A (impact_map.py, changed_since.py), then Part B (rule table, JSON regenerated), run A/B tests green.
4. Part C (digests.py, schema, runner guard), then Part D (runner promotion helper, notes), scripts S1/S2. Then Part E in its own commit: manifest JSON, count constant, `evidence.py`, readiness script, the 13 pinned files edited in the same commit so that no commit leaves them red, impact JSON regenerated; the manifest digest changes at this commit (section 9.3).
5. Regression Tier 1; quality gate (`igx_gov ticket`); hand-off note: list of every file changed in the six product paths and the generator command line.
6. Last: wave-diff mapping (3.6) on the integrated wave branch after FC-1..FC-4, W-A, W-B and W-F are merged into it; regenerate the JSON; rerun A12, the two PDC tests and Tier 1.
7. Lifecycle `quality_gate`. The independent audit runs at the wave SHA (section 11); the implementer records no PASS.

Merge position: files are disjoint from every other wave ticket (`TICKETS_DRAFT.md` lists no other ticket on `enterprise_audit`); only step 6 depends on the others.

## 9. ACT-N11 / IIAC / EBH-7 / baselines

- P1..P9 (the six paths of the master-plan set plus the manifest JSON, `manifest.py` and `evidence.py`) match `PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")` (`iiac_common.py@H:22`); any change after the bound product tree fails activation with `ACT-N11 product path changed after the audited SHA` (`iiac_activation_gate.py@H:576`) until the single RB-2 re-pin (PT-Q03; DP-58). The activation record at H is `IN_FORCE`, `corrective_implementation_sha = 969cbd964c679d563b02e55f065b452d46fdf90b`, `audit_execution_sha = a73d15ab341b30e9748aad03894f6faa75ad4e09` (`IIAC_POLICY_ACTIVATION_RECORD.json@H:4,8,9`). The ticket proposes no re-pin.
- Material governance paths (`iiac_activation_gate.py@H:40-49`): none of the six is touched, so the "stale audit after material governance change" rule (`@H:569`) is not triggered by this ticket; `PT-E05` (IIAC validators) is separate.
- ENT baseline: `ENTERPRISE_AUDIT_BASELINE.json` (accepted baseline `480550d80b2515fc8cbd2a0ba6f79d1ee1c44541`, `check_manifest_digest 0fe966e4…`) is not edited (ruling Q-RW-04); the manifest is edited (Part E), so the digest comparison `assert_baseline_manifest_digest_current` no longer holds against the accepted baseline at the wave head: certifying FULL and FREEZE runs report `BASELINE:MANIFEST_DIGEST` until `T-AF2-ENT-BASELINE-PREFREEZE` re-promotes (section 9.3; the refusal itself is guarded by C08 and E06). The ENT baseline chain (`--bootstrap-candidate`, delegated promotion) belongs to `T-AF2-ENT-BASELINE-PREFREEZE` (W5) and DS-ADM, which assert that no product path changed after G2 (`MEP-AF2_v2.1_final.md` rule PD2-03): W-D lands in the RW train before G2, which is what that rule needs. This ticket runs no `promote`, no `--bootstrap-*`, no `--evidence-bundle` certifying run.
- EBH-7: the nine product paths belong to the single final pin's product delta (DP-61); the new and edited test files and the three scripts are non-allowlisted paths and merge before the pin.
- ROADMAP-REPLAY-X (DP-60): the product paths are already named in the PRE-ENT L3 production delta list (`tests/qualification/roadmap_replay_x/_pre_ent_l3_prod_delta.py@H:91-115`; the manifest JSON, `manifest.py` and `evidence.py` at lines 98, 108, 110); the wave's new delta must be dispositioned by `PT-Q04/Q05`, with the nine paths as input (nothing is edited here).
- EVOL-X: the checks `ENT-A-EVOL-01..03` are the ones the map selects for `impact_map.py` and its JSON; EVOL-X's own gate tests do not call the engine functions (static grep, F-21). Which EVOL-X records pin digests of these two files is UNVERIFIED.
- `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` lists engine and test paths by name (generated 2026-11-05, 17510 governed paths); whether five new test files need a matrix update is a question for the RB-2 session (UNVERIFIED).

### 9.1 Overlaps and boundaries

| With | Boundary |
|---|---|
| `PT-E03` (T-ENF-3, non-product) | owns `tests/qualification/ent_audit_x/acceptance/` (conftest hook, `ENT_AUDIT_X_CERTIFYING`), `scripts/maintenance/check_architecture_debt_register.py` and the semantic-wrapper automation (A6-001, A6-006, DP-66 last clause). W-D edits none of these and closes none of those findings; W-D needs it merged so the wave audit can run the 86 wrappers in a certifying run on the new engine |
| `PT-G01` (non-product gates) | owns EBH-3 scan extension, seam ledger, construction/serving gates under `scripts/maintenance`, `governance/architecture`, `tests/unit/architecture`. No shared file. W-D adds no gate and G01 must not register new checks in the ENT manifest (after this ticket the manifest holds 91 checks; PD2-03: gates added after FCS live outside the manifest) |
| `PT-T01` | its Campaign B test NT-08 probes `selected_check_ids_from_impact` (`WP-11/PLAN.md` CK-17), which keeps its old semantics; the escalation lives in the public selector. The omega run W13 and PT-T01's tracked version must call `select_checks_for_changed_since` (contradiction C-07); W-D owns the tracked tests A01..A16, PT-T01 must not add a second copy |
| wave audit `PT-Q02` / `AUD-ENF-3` | one independent session at the exact RW head performs AUD-ENF-3 as part of the wave audit (`DAG.md` row `PT-WD`); section 11 |
| `PT-Q04/Q05` (ROADMAP-REPLAY-X, DUP-X) | consume the six product paths as delta input; no edit here |
| `PT-C01` (Campaign C) | the mapped checks for the six families moved into this ticket (ruling Q-RW-04, `DAG.md` row `PT-C01`); Campaign C keeps the independent re-judging sample |
| `T-AF2-ENT-BASELINE-PREFREEZE` (W5, `T-IGX-045-A`) | owns the rewrite of `ENTERPRISE_AUDIT_BASELINE.json` against the new manifest digest (delegated chain, `MEP-AF2_v2.1_final.md` section I) and asserts that no product path changed after G2; this ticket's manifest is its input and the source of the red window in 9.3 |
| `PT-WB`, FC-1..FC-4, `PT-WA`, `PT-WF` | no shared files; they hand W-D the touched-module list for 3.6 |
| `PT-E05`, `PT-D08` (now depend on `SCOPE`) | `PT-E05` edits the IIAC validators under `scripts/governance/` (material paths), `PT-D08` two product-path docs; no shared file with this ticket; merge order inside the RW train is free except the last W-D commit (section 8, step 6) |

### 9.2 Why this ticket promotes no baseline

Promotion is a separate, Operator-owned act (`promote_accepted_baseline` refuses without an Operator decision record and a closed evidence bundle: `test_promotion_requires_operator_records_path`, `test_promote_rejects_agent_only_record`; DS-ADM in the master plan). The ticket does not edit the baseline file and runs no promotion or bootstrap command; it does change what the baseline binds (the manifest digest), which is why section 9.3 exists. The stage-gate-only declaration changes the machine-readable coverage artefact only; no FRZ status moves and nothing is certified.

### 9.3 Expected-red register: ENT runs that go red because of the manifest digest change (input for the known-red register)

Cause: Part E changes the manifest, so `manifest_digest()` no longer equals `check_manifest_digest` in `ENTERPRISE_AUDIT_BASELINE.json` (`0fe966e4…691b8b`); `runner.py@B:220-234` adds the BLOCKER finding `BASELINE:MANIFEST_DIGEST` (check `ENT-A-BND-01`) for certifying FULL and FREEZE runs with a loaded baseline. Window: from the Part E commit until `T-AF2-ENT-BASELINE-PREFREEZE` re-promotes. Statements below are read from code and workflows at H and are not shown by a run (UNVERIFIED until the red-first and wave runs).

| # | Run | Expected state in the window | Owner | Cleared by |
|---|---|---|---|---|
| 1 | certifying `run_enterprise_audit.py --mode FULL` or `FREEZE` (no `--non-certifying`) against the committed baseline | finding `BASELINE:MANIFEST_DIGEST`, aggregate not PASS, non-zero exit | ENT-AUDIT-X owner; chain executor `T-AF2-ENT-BASELINE-PREFREEZE` (W5, `T-IGX-045-A`) | baseline re-promotion on the final manifest |
| 2 | `enterprise-audit-nightly.yml` second step (FULL by default, FREEZE by dispatch; runs without `--non-certifying`) | red from the first scheduled run after the wave merge; its first step (`pytest tests/qualification/ent_audit_x`) is expected green after Part E | ENT-AUDIT-X owner | same re-promotion |
| 3 | wave audit PT-Q02 / AUD-ENF-3 certifying FULL or FREEZE replay at the RW SHA (W4, before the baseline chain at W5) | the auditor runs it non-certifying, or reads the finding as expected and says so in the report | wave auditor session | not cleared inside PT-Q02; PT-Q13 (FULL at M3) runs after the re-promotion |
| 4 | env-gated L3 tests `l3/test_l3_freeze_real_85.py` (`ENT_AUDIT_RUN_L3_FREEZE=1`) and `l3/test_l3_full_real_85.py` (`ENT_AUDIT_RUN_L3_FULL=1`) when someone sets the variables | certifying runs, same finding; skipped by default | ENT-AUDIT-X owner | re-promotion (their count pins are edited by this ticket) |
| 5 | semantic gate matrix `integration/test_semantic_gate_execution.py` (`ENT_AUDIT_X_GATE_MATRIX=1`) and the semantic wave in `qual_wave_lib.py` | red or count mismatch if the six new checks have no acceptance wrapper while the manifest has 91 | `PT-E03` owner with the ENT-AUDIT-X owner (Q-WD-13) | wrappers generated, or the count rule restated |
| 6 | any pinned test outside the 13 edited files, in a branch or worktree that pins 85 or the digest literal (static grep at H found none other; completeness UNVERIFIED) | red after merge with the RW train | the implementer of that ticket; auditor greps at the wave SHA | edit of the expected value in that ticket |
| 7 | validators that compare an older report's `check_manifest_digest` with the live manifest (`vbn.py@B:78-83`, `validate_domain_report(expected_manifest_digest=...)`) on historical bundles such as `docs/.../ent_audit_x/b6_independent_handoff/` | possible `check_manifest_digest stale` errors if any test or script validates those bundles against the live manifest; no such caller was found in the lists above (UNVERIFIED) | auditor | none needed if no caller; else the caller is pinned to the bundle's own digest |
| 8 | the two pinned PDC wave-diff tests of 3.6 (`test_p6_impact_map_coverage.py::test_every_pdc_touched_product_module_maps_to_at_least_one_check`, the xd12 twin) | latent red at the train head by static replication: 5 touched `agents/` modules unmapped (3.6); no workflow job at the train head runs `tests/qualification/pdc_x`, so no CI job is red, a local or audit run would be (UNVERIFIED, not run) | `PT-WD` (last commit, 3.6); introduced by `PT-WA` | the last commit of this ticket |

Precondition from RW-PLAN v13 (RWA-03, section (c) table 2): `BASELINE:MANIFEST_DIGEST` is introduced by the train, so an explicit exception ruling (orchestrator or Operator) is recorded before the `PT-WD` PR merges into the train. RW-PLAN table 2 names "ENT-AUDIT-X runs (enterprise-audit.yml jobs)"; read from the workflows, the PR jobs of `enterprise-audit.yml` are not expected red (FAST non-certifying; the unit, integration and l3 directories are edited by this ticket to the new count), and the certifying runs are rows 1 to 4 (nightly workflow `enterprise-audit-nightly.yml` and operator CLI runs). The table 2 wording should name rows 1 to 4 (UNVERIFIED until the runs).

Not expected red, for completeness (read from the code): the FAST non-certifying run of the required PR check `ent-audit-fast` (the digest comparison runs only for FULL and FREEZE certifying runs, `runner.py@B:220`; the new checks are not in FAST), CHANGED-SINCE runs, and every `--non-certifying` or dry-run call.

## 10. ARCH-FREEZE impact and what closes it

| Finding | Class | Closure condition (none is asserted by this plan) |
|---|---|---|
| A6-007 | CONDITIONS | A01..A16 green at the wave SHA, RED at B; the dossier's census completeness (A12); mutation check; independent audit (AUD-ENF-3); Campaign B W13 at FCS (`PT-B01/B1W/BA` follow-on of the master plan); FRZ-EVOL-06 / FRZ-AUD-06 stay as recorded until the Operator accepts the audit |
| A6-004 | CONDITIONS | six mapped ENT checks exist and are attributed (E01, E02, E04, E05); the coverage artefact declares the criteria they do not exercise (C04, C05); audit confirms that each check exercises the property the design names; the 42 PASS rows keep the stage-gate evidence they cite and no FRZ row changes status because a check was added; `FRZ-AUD-03` PASS must not be read as covering the six families |
| A6-005 | NONE (dossier), CONDITIONS (master-plan effective) | C01, C02, C03 green; every manifest row (91 after Part E) attributed explicitly; pinned digests of `frz_coverage` in published bundles are historical files, not regenerated; audit |
| A6-003 | NONE | B01, B02 green; the 12-SHA diagnostic exits 0 for every cert SHA (computability, not narrowness); audit |
| A6-008 | NONE | D01..D06 green; audit |

## 11. Independent audit design (AUD-ENF-3, inside PT-Q02)

Exact SHA: the wave head `RW` frozen after all `FC-*`, `W-*` merges; any later product change invalidates it. Auditor: a separate non-implementing session that did not review the implementation and is not an IIAC signing or activation session, using `/intergrax-independent-audit`; the implementer records no PASS. The auditor replays: (1) every new test at B (the 26 RED fail, the 15 GUARD pass) and at the wave SHA (all green); (2) one mutation per part: drop the product-escalation call, drop the `--no-renames` flag, re-allow legacy ids as coverage, remove one attribution row, remove the incomplete guard, return `strict=False` for certifying runs, hand-edit one JSON key; each must turn at least one test red; (3) behavioural probes on the real repository: the planner over `B..wave head` and over the 12 cert SHAs (`build_impact_map.py`, bounded), one probe path per top-level package, the dry FULL coverage artefact diff against B (exactly the 27 rows of the nine re-attributed categories change and the three DUP rows keep their values, plus the `stage_gate_only` block, `row_count` 91 with the six added rows, the artefact validates against the changed schema and the old artefacts still validate); (4) Tier 1 and Tier 2 regression at the wave SHA; (5) ACT-N11 reconciliation: the product-path files of the commits with trailer `Ref: governance/tickets/T-IGX-043-I.md` equal P1..P6, no file of the forbidden list changed, `sha256` of `ENTERPRISE_AUDIT_BASELINE.json` equals its value at B, the manifest JSON differs from B by exactly the six added checks, and the recomputed manifest digest is recorded; the register of 9.3 matches the observed runs; (6) selection is monotone: A16 plus a replay of the addendum-4 per-file old/new probe over all tracked files (`iiac_evidence/PDC-X/P7-00/addendum_4/adversarial_scripts/addendum_4_impact_map_probes.py` is the prior-art script); (7) the wave-diff mapping list of 3.6 is attached and each group's entry is judged. Label and scope are the auditor's decision; this plan states none.

## 12. Risks

- R1 FULL becomes the common CHANGED-SINCE outcome: 23 % of the tracked product files and 64 of the 93 second-level groups (a synthetic file at the group root) are unmapped at B. CHANGED-SINCE is not used in any workflow (`git grep` over `.github`, FACT_CHECK F-19); only the operator scripts `audit.sh`/`audit.bat` and `build_impact_map.py` call it. Cost of a FULL run is not measured (A-06); the PR job already runs the unit, integration and L3 directories plus a FAST CLI in a median 760 s (`WP-7/PLAN.md` section 2).
- R2 The two PDC tests (3.6) are the sharpest dependency; if the integrated wave head still has an unmapped touched module, they fail. Mitigation: last-commit rule and test A12/the two tests in every run.
- R3 Behaviour change for the engine tree (raise to FULL) and for the incomplete-floor promotion are decision-dependent (Q-WD-02, Q-WD-06).
- R4 `frz_coverage.v1.json` content changes for 27 rows (nine categories, three checks each); published bundles are historical files and untouched; any consumer that compares live artefacts to a pinned digest would differ (none found in `tests`, `scripts`, `.github`; UNVERIFIED for external records).
- R5 Closed-world inventories (IIAC impact matrix) may reject changed or new paths (UNVERIFIED).
- R6 New tests under `tests/qualification/ent_audit_x/unit` run in the required PR job; a flaky filesystem or git-history dependence (A06/A07 use scratch repositories, A12/A16 walk the tree and `git ls-files`) would add noise. Mitigation: no network, no shallow-clone assumption beyond what the existing xd12 test already needs.
- R7 JSON growth in step 6 (about 85 lines per explicit-FULL group); readability of the generator tables.
- R8 `git diff --no-renames` changes which paths appear for rename-heavy ranges (more entries, not fewer).
- R9 Part E changes the manifest digest: the red window of 9.3 (nightly, certifying runs) lasts until the baseline re-promotion; an ENT run that is red for this reason is expected, but a reader must not read it as a regression of Parts A to D.
- R10 No check design exists for the six families (Q-WD-11): a check that asserts too little adds a coverage claim without evidence; one for an OPEN criterion fails every FULL run. The independent audit judges each check's property against the criteria it is attributed to.
- R11 The pinned-test list is a lower bound; a missed pin turns a required PR job red after the manifest commit.

## 13. Rollback

Revert the wave merge (one commit train); the IIAC rebind must then be redone (a revert alone does not restore the activation, AR-42). The manifest digest returns to `0fe966e4…` on revert and the baseline file is not changed by this ticket, so none is unwound; if `T-AF2-ENT-BASELINE-PREFREEZE` has already re-promoted against the new manifest, that baseline must be re-promoted again (its owner). Part C's schema addition is additive, so old artefacts stay valid either way. If only Part A proves unsafe, revert it before the audit SHA is frozen, never after (a later product change invalidates the audit). Files written under `.tmp/` by diagnostics are not part of the repository.

## 14. Effort (planning estimate, not measured)

Part A M; Part B S; Part C M; Part D S; scripts S; wave-diff mapping S to M (size known only at the wave head); tests M; audit hand-off S. Part E M to L (check design is owner work and is not estimated here; 13 pinned tests edited). Total M to L (the master-plan row lists four product files and no effort; it understates the write set, C-01).

## 15. Contradictions with the master plan, WP-7 and the dossier

C-01 The master-plan row `PT-WD` names four paths; the design needs six product paths (adds `changed_since.py`, which holds the selection and the evol check, and `frz_coverage.v1.schema.json`, which forbids extra properties) plus two scripts. C-02 The row, the regenerated DAG and `MEP-AF2_v2.1_final.md` (line 40, line 463) say new mapped ENT checks for the six families land in `PT-WD` before G2; DP-66 and the dossier option said declaration now, mapping assessed in Campaign C: resolved by ruling Q-RW-04 (the checks land here; DP-66's declaration stays for the criteria they do not exercise). C-03 WP-7 and the register treat A6-003 as a data fix; the evol check ignores the impact JSON (`changed_since.py@B:83-100`), so code is needed. C-04 The dossier says A6-005 needs "mapping data, no code change"; the mapping is hard-coded in `digests.py` (WP-7 already corrected this). C-05 WP-7 proposes `status_promoted_from` in the report "header"; this plan puts it in the existing `limitations` block (no schema change) unless Q-WD-06 option B2 is answered otherwise. C-06 The dossier's 60 criteria / 42 PASS rows are correct only when the aggregator FRZ-AUD-03 is excluded (the families hold 61 and 43 rows with it, F-09). C-07 WP-11 designed NT-08 against `selected_check_ids_from_impact`, which this design leaves unchanged; the executed alpha probe already judges the production selector `select_checks_for_changed_since` (RESULTS_B1_ALPHA, probe docstring), so the tracked and omega variants keep that target. The alpha census (153 uncovered packages at L1+L2) and this plan's earlier count (64 of 93 second-level groups) use different units; A12 now uses the probe's unit. C-08 WP-7 says CHANGED-SINCE selection ids are manifest ids; the selector returns legacy ids too (86 ids against 85 checks, dossier `A6/notes_parent.md`), and 12 map entries are legacy-only. C-09 PDC status text vs ladder (section 2). C-10 Ruling Q-RW-04 names the manifest, `manifest.py` and eight pinned tests; the design finds `evidence.py` (three literal `85` closed-world pins), `mechanical_readiness_check.py` and, by static grep at H, twelve test files plus one fixture that pin the count or the digest literal. C-11 The ruling excludes `ENTERPRISE_AUDIT_BASELINE.json` from the train while `mechanical_readiness_check.py` and the L3 tests still read it; the red window of 9.3 is the consequence, owned by `T-AF2-ENT-BASELINE-PREFREEZE`.

## 16. Out-of-scope observations (recorded, not fixed here)

(a) `_validate_manifest` hard-codes 85 (`manifest.py@B:57-62`); adding any ENT check means editing it, the manifest digest and the baseline chain (handled by Part E and 9.3 under ruling Q-RW-04). (b) `test_every_check_maps_to_frz_or_sys_inv` and `test_frz_aud_03_unmapped_neg.py` test a local copy of the rule (`pg_id` or `TEN-`/`VBN-`), not `_frz_for_check`; they do not detect fall-through. (c) `ent-audit-fast` runs the real FAST CLI, so A6-008 and the incomplete-floor case touch the live required check once Stage 1 exists. (d) `impact_map.py@B:230-251` duplicates the parsing of the two loaders (folded into Part A). (e) `scripts/qualification/pdc_x_wbs.py@H:434-435,537-538,940-941` carries WBS text naming `impact_map.py` and the JSON (records only). (f) 150 of the 2043 unmapped product files at B are not code (97 `.md`, 21 `.toml`, 16 `.gitkeep`, 10 `.ipynb`, 6 `.json`): Q-WD-03.
