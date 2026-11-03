# ENT-AUDIT-X — post-implementation final closure master plan

**Disposition:** `ENT-AUDIT-X — CURRENT-HEAD COMPLETE CLOSURE EXECUTION PLAN / READY FOR AGENT IMPLEMENTATION`

**CURRENT_DOCUMENTATION_HEAD:** `c6d600df56f3fedb85bcdce12ba508ea8274ccd4`

**Audited implementation baseline (IMPLEMENTATION_SHA):** `d4974efc255bf6ff6ca96e00fb084293ea685d66`

**Prior PLAN_PUBLICATION_SHA:** `87e185dd3bcce45f107149b4b3691c3ede614c8a`

**POST_IMPLEMENTATION_SSOT_REPAIR_PUBLICATION_SHA:** `8efac05b4ca18a31a754cf003851f5bf24725f76`

**Historical TEST_EXECUTION_SHA (not current qualification):** `0cee41a88274a53d14bb3558752d55bdde0d0b01` — log @ older wave; **do not** treat as proof @ `d4974efc255bf6ff6ca96e00fb084293ea685d66`.

**Current runtime qualification:** OPEN — pin new **TEST_EXECUTION_SHA** only after certifying FULL/L3 @ implementation baseline.

**Mode:** execution SSOT — no B6 PASS / FRZ CLOSED / ARCH-FREEZE / ENT stage CLOSED / OD APPROVED by agent.

Canonical scope: [corrective plan](./ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md) C001–C064; C065–C075 Appendix B.

### W-0 substeps (documentation)

| Substep | Wave | Scope | Deps |
| --- | --- | --- | --- |
| C075-S1 | W-0 | Current-HEAD inventory: git ancestry, PL-07 collect-only, ladder append-only row, disposition | — |
| C075-S2 | W-G | Final SSOT reconcile: Matrix C/E, §14 DAG, §18 validator PASS, annex sync | C041,C048,C055 |

### FP-01..FP-10

| FP | Resolution @ audited implementation baseline |
| --- | --- |
| FP-01 | Single execution SSOT; waves W-0..W-G + substeps. |
| FP-02 | C051+C074 in W-D before Wave F qual. |
| FP-03 | Matrix C 75/75 with owner, deps, pos/neg tests. |
| FP-04 | Impl 4·56·15; Qual 2·73 @ `d4974efc255bf6ff6ca96e00fb084293ea685d66`. |
| FP-05 | §14 dependency table + topological order (wave-safe). |
| FP-06 | §13 agent commit+push; operator OD; admin OD-ENT-02; B6 external. |
| FP-07 | `development` protection HTTP 404 @ documentation HEAD. |
| FP-08 | Matrices E/F/G/D/A complete; Matrix E from manifest+Annex C. |
| FP-09 | C066 after TES artifacts; C051/C074 infra pre Wave F. |
| FP-10 | §18 PASS only when `validate_post_implementation_master_plan.py` exit 0. |

### PL-08 — ECAP (canonical)

Implementation waves (W-A..W-E code) → agent **commit+push** → clean checkout → pin **TEST_EXECUTION_SHA** → L1/L2 → real **85/85** semantic gates (zero skip) → certifying **FULL** → **ADV31** @ TES → first **FREEZE candidate** → **EVIDENCE_PUBLICATION_SHA (B5)** → independent **Protocol 03 B6** → operator **OD-ENT-01** → **ACCEPTED_BASELINE_SHA** → **normal FREEZE** → **OD-ENT-08** / **FRZ-AUD-01..10** → roadmap §5 → **ARCH-FREEZE** handoff.

### PL-07

**282** tests collect-only in `tests/qualification/ent_audit_x` @ `d4974efc255bf6ff6ca96e00fb084293ea685d66` (collect-only SHA = documentation HEAD; **not** TEST_EXECUTION_SHA).

## Matrix C — 75 TaskID



| TaskID | Batch | Impl | Qual | Exact remaining delta @ HEAD | Owner | Deps | Positive test | Negative test | Acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C001 | W-A | PARTIAL | NOT_QUALIFIED | BaselinePhase; candidate vs accepted paths | baseline.py |  | `test_baseline_phase_no_accepted_without_operator` | `test_save_accepted_without_od_record_fails` | No accepted without OD-ENT-01 |
| C002 | W-A | PARTIAL | NOT_QUALIFIED | Remove auto `save_baseline`; candidate write only | runner.py | C001 | `test_freeze_bootstrap_writes_candidate_not_accepted` | `test_freeze_bootstrap_never_touches_accepted` | No auto-promotion |
| C003 | W-A | PARTIAL | NOT_QUALIFIED | `promote-baseline`: **operator trust boundary only** (F-07); verify OD-ENT-01 **record schema + ladder SHA pins**; **no** closed-bundle verify here (F-01) | run_enterprise_audit.py; baseline.py | C001,C006 | `test_promote_rejects_agent_only_record` | `test_promote_without_record_rejected` | Promotion impossible from agent-dropped file alone |
| C004 | W-A | IMPLEMENTED | QUALIFIED | None — complete @ audited HEAD | contracts.py; runner.py |  | `test_report_certifying_run_default_true` | `test_certifying_mode_rejects_check_id_filter` | No certifying filter |
| C005 | W-A | IMPLEMENTED | QUALIFIED | None — complete @ audited HEAD | CLI | C004 | `test_cli_check_id_requires_non_certifying` | `test_cli_full_with_check_id_exits_nonzero` | P0-04 closed |
| C006 | W-A | PARTIAL | NOT_QUALIFIED | Closed-world artifact catalog | evidence.py; catalog JSON |  | `test_evidence_build_matches_catalog` | `test_extra_artifact_fails_verify` | Closed verifier |
| C007 | W-E | PARTIAL | NOT_QUALIFIED | Remove scheduled bootstrap | nightly yml | C002 | `workflow lint` | `review` | No nightly accept |
| C008 | W-A | PARTIAL | NOT_QUALIFIED | `validate_domain_report()` 31 IDs | vbn.py |  | `test_vbn_validator_31_ids` | `test_vbn_forged_sha_rejected` | P0-03 contract |
| C009 | W-B | PARTIAL | NOT_QUALIFIED | FREEZE: (a) no accepted + no bootstrap → VIOLATION; (b) bootstrap → **candidate only** (F-02); (c) accepted → normal FREEZE | runner.py | C001,C002 | `test_freeze_without_accepted_and_without_bootstrap_blocked` | `test_first_candidate_freeze_writes_candidate` | F-02 lifecycle |
| C010 | W-B | PARTIAL | NOT_QUALIFIED | Fresh domain report; provenance cache | vbn.py | C008 | `test_vbn_fresh_run_on_full` | `test_stale_cache_rejected` | Fresh proof |
| C011 | W-B | PARTIAL | NOT_QUALIFIED | GAP/PARTIAL → BLOCKER | runner.py;modes.py | C008 | `test_full_fails_on_domain_gap` | `test_full_passes_31_qualified` | 31/31 |
| C012 | W-B | PARTIAL | NOT_QUALIFIED | `assert_ancestor` | repo_identity.py |  | `test_ancestor_valid` | `test_baseline_sha_not_ancestor_fails` | ECAP |
| C013 | W-B | PARTIAL | NOT_QUALIFIED | temp+rename | report.py |  | `test_report_atomic_on_interrupt` | `partial write` | No torn JSON |
| C014 | W-B | PARTIAL | NOT_QUALIFIED | per-run isolation | sandbox.py |  | `test_concurrent_runs_isolated` | `collision` | Isolation |
| C015 | W-B | PARTIAL | NOT_QUALIFIED | Extend redaction patterns (v2); core `_redact` @ audited SHA | sandbox.py |  | `test_redaction_strips_tenant_uuid` | `leak fixture` | P1-11 |
| C016 | W-B | PARTIAL | NOT_QUALIFIED | `run_status` completed/interrupted wired; interrupt path qual @ TES | runner.py |  | `test_manifest_run_status_complete` | `kill mid-run` | Partial class |
| C017 | W-B | PARTIAL | NOT_QUALIFIED | Timeout/orphan policy | sandbox.py |  | `test_sandbox_timeout_exit_124` | `hung proc` | Windows kill |
| C018 | W-B | PARTIAL | NOT_QUALIFIED | domains_accounted&lt;31 fail | runner.py | C011 | `integration` | `gap inject` | VBN enforced |
| C019 | W-B | PARTIAL | NOT_QUALIFIED | Closed-world vs Annex C | manifest.py |  | `test_manifest_matches_annex_c_ids` | `extra ID` | 85 exact |
| C020 | W-B | PARTIAL | NOT_QUALIFIED | FRZ map JSON | digests.py |  | `test_frz_map_not_all_aud03` | `missing map` | Per-check FRZ |
| C021 | W-B | PARTIAL | NOT_QUALIFIED | `frz_mapping` rows | manifest | C020 | `schema` | `invalid FRZ` | Matrix |
| C022 | W-B | PARTIAL | NOT_QUALIFIED | schema_version drift | baseline.py |  | `test_drift_regressed_on_schema_bump` | `stable` | P1-02 |
| C023 | W-B | PARTIAL | NOT_QUALIFIED | RESOLVED reopen | baseline.py | C022 | `test_resolved_reopens_as_regressed` | `stable` | FRZ-AUD-07 |
| C024 | W-B | PARTIAL | NOT_QUALIFIED | impact map data | changed_since.py |  | `test_impact_map_selects_invariant_owner` | `unknown BLOCKER` | Fail-closed |
| C025 | W-B | PARTIAL | NOT_QUALIFIED | unknown paths | changed_since.py | C024 | `test_unknown_platform_path_fail_closed` | `mapped ok` | P1-03 |
| C026 | W-B | PARTIAL | NOT_QUALIFIED | Closed bundle verify; enables **production** promote gate (wired in C042 after C026, not C003 dep) | validators.py; evidence.py | C006 | `test_verify_closed_bundle_ok` | `tamper/extra` | P1-12 |
| C027 | W-B | PARTIAL | NOT_QUALIFIED | path traversal | validators.py | C026 | `unit` | `../` paths` | Security |
| C028 | W-B | PARTIAL | NOT_QUALIFIED | manifest_digest mismatch | baseline.py;runner.py | C019 | `test_manifest_digest_change_invalidates_baseline` | `same ok` | Requal rule |
| C029 | W-B | PARTIAL | NOT_QUALIFIED | G2 generator | build_impact_map.py | C024 | `dry-run` | `diff` | SSOT |
| C030 | W-B | PARTIAL | NOT_QUALIFIED | orphan deps | schedule.py | C019 | `orphan test` | `cycle` | DAG |
| C031 | W-D | NOT_STARTED | NOT_QUALIFIED | REAL FULL 85/85 | l3/test_l3_full_real_85.py | C004,C011 | `test_l3_full_real_85_of_85` | `fail inject` | FULL oracle |
| C032 | W-D | NOT_STARTED | NOT_QUALIFIED | **Two tests:** (1) `test_l3_first_candidate_freeze_85` — bootstrap candidate, **no** accepted baseline (F-02); (2) `test_l3_normal_freeze_85` — requires **accepted** fixture post-OD (lab/simulated) | l3/test_l3_freeze_real_85.py | C002,C009,C011,C031 | `first-candidate pass` | `plain FREEZE blocked` | Does not require pre-accepted baseline for (1) |
| C033 | W-C | PARTIAL | NOT_QUALIFIED | gate helper | acceptance/conftest.py | C004 | `helper` | `N/A` | DRY |
| C034 | W-C | PARTIAL | NOT_QUALIFIED | gate subprocess | acceptance ENT-A | C033 | `per gate` | `fail inject` | 63 semantic |
| C035 | W-C | PARTIAL | NOT_QUALIFIED | gate subprocess | acceptance TEN | C033 | `13` | `tenant neg` | 13/13 |
| C036 | W-C | PARTIAL | NOT_QUALIFIED | gate subprocess | acceptance VBN | C033 | `9` | `VBN neg` | 9/9 |
| C037 | W-D | PARTIAL | NOT_QUALIFIED | ADV 01–10 | adversarial | C004,C026 | `per ADV` | `N/A` | fault inject |
| C038 | W-D | PARTIAL | NOT_QUALIFIED | ADV 11–20 | adversarial | C037 | `per ADV` | `N/A` | fault inject |
| C039 | W-D | PARTIAL | NOT_QUALIFIED | ADV 21–31 | adversarial | C037 | `per ADV` | `N/A` | fault inject |
| C040 | W-A | PARTIAL | NOT_QUALIFIED | ent_l1/l2/l3_heavy | pytest markers |  | `collect` | `N/A` | taxonomy |
| C041 | W-D | NOT_STARTED | NOT_QUALIFIED | Pytest inventory **F-04:** unit **28**, integration **41**, l3 **7**, acceptance **88** → **164** @ HEAD (`0321bf525`) | plan § | C040 | `pytest --collect-only` | `N/A` | Mechanical reconcile |
| C042 | W-D | PARTIAL | NOT_QUALIFIED | Unauthorized promote; agent-only OD file rejected; full bundle check after C026 | test_baseline_promotion.py | C003,C026 | `operator fixture path` | `agent file / no bundle` | Trust boundary |
| C043 | W-D | PARTIAL | NOT_QUALIFIED | forged JSON | test_vbn_forged_report.py | C008 | `detect` | `valid` | Security |
| C044 | W-D | NOT_STARTED | NOT_QUALIFIED | mutate status | test_report_status_mutation.py | C026 | `tamper` | `clean` | audit-of-auditor |
| C045 | W-E | PARTIAL | NOT_QUALIFIED | Base workflow skeleton + path list (extended by C059) | enterprise-audit.yml |  | `syntax` | `N/A` | PR cov |
| C046 | W-E | PARTIAL | NOT_QUALIFIED | align C007 | nightly yml | C007 | `CI` | `N/A` | nightly |
| C047 | W-E | PARTIAL | NOT_QUALIFIED | required checks | runbook |  | `review` | `N/A` | BLK-CI |
| C048 | W-G | PARTIAL | NOT_QUALIFIED | re-pin rules | SHA ladder | C031 | `review` | `doc-only qual` | P1-17 |
| C049 | W-G | NOT_STARTED | NOT_QUALIFIED | env fingerprint | evidence.py | C026 | `present` | `missing` | repro |
| C050 | W-G | PARTIAL | NOT_QUALIFIED | bundle builder | build_evidence_bundle.py | C026,C031 | `integration` | `tamper` | bundle |
| C051 | W-D | NOT_STARTED | NOT_QUALIFIED | L1→L3 orchestrator | run_qual_wave.py | C031,C032,C040 | `e2e` | `fail mid` | one wave |
| C052 | W-G | NOT_STARTED | NOT_QUALIFIED | retention/replay | runbook | C047 | `doc` | `N/A` | CI evidence |
| C053 | W-G | NOT_STARTED | NOT_QUALIFIED | mechanical fill | dossier template | C050 | `checklist` | `N/A` | B6 candidate |
| C054 | W-G | PARTIAL | NOT_QUALIFIED | pre-B6 script incl. audit.bat/sh + PR workflow checks | mechanical_readiness_check.py | C050,C059,C064 | `pass` | `inject fail` | mechanical |
| C055 | W-G | NOT_STARTED | NOT_QUALIFIED | 136 rows | Annex F |  | `review` | `N/A` | Gap SSOT |
| C056 | W-G | NOT_STARTED | NOT_QUALIFIED | repo publish | corrective plan md | C055 | `lint` | `N/A` | operator SSOT |
| C057 | W-G | NOT_STARTED | NOT_QUALIFIED | OD-ENT matrix | operator record |  | `table` | `N/A` | no agent sign |
| C058 | W-G | PARTIAL | NOT_QUALIFIED | replay tree | b6-handoff layout | C050,C053 | `validator` | `missing` | independent |
| C059 | W-E | PARTIAL | NOT_QUALIFIED | PR FAST; checkout **`pull_request.head.sha`**; report SHA fields; artifact metadata; job summary | .github/workflows/enterprise-audit.yml | C004,C045 | `test_ci_pr_report_sha_matches_head` | `invariant PR BLOCKED` | OD-ENT-02 separate |
| C060 | W-E | PARTIAL | NOT_QUALIFIED | Windows launcher: default/fast/full/changed/freeze; repo root discovery; uv/python check; unique `.tmp/proof/ent-audit-x/local-<ts>/`; exit code propagation; **no** `--bootstrap-baseline`; freeze requires accepted baseline | scripts/enterprise_audit/audit.bat | C004,C005,C062 | `test_audit_bat_default_fast` (Windows)` | `missing uv, invalid sha, spaces in path` | Same runner as CI |
| C061 | W-E | PARTIAL | NOT_QUALIFIED | Bash launcher: functional parity with BAT; `chmod +x`; POSIX-safe quoting; no cloud deps | scripts/enterprise_audit/audit.sh | C060,C062 | `test_audit_sh_default_fast` (Linux/macOS)` | `same negatives as BAT` | Parity with BAT |
| C062 | W-E | PARTIAL | NOT_QUALIFIED | Post-run console summary: mode, SHA, status, executed/skipped/failed counts, top findings, duration, MD/JSON paths, optional bundle path; `--open` local only; FAST labeled non-certifying-full | run_enterprise_audit.py; optional report_summary.py | C004 | `test_cli_prints_summary_fields` | `FAST never claims FULL cert` | C4-D contract |
| C063 | W-E | NOT_STARTED | NOT_QUALIFIED | 16 acceptance tests (C4-E matrix); mark `ent_launcher`; Windows job for BAT | tests/qualification/ent_audit_x/integration/test_audit_launchers_*.py | C059,C062 | `items 1–16 pass in target OS` | `N/A` | Cross-entrypoint semantics |
| C064 | W-G | NOT_STARTED | NOT_QUALIFIED | User quickstart (FAST one-liner, FULL, CHANGED-SINCE, FREEZE, reports, GitHub required checks); FRZ-AUD-10 evidence mapping; **no** claim OD-ENT-02 done | runbook; corrective plan; CLI doc; dossier §CI; docs/.../ENT_AUDIT_X_OPERATOR_QUICKSTART.md | C047,C059,C063 | `doc link checks` | `N/A` | Operator can run audit without long uv command |
| C065 | W-D | PARTIAL | NOT_QUALIFIED | Wire promote + operator_records; verify closed bundle (after C026) | baseline.py::verify_operator_decision_record | C003,C026 | `promote w/ valid ladder fixture` | `forged JSON` | no promote without B6 pin |
| C066 | W-G | PARTIAL | NOT_QUALIFIED | Rewrite checker; JSON report | mechanical_readiness_check.py::main | C031,C032,C051,C074 | `all gates true` | `missing L3` | TECHNICAL_READY only |
| C067 | W-A | IMPLEMENTED | NOT_QUALIFIED | None — code @ audited HEAD; qualification @ TEST_EXECUTION_SHA only | runner.py,modes.py | C004 | `dry-run exit 2` | `dry-run exit 0 cert` | no dry-run CLEAN certifying |
| C068 | W-A | PARTIAL | NOT_QUALIFIED | Cross-check manifest | evidence.py::build_evidence_manifest | C006,C067 | `real FULL bundle` | `dry-run bundle fail` | bundle rejected |
| C069 | W-A | NOT_STARTED | NOT_QUALIFIED | Add dependency; validate all artifacts | validators.py | C006 | `valid pass` | `extra props fail` | all artifacts |
| C070 | W-E | PARTIAL | NOT_QUALIFIED | Replace OUT= pattern | audit.bat,audit.sh, CLI | C060,C061 | `parallel runs` | `collision` | unique paths |
| C071 | W-B | IMPLEMENTED | NOT_QUALIFIED | None — code @ audited HEAD; qualification @ TEST_EXECUTION_SHA only | modes.py::aggregate_status | C023 | `resolved PASS` | `blocker resolved` | no false BLOCK |
| C072 | W-B | PARTIAL | NOT_QUALIFIED | Impact map JSON + `select_checks_for_changed_since`; qual mapping tests @ TES | changed_since.py, impact JSON | C012,C024 | `mapped change` | `unmapped BLOCK` | fail-closed |
| C073 | W-A | PARTIAL | NOT_QUALIFIED | Catalog profile versions; strict verify @ TES | evidence_artifact_catalog.v1.json | C006 | `profile match` | `missing artifact` | mode closure |
| C074 | W-D | NOT_STARTED | NOT_QUALIFIED | Harness + markers | tests/.../l3/test_l3_qual_wave.py (new) | C031,C032 | `85/85 PASS` | `partial fail` | strict exit 0 |
| C075 | W-G | PARTIAL | NOT_QUALIFIED | C075-S2 final reconcile (S1 @ W-0 done in this repair wave) | this doc, Annex F/H, ladder, dossier | C041,C048,C055 | `link lint` | `stale pin` | SSOT match HEAD |
## Matrix D

### GAP-RC (33)

| GAP-ID | Source | Root cause | Sev | Code / status | TaskID | Pos | Neg | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAP-RC-OD-TRUST | A-01, P0-01, F-07 | OD-ENT-01 verifies boolean JSON only | CRIT | `baseline.py::verify_operator_decision_record` | OPEN | C003,C042,C065 | promote w/ ladder-valid fixture | forged `operator_origin=true` | OD record + bundle | Promotion fails without ECAP anchors |
| GAP-RC-MECH-READY | A-02,A-20 | Readiness = file existence | CRIT | OPEN | C054,C066 | readiness @ qual SHA | missing L3 artifact | readiness JSON | `TECHNICAL_READY_FOR_B6` false without qual artifacts |
| GAP-RC-EVIDENCE-TRUST | A-03,A-06,A-14,P1-01,P1-12 | Bundle not tied to execution manifest | CRIT | PARTIAL | C006,C026,C068,C073 | closed bundle @ real FULL | dry-run bundle | manifest+digest | No qual bundle without 85 executed |
| GAP-RC-L3-FULL | A-04,P0-02,P1-15 | No real FULL/FREEZE 85/85 L3 | CRIT | OPEN | C031,C032,C074 | L3 FULL exit 0 strict | single-check/dry-run | `.tmp/proof/` | `checks_executed==85`, coverage FULL |
| GAP-RC-DRY-RUN | A-05,ADV-05 | FAST dry-run can exit CLEAN | CRIT | OPEN | C067,C004 | dry-run INCOMPLETE | dry-run PASS | report+manifest | `certifying_run`+dry_run → non-qual |
| GAP-RC-VBN-31 | A-07,A-08,P0-03,P1-07 | No D01..D31 validator/provenance | CRIT | OPEN | C008–C011,C018,C043 | 31 IDs fresh | stale cache | vbn_domains.v1.json | domains_accounted==31 on certifying FULL |
| GAP-RC-CHANGED-IMPACT | A-09,P1-03 | Path-only CHANGED-SINCE | MAJ | OPEN | C024,C025,C029,C072 | impact map selects checks | unmapped path BLOCKER | impact map JSON | SSOT Changed→Invariant→CheckID |
| GAP-RC-GIT-ANCESTRY | A-10,P1-13 | Ancestry only on FREEZE w/ accepted | MAJ | PARTIAL | C012,C072 | ancestor valid | forked baseline | git log | CHANGED-SINCE + publish pins |
| GAP-RC-RESOLVED-BLOCK | A-11 | RESOLVED keeps old severity in aggregate | MAJ | OPEN | C071,C023 | RESOLVED not BLOCKED | BLOCKER resolved | drift unit | aggregate ignores RESOLVED severity |
| GAP-RC-DRIFT-SCHEMA | A-12,P1-02 | `current_schema=1` hardcoded | MAJ | OPEN | C022,C028 | schema bump REGRESSED | stable pass | baseline rows | per-check schema in manifest |
| GAP-RC-JSON-SCHEMA | A-13 | Shallow validator | MAJ | OPEN | C069 | full schema pass | extra fields fail | validator errors | jsonschema on all artifacts |
| GAP-RC-ATOMIC-PUB | A-15 | Per-file atomic only | MAJ | PARTIAL | C013 | interrupt mid-publish | torn set | output dir | all-or-nothing report set |
| GAP-RC-SANDBOX | A-16,P1-10 | Isolation/timeouts/concurrency | MAJ | OPEN | C014–C017 | concurrent runs | collision | sandbox tests | isolated output + kill orphans |
| GAP-RC-REDACTION | A-17,P1-11 | Marker redaction only | MAJ | OPEN | C015 | tenant uuid stripped | leak fixture | stdout/json | negative leakage tests |
| GAP-RC-SEMANTIC-GATES | A-18 | Gates = entrypoint exists not exec | CRIT | OPEN | C033–C036 | per-gate subprocess | inject fail | outcomes.json | 85/85 gates exit 0 @ qual SHA |
| GAP-RC-ADV-INJECT | A-19,P1-04 | ADV mostly existence | MAJ | OPEN | C037–C039 | per ADV fault | N/A | adv log | 31 fault-injection tests |
| GAP-RC-PR-PATH-SKIP | A-22,P1-05 | paths-filter skips PR jobs | MAJ | OPEN | C046,C059 | path audit test | doc-only PR | workflow | broad triggers or required paths doc |
| GAP-RC-NIGHTLY-FREEZE | A-23,P1-06 | Nightly FREEZE bootstrap | MAJ | OPEN | C007,C046 | scheduled FULL only | auto bootstrap | yaml | FREEZE bootstrap operator-only |
| GAP-RC-LAUNCHER-BAT | A-24 | Unknown arg → FAST | MAJ | OPEN | C060,C063 | valid modes | unknown arg ≠0 | bat test | invalid → exit 1 |
| GAP-RC-LAUNCHER-SH | A-25 | Same as BAT | MAJ | OPEN | C061,C063 | parity | unknown mode | sh test | invalid → exit 1 |
| GAP-RC-LAUNCHER-CWD | A-26 | No spaces-in-path qual | MAJ | OPEN | C063 | cwd≠repo root | spaced path | launcher test | uv cwd = repo root |
| GAP-RC-OUTPUT-COLLISION | A-27 | RANDOM output dirs | MAJ | OPEN | C070 | unique run_id | parallel clash | proof dir | uuid+pid lock |
| GAP-RC-DOSSIER | A-28 | Draft dossier unpinned | MAJ | OPEN | C053,C075 | dossier fields | missing SHA | dossier md | TEST_EXECUTION_SHA filled post-qual |
| GAP-RC-SHA-LADDER | A-29,P1-17 | Ladder stale vs HEAD | MAJ | OPEN | C048,C075 | append-only row | rewrite old | ladder md | IMPLEMENTATION_SHA `1fa5347a5` row |
| GAP-RC-SSOT-DRIFT | A-30,P1-16,P1-18 | Doc counts ≠ collect-only | MAJ | OPEN | C041,C055,C064,C075 | collect-only match | stale § | pytest log | 180 nodeids reconciled |
| GAP-RC-FRZ-MAP | P1-08 | FRZ artifact coarse | MAJ | OPEN | C020,C021 | per-check FRZ | missing map | frz_coverage | each check has frz_mapping |
| GAP-RC-RUN-STATUS | P1-14 | No run_status on interrupt | MAJ | OPEN | C016 | complete enum | kill mid-run | manifest | partial runs INCOMPLETE |
| GAP-RC-REQUAL-DIGEST | requal | manifest_digest change | MAJ | OPEN | C028 | digest change blocks | same digest ok | baseline | accepted invalid until requal |
| GAP-RC-AUDIT-OF-AUDITOR | false PASS | status field tamper | MAJ | OPEN | C044 | mutate status | clean verify | report | tamper detected |
| GAP-RC-EVID-PUBLISH | CP-08 | No orchestrated publish | MAJ | OPEN | C049–C051,C058 | bundle builder e2e | tamper | `.tmp/proof/` | immutable publication tree |
| GAP-RC-CI-PROFILE | GAP-P-01 | enterprise profile ≠ ENT audit | MIN | OPEN | C045 | distinct job | conflation | yaml | separate workflow identity |
| GAP-RC-CERTIFY-SPLIT | P0-04 | — | — | **CLOSED_WITH_EVIDENCE** | C004,C005 | certifying default | check-id blocked | tests | no re-impl |
| GAP-RC-CANDIDATE-FSM | P0-01 partial | — | — | **PARTIAL** | C001,C002 | no auto accepted | bootstrap candidate | tests | promotion still GAP-RC-OD-TRUST |

### GAP-B (6)

| GAP-B-ID | Status | Owner | Required evidence | Closure |
| --- | --- | --- | --- | --- |
| BLK-EVOL-B6 | EXTERNAL | Independent auditor + operator | EVOL Protocol 03 PASS @ EVOL TES/FINAL; ladder B6 SHA | Before ENT stage CLOSED (OD-ENT-00) |
| BLK-EVOL-FRZ | EXTERNAL | Operator | FRZ-EVOL-01..10 promoted after EVOL B6 | ARCH-FREEZE EVOL criteria |
| BLK-EVOL-HANDOFF | EXTERNAL | Operator | W7-06 handoff pin + EVOL formal closure SHA | ENT EVOL digest sync |
| BLK-CI-ENFORCE | EXTERNAL | OD-ENT-02 + GitHub admin | Branch protection + required checks @ TES | FRZ-AUD-10 |
| BLK-ROADMAP-ENT-DEP | EXTERNAL | Operator OD-ENT-00 | Written decision: EVOL-X formally CLOSED or documented exception | Roadmap §5 ENT-AUDIT-X row |
| BLK-ENT-B6 | EXTERNAL | Independent Protocol 03 | B5 bundle + 85/85 @ TES + ADV31; dossier PASS | OD-ENT-01 / FRZ-AUD PASS |


## Matrix E — 85 semantic oracles

| CheckID | kind | Entrypoint | Invariant | Semantic oracle @ TES | Negative mutation | FRZ | GAP-COV | TaskID | Evidence | Qual @ HEAD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ENT-A-BND-01 | pytest_nodeid | `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py::test_hardening_3_contracts_do_not_import_runtime_except_allowlist` | Illegal tier imports absent | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_bnd_01.py::test_acceptance_ent_a_bnd_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-BND-01 @ TES | adversarial fault inject for ENT-A-BND-01 → non-zero exit | FRZ-BND-* | GAP-COV-ENT-A-BND-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-BND-02 | pytest_nodeid | `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py::test_hardening_3_unallowlisted_runtime_import_detected_in_isolated_tree` | Reverse dependency scan | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_bnd_02.py::test_acceptance_ent_a_bnd_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-BND-02 @ TES | adversarial fault inject for ENT-A-BND-02 → non-zero exit | FRZ-BND-* | GAP-COV-ENT-A-BND-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-BND-03 | python_script | `scripts/maintenance/check_agents_no_tier3_imports.py` | Agents/apps boundary | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_bnd_03.py::test_acceptance_ent_a_bnd_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-BND-03 @ TES | adversarial fault inject for ENT-A-BND-03 → non-zero exit | FRZ-BND-* | GAP-COV-ENT-A-BND-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-CMP-01 | pytest_nodeid | `tests/qualification/compat_x/test_compat_x_inventory_gates.py::test_cx_p0_r1_q05_no_parallel_authority_in_production_inventory` | No second permanent mechanism | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_cmp_01.py::test_acceptance_ent_a_cmp_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-CMP-01 @ TES | adversarial fault inject for ENT-A-CMP-01 → non-zero exit | FRZ-CMP-* | GAP-COV-ENT-A-CMP-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-CMP-02 | pytest_nodeid | `tests/qualification/compat_x/test_compat_x_persisted_r2_final_gates.py::test_cx_r2_final_probe_d_version_owner_drift_fails` | Deprecation enforcement | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_cmp_02.py::test_acceptance_ent_a_cmp_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-CMP-02 @ TES | adversarial fault inject for ENT-A-CMP-02 → non-zero exit | FRZ-CMP-* | GAP-COV-ENT-A-CMP-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-CMP-03 | pytest_nodeid | `tests/qualification/compat_x/test_compat_x_inventory_gates.py::test_cx_p0_r1_adversarial_07_shim_discovered_without_path_list_edit` | Shim not authority | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_cmp_03.py::test_acceptance_ent_a_cmp_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-CMP-03 @ TES | adversarial fault inject for ENT-A-CMP-03 → non-zero exit | FRZ-CMP-* | GAP-COV-ENT-A-CMP-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-CTR-01 | pytest_nodeid | `tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes` | Public contract purity | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_ctr_01.py::test_acceptance_ent_a_ctr_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-CTR-01 @ TES | adversarial fault inject for ENT-A-CTR-01 → non-zero exit | FRZ-CTR-* | GAP-COV-ENT-A-CTR-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-CTR-02 | pytest_nodeid | `tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes` | Contract/impl separation | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_ctr_02.py::test_acceptance_ent_a_ctr_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-CTR-02 @ TES | adversarial fault inject for ENT-A-CTR-02 → non-zero exit | FRZ-CTR-* | GAP-COV-ENT-A-CTR-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-CTR-03 | pytest_nodeid | `tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes` | Narrow responsibility | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_ctr_03.py::test_acceptance_ent_a_ctr_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-CTR-03 @ TES | adversarial fault inject for ENT-A-CTR-03 → non-zero exit | FRZ-CTR-* | GAP-COV-ENT-A-CTR-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-DOC-01 | pytest_nodeid | `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py::test_replay_x_mandatory_docs_present` | Canonical docs present | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_doc_01.py::test_acceptance_ent_a_doc_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-DOC-01 @ TES | adversarial fault inject for ENT-A-DOC-01 → non-zero exit | FRZ-DOC-* | GAP-COV-ENT-A-DOC-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-DOC-02 | python_script | `scripts/enterprise_audit/check_manifest_annex_c_alignment.py` | Ownership manifest drift | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_doc_02.py::test_acceptance_ent_a_doc_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-DOC-02 @ TES | adversarial fault inject for ENT-A-DOC-02 → non-zero exit | FRZ-DOC-* | GAP-COV-ENT-A-DOC-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-DOC-03 | pytest_nodeid | `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py::test_replay_x_frz_row_count` | Roadmap/checklist sync | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_doc_03.py::test_acceptance_ent_a_doc_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-DOC-03 @ TES | adversarial fault inject for ENT-A-DOC-03 → non-zero exit | FRZ-DOC-* | GAP-COV-ENT-A-DOC-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-DUP-01 | pytest_file | `tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py` | Duplicate factories | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_dup_01.py::test_acceptance_ent_a_dup_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-DUP-01 @ TES (shared file — wrapper scopes assertion to ENT-A-DUP-01) | adversarial fault inject for ENT-A-DUP-01 → non-zero exit | FRZ-DEBT / bypass | GAP-COV-ENT-A-DUP-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-DUP-02 | pytest_file | `tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py` | Shadow registries | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_dup_02.py::test_acceptance_ent_a_dup_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-DUP-02 @ TES (shared file — wrapper scopes assertion to ENT-A-DUP-02) | adversarial fault inject for ENT-A-DUP-02 → non-zero exit | REG | GAP-COV-ENT-A-DUP-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-DUP-03 | pytest_file | `tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py` | Alternate execution paths | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_dup_03.py::test_acceptance_ent_a_dup_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-DUP-03 @ TES (shared file — wrapper scopes assertion to ENT-A-DUP-03) | adversarial fault inject for ENT-A-DUP-03 → non-zero exit | EXE/GOV | GAP-COV-ENT-A-DUP-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-EVOL-01 | pytest_file | `tests/qualification/evol_x/gates/test_evol_x_selection.py` | Impact-aware selection | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_evol_01.py::test_acceptance_ent_a_evol_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-EVOL-01 @ TES (shared file — wrapper scopes assertion to ENT-A-EVOL-01) | adversarial fault inject for ENT-A-EVOL-01 → non-zero exit | FRZ-EVOL-* | GAP-COV-ENT-A-EVOL-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-EVOL-02 | pytest_file | `tests/qualification/evol_x/gates/test_evol_x_selection_fault_injection.py` | Conservative regression | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_evol_02.py::test_acceptance_ent_a_evol_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-EVOL-02 @ TES (shared file — wrapper scopes assertion to ENT-A-EVOL-02) | adversarial fault inject for ENT-A-EVOL-02 → non-zero exit | FRZ-EVOL-06 | GAP-COV-ENT-A-EVOL-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-EVOL-03 | pytest_file | `tests/qualification/evol_x/gates/test_evol_x_closed_world_gates.py` | Post-freeze hooks | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_evol_03.py::test_acceptance_ent_a_evol_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-EVOL-03 @ TES (shared file — wrapper scopes assertion to ENT-A-EVOL-03) | adversarial fault inject for ENT-A-EVOL-03 → non-zero exit | FRZ-EVOL-09 | GAP-COV-ENT-A-EVOL-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-EXE-01 | pytest_file | `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py` | Single execution authority | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_exe_01.py::test_acceptance_ent_a_exe_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-EXE-01 @ TES (shared file — wrapper scopes assertion to ENT-A-EXE-01) | adversarial fault inject for ENT-A-EXE-01 → non-zero exit | FRZ-EXE-* | GAP-COV-ENT-A-EXE-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-EXE-02 | pytest_file | `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py` | No peer schedulers | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_exe_02.py::test_acceptance_ent_a_exe_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-EXE-02 @ TES (shared file — wrapper scopes assertion to ENT-A-EXE-02) | adversarial fault inject for ENT-A-EXE-02 → non-zero exit | FRZ-EXE-* | GAP-COV-ENT-A-EXE-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-EXE-03 | pytest_file | `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py` | Canonical execution identity | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_exe_03.py::test_acceptance_ent_a_exe_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-EXE-03 @ TES (shared file — wrapper scopes assertion to ENT-A-EXE-03) | adversarial fault inject for ENT-A-EXE-03 → non-zero exit | FRZ-EXE-* | GAP-COV-ENT-A-EXE-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-GOV-01 | pytest_nodeid | `tests/qualification/governance/gov_x2/test_gov_x2_qualification_batch.py::test_gov_x2_catalog_covers_gx2_01_through_gx2_20` | Governance ≠ Execution | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_gov_01.py::test_acceptance_ent_a_gov_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-GOV-01 @ TES | adversarial fault inject for ENT-A-GOV-01 → non-zero exit | FRZ-GOV-* | GAP-COV-ENT-A-GOV-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-GOV-02 | pytest_nodeid | `tests/qualification/governance/gov_x2/test_gov_x2_qualification_batch.py::test_gov_x2_invariants_declare_pass_with_proof_nodes` | Authority narrowing | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_gov_02.py::test_acceptance_ent_a_gov_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-GOV-02 @ TES | adversarial fault inject for ENT-A-GOV-02 → non-zero exit | FRZ-GOV-* | GAP-COV-ENT-A-GOV-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-GOV-03 | pytest_nodeid | `tests/qualification/governance/gov_x2/test_gov_x2_proof_replay_gates.py::test_gov_x2_each_invariant_has_at_least_one_proof_node` | Fresh authorization | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_gov_03.py::test_acceptance_ent_a_gov_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-GOV-03 @ TES | adversarial fault inject for ENT-A-GOV-03 → non-zero exit | FRZ-GOV-* | GAP-COV-ENT-A-GOV-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-INV-01 | pytest_file | `tests/qualification/evol_x/gates/test_evol_x_closed_world_gates.py` | SYS-INV index coverage | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_inv_01.py::test_acceptance_ent_a_inv_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-INV-01 @ TES (shared file — wrapper scopes assertion to ENT-A-INV-01) | adversarial fault inject for ENT-A-INV-01 → non-zero exit | SYS-INV-* | GAP-COV-ENT-A-INV-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-INV-02 | pytest_nodeid | `tests/qualification/ent_audit_x/unit/test_frz_coverage_matrix_complete.py::test_every_check_maps_to_frz_or_sys_inv` | Cross-family consistency | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_inv_02.py::test_acceptance_ent_a_inv_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-INV-02 @ TES | adversarial fault inject for ENT-A-INV-02 → non-zero exit | multi-FRZ | GAP-COV-ENT-A-INV-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-INV-03 | pytest_nodeid | `tests/qualification/ent_audit_x/integration/test_ent_audit_cannot_exit_clean_with_skipped_mandatory.py::test_full_mode_exit_not_clean_when_check_skipped` | Fail-closed global | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_inv_03.py::test_acceptance_ent_a_inv_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-INV-03 @ TES | adversarial fault inject for ENT-A-INV-03 → non-zero exit | SYS-INV fail-closed | GAP-COV-ENT-A-INV-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-OBS-01 | pytest_nodeid | `tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py::test_g1_otlp_transport_port_export_parameter` | Obs ≠ execution truth | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_obs_01.py::test_acceptance_ent_a_obs_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-OBS-01 @ TES | adversarial fault inject for ENT-A-OBS-01 → non-zero exit | FRZ-OBS-* | GAP-COV-ENT-A-OBS-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-OBS-02 | pytest_nodeid | `tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py::test_g6_contracts_observability_export_no_runtime_imports` | Event spine single | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_obs_02.py::test_acceptance_ent_a_obs_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-OBS-02 @ TES | adversarial fault inject for ENT-A-OBS-02 → non-zero exit | FRZ-OBS-* | GAP-COV-ENT-A-OBS-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-OBS-03 | pytest_nodeid | `tests/qualification/trace_x/test_trace_x_p4_r1_recorder_tenant.py::test_txp4r1_q06_sequential_tenant_isolation` | Correlation propagation | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_obs_03.py::test_acceptance_ent_a_obs_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-OBS-03 @ TES | adversarial fault inject for ENT-A-OBS-03 → non-zero exit | FRZ-TRC-* | GAP-COV-ENT-A-OBS-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-OWN-01 | pytest_file | `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py` | Exactly-one semantic owner | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_own_01.py::test_acceptance_ent_a_own_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-OWN-01 @ TES (shared file — wrapper scopes assertion to ENT-A-OWN-01) | adversarial fault inject for ENT-A-OWN-01 → non-zero exit | FRZ-OWN-* | GAP-COV-ENT-A-OWN-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-OWN-02 | pytest_file | `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py` | Composition owner sanctioned | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_own_02.py::test_acceptance_ent_a_own_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-OWN-02 @ TES (shared file — wrapper scopes assertion to ENT-A-OWN-02) | adversarial fault inject for ENT-A-OWN-02 → non-zero exit | FRZ-OWN-* | GAP-COV-ENT-A-OWN-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-OWN-03 | pytest_file | `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py` | Shadow authority detection | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_own_03.py::test_acceptance_ent_a_own_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-OWN-03 @ TES (shared file — wrapper scopes assertion to ENT-A-OWN-03) | adversarial fault inject for ENT-A-OWN-03 → non-zero exit | FRZ-OWN-* | GAP-COV-ENT-A-OWN-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-PLG-01 | pytest_file | `tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py` | Structural replaceability | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_plg_01.py::test_acceptance_ent_a_plg_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-PLG-01 @ TES (shared file — wrapper scopes assertion to ENT-A-PLG-01) | adversarial fault inject for ENT-A-PLG-01 → non-zero exit | FRZ-PLG-* | GAP-COV-ENT-A-PLG-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-PLG-02 | pytest_file | `tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py` | Provider behind contract | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_plg_02.py::test_acceptance_ent_a_plg_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-PLG-02 @ TES (shared file — wrapper scopes assertion to ENT-A-PLG-02) | adversarial fault inject for ENT-A-PLG-02 → non-zero exit | FRZ-PLG-* | GAP-COV-ENT-A-PLG-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-PLG-03 | pytest_file | `tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py` | No consumer patch replace | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_plg_03.py::test_acceptance_ent_a_plg_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-PLG-03 @ TES (shared file — wrapper scopes assertion to ENT-A-PLG-03) | adversarial fault inject for ENT-A-PLG-03 → non-zero exit | FRZ-RPL-* | GAP-COV-ENT-A-PLG-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-PRD-01 | python_script | `scripts/gates/check_application_production_gates.py` | Health/readiness semantics | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_prd_01.py::test_acceptance_ent_a_prd_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-PRD-01 @ TES | adversarial fault inject for ENT-A-PRD-01 → non-zero exit | FRZ-PRD-* arch slice | GAP-COV-ENT-A-PRD-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-PRD-02 | python_script | `scripts/gates/check_application_production_gates.py` | Degraded visibility | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_prd_02.py::test_acceptance_ent_a_prd_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-PRD-02 @ TES | adversarial fault inject for ENT-A-PRD-02 → non-zero exit | FRZ-PRD-* | GAP-COV-ENT-A-PRD-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-PRD-03 | pytest_nodeid | `tests/qualification/ent_audit_x/unit/test_report_includes_prod_q_dimension_b_limitation.py::test_limitations_list_prod_q_live_open` | Live slice limitation flag | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_prd_03.py::test_acceptance_ent_a_prd_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-PRD-03 @ TES | adversarial fault inject for ENT-A-PRD-03 → non-zero exit | FRZ-PRD live OPEN | GAP-COV-ENT-A-PRD-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-REC-01 | pytest_nodeid | `tests/qualification/state_x/test_state_x_p0_baseline.py::test_sx_p0_q15_backup_restore_responsibility_classified` | Backup/restore responsibility | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_rec_01.py::test_acceptance_ent_a_rec_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-REC-01 @ TES | adversarial fault inject for ENT-A-REC-01 → non-zero exit | FRZ-REC-* | GAP-COV-ENT-A-REC-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-REC-02 | pytest_nodeid | `tests/qualification/state_x/test_state_x_r3_auxiliary_durable_state.py::test_r3_r1_q13_redelivery_preserves_consumption` | Restore no divergent truth | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_rec_02.py::test_acceptance_ent_a_rec_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-REC-02 @ TES | adversarial fault inject for ENT-A-REC-02 → non-zero exit | FRZ-REC-* | GAP-COV-ENT-A-REC-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-REC-03 | pytest_nodeid | `tests/qualification/state_x/test_state_x_r3_auxiliary_durable_state.py::test_r3_r1_q12_corrupt_state_fails_closed` | Replay tenant continuity | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_rec_03.py::test_acceptance_ent_a_rec_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-REC-03 @ TES | adversarial fault inject for ENT-A-REC-03 → non-zero exit | FRZ-REC-* | GAP-COV-ENT-A-REC-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-REG-01 | pytest_nodeid | `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py::test_replay_x_frz_row_count` | Stale allowlists = finding | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_reg_01.py::test_acceptance_ent_a_reg_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-REG-01 @ TES | adversarial fault inject for ENT-A-REG-01 → non-zero exit | FRZ-REG-* | GAP-COV-ENT-A-REG-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-REG-02 | pytest_file | `tests/qualification/evol_x/gates/test_evol_x_closed_world_gates.py` | Deterministic qual | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_reg_02.py::test_acceptance_ent_a_reg_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-REG-02 @ TES (shared file — wrapper scopes assertion to ENT-A-REG-02) | adversarial fault inject for ENT-A-REG-02 → non-zero exit | FRZ-REG-* | GAP-COV-ENT-A-REG-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-REG-03 | python_script | `scripts/gates/check_application_production_gates.py` | Env failure ≠ PASS | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_reg_03.py::test_acceptance_ent_a_reg_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-REG-03 @ TES | adversarial fault inject for ENT-A-REG-03 → non-zero exit | FRZ-REG-* | GAP-COV-ENT-A-REG-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-REL-01 | pytest_nodeid | `tests/unit/runtime/architecture/test_harness_w4_r1_production_tool_boundedness_gate.py::test_w4_r1_production_roots_pass_dependency_attempt_boundary` | Bounded admission | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_rel_01.py::test_acceptance_ent_a_rel_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-REL-01 @ TES | adversarial fault inject for ENT-A-REL-01 → non-zero exit | FRZ-REL-* | GAP-COV-ENT-A-REL-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-REL-02 | pytest_nodeid | `tests/qualification/harness_01/test_harness_01_gates.py::test_harness_01_tool_budget_record_precedes_invoker_invoke` | Retry caps | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_rel_02.py::test_acceptance_ent_a_rel_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-REL-02 @ TES | adversarial fault inject for ENT-A-REL-02 → non-zero exit | FRZ-REL-* | GAP-COV-ENT-A-REL-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-REL-03 | pytest_nodeid | `tests/qualification/harness_01/test_harness_01_gates.py::test_harness_01_zero_bypass_findings_index_reports_no_blocker_rows` | Overload classification | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_rel_03.py::test_acceptance_ent_a_rel_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-REL-03 @ TES | adversarial fault inject for ENT-A-REL-03 → non-zero exit | FRZ-REL-* | GAP-COV-ENT-A-REL-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-SEC-01 | python_script | `scripts/maintenance/check_immutable_security_audit_trail.py` | Secret boundary | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_sec_01.py::test_acceptance_ent_a_sec_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-SEC-01 @ TES | adversarial fault inject for ENT-A-SEC-01 → non-zero exit | FRZ-SEC-* | GAP-COV-ENT-A-SEC-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-SEC-02 | python_script | `scripts/maintenance/check_immutable_security_audit_trail.py` | Sensitive flow classification | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_sec_02.py::test_acceptance_ent_a_sec_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-SEC-02 @ TES | adversarial fault inject for ENT-A-SEC-02 → non-zero exit | FRZ-SEC-* | GAP-COV-ENT-A-SEC-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-SEC-03 | pytest_nodeid | `tests/qualification/ent_audit_x/unit/test_enterprise_audit_report_redaction.py::test_report_json_contains_no_environment_secrets` | Redaction in reports | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_sec_03.py::test_acceptance_ent_a_sec_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-SEC-03 @ TES | adversarial fault inject for ENT-A-SEC-03 → non-zero exit | FRZ-SEC-* | GAP-COV-ENT-A-SEC-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-STA-01 | pytest_nodeid | `tests/qualification/state_x/test_state_x_p0_baseline.py::test_sx_p0_q19_duplicated_truth_gate` | Single semantic truth owner | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_sta_01.py::test_acceptance_ent_a_sta_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-STA-01 @ TES | adversarial fault inject for ENT-A-STA-01 → non-zero exit | FRZ-STA-* | GAP-COV-ENT-A-STA-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-STA-02 | pytest_nodeid | `tests/qualification/state_x/test_state_x_p0_baseline.py::test_sx_p0_q11_projections_coordination_not_mint_authority` | Configured vs effective | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_sta_02.py::test_acceptance_ent_a_sta_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-STA-02 @ TES | adversarial fault inject for ENT-A-STA-02 → non-zero exit | FRZ-STA-* | GAP-COV-ENT-A-STA-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-STA-03 | pytest_nodeid | `tests/qualification/state_x/test_state_x_r3_auxiliary_durable_state.py::test_r3_r1_q10_tenant_isolation_same_run_id` | Duplicate stores | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_sta_03.py::test_acceptance_ent_a_sta_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-STA-03 @ TES | adversarial fault inject for ENT-A-STA-03 → non-zero exit | FRZ-STA-* | GAP-COV-ENT-A-STA-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-TEN-01 | pytest_file | `tests/qualification/tenant_x/test_tenant_x_final_current_head.py` | TEN module rollup | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_ten_01.py::test_acceptance_ent_a_ten_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-TEN-01 @ TES (shared file — wrapper scopes assertion to ENT-A-TEN-01) | adversarial fault inject for ENT-A-TEN-01 → non-zero exit | FRZ-TEN-* | GAP-COV-ENT-A-TEN-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-TEN-02 | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r1_current_head.py::test_tx_r1_head_q01_inventory_reconciles_twenty_two_mechanisms` | Adversarial coverage | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_ten_02.py::test_acceptance_ent_a_ten_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-TEN-02 @ TES | adversarial fault inject for ENT-A-TEN-02 → non-zero exit | FRZ-TEN-12 | GAP-COV-ENT-A-TEN-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-TEN-03 | pytest_nodeid | `tests/qualification/ent_audit_x/integration/test_ent_a_ten_03_full_schedules_all_tenant_checks.py::test_full_mode_plans_all_thirteen_ten_star_checks` | Continuous re-orchestration | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_ten_03.py::test_acceptance_ent_a_ten_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-TEN-03 @ TES | adversarial fault inject for ENT-A-TEN-03 → non-zero exit | FRZ-TEN-* | GAP-COV-ENT-A-TEN-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-TYP-01 | pytest_file | `tests/unit/architecture/test_ebh_2a_public_contract_boundary_gate.py` | Semantic Any at boundaries | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_typ_01.py::test_acceptance_ent_a_typ_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-TYP-01 @ TES (shared file — wrapper scopes assertion to ENT-A-TYP-01) | adversarial fault inject for ENT-A-TYP-01 → non-zero exit | FRZ-TYP-* | GAP-COV-ENT-A-TYP-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-TYP-02 | pytest_nodeid | `tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes` | type: ignore masking | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_typ_02.py::test_acceptance_ent_a_typ_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-TYP-02 @ TES | adversarial fault inject for ENT-A-TYP-02 → non-zero exit | FRZ-TYP-* | GAP-COV-ENT-A-TYP-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-TYP-03 | pytest_nodeid | `tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes` | String/dict pseudo-contracts | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_typ_03.py::test_acceptance_ent_a_typ_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-TYP-03 @ TES | adversarial fault inject for ENT-A-TYP-03 → non-zero exit | FRZ-TYP-* | GAP-COV-ENT-A-TYP-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-VBN-01 | pytest_nodeid | `tests/qualification/ebh_5/test_ebh_5_g06_enterprise_integrity_gates.py::test_ebh5_g06_01_roadmap_31_domains_accounted` | Vendor inventory current | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_vbn_01.py::test_acceptance_ent_a_vbn_01_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-VBN-01 @ TES | adversarial fault inject for ENT-A-VBN-01 → non-zero exit | FRZ-RPL-* | GAP-COV-ENT-A-VBN-01 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-VBN-02 | pytest_nodeid | `tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero` | Fallback classification | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_vbn_02.py::test_acceptance_ent_a_vbn_02_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-VBN-02 @ TES | adversarial fault inject for ENT-A-VBN-02 → non-zero exit | FRZ-CFG-* | GAP-COV-ENT-A-VBN-02 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| ENT-A-VBN-03 | pytest_file | `tests/qualification/config_x/test_config_x_final_reconciliation_gates.py` | Reachability | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_vbn_03.py::test_acceptance_ent_a_vbn_03_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=ENT-A-VBN-03 @ TES (shared file — wrapper scopes assertion to ENT-A-VBN-03) | adversarial fault inject for ENT-A-VBN-03 → non-zero exit | FRZ-RPL-* | GAP-COV-ENT-A-VBN-03 | C034 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-ASYNC | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py::test_tx_r4_q06_adversarial_context_view_request_mismatch` | Async resume tenant continuity | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_async.py::test_acceptance_ten_async_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-ASYNC @ TES | context swap | FRZ-TEN-03 | GAP-COV-TEN-ASYNC | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-AUTHORITY | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py::test_tx_r4_q07_adversarial_background_identity` | Child scope ⊆ parent | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_authority.py::test_acceptance_ten_authority_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-AUTHORITY @ TES | widen scope inject | FRZ-TEN-03 | GAP-COV-TEN-AUTHORITY | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-CONFIG | pytest_file | `tests/qualification/config_x/test_config_x_final_reconciliation_gates.py` | Tenant-scoped configuration | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_config.py::test_acceptance_ten_config_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-CONFIG @ TES (shared file — wrapper scopes assertion to TEN-CONFIG) | global config fallback | FRZ-TEN-05 | GAP-COV-TEN-CONFIG | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-CREDENTIALS | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r2_current_head.py::test_tx_r2_head_q01_qualification_closure_on_current_head` | No cross-tenant credential resolution | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_credentials.py::test_acceptance_ten_credentials_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-CREDENTIALS @ TES | credential bleed inject | FRZ-TEN-06 | GAP-COV-TEN-CREDENTIALS | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-CROSS-TENANT | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py::test_tx_r4_q05_adversarial_context_llm_cross_tenant` | Legal cross-tenant needs explicit authority | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_cross_tenant.py::test_acceptance_ten_cross_tenant_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-CROSS-TENANT @ TES | implicit admin bypass | FRZ-TEN-09 | GAP-COV-TEN-CROSS-TENANT | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-EVIDENCE | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py::test_tx_r4_q09_adversarial_functional_evidence_tenant` | Evidence cannot establish other tenant truth | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_evidence.py::test_acceptance_ten_evidence_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-EVIDENCE @ TES | accept wrong-tenant evidence | FRZ-TEN-07 | GAP-COV-TEN-EVIDENCE | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-FAIL-CLOSED | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py::test_tx_r4_q12_fail_closed_unknown_synthetic_context_caller` | Missing/invalid tenant fails closed | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_fail_closed.py::test_acceptance_ten_fail_closed_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-FAIL-CLOSED @ TES | silent default tenant | FRZ-TEN-10 / SYS-INV tenant | GAP-COV-TEN-FAIL-CLOSED | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-IDENTITY | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r1_current_head.py::test_tx_r1_head_q01_inventory_reconciles_twenty_two_mechanisms` | Canonical tenant identity where required | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_identity.py::test_acceptance_ten_identity_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-IDENTITY @ TES | `test_ten_identity_ent_skip_neg` | FRZ-TEN-01 | GAP-COV-TEN-IDENTITY | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-PROPAGATION | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_p0_current_head.py::test_tx_p0_head_q03_propagation_edges_reference_known_sources` | Tenant propagated across boundaries | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_propagation.py::test_acceptance_ten_propagation_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-PROPAGATION @ TES | `test_ten_propagation_drop_neg` | FRZ-TEN-02 | GAP-COV-TEN-PROPAGATION | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-PROVIDER | pytest_nodeid | `tests/qualification/existing_capability_configuration/test_configuration_realization_certification.py::test_cert_p02_resolver_target_provider_mismatch_before_strategy` | Provider/profile preserve tenant | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_provider.py::test_acceptance_ten_provider_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-PROVIDER @ TES | mismatched tenant pass | FRZ-TEN-05 | GAP-COV-TEN-PROVIDER | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-RECOVERY | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py::test_tx_r4_q08_adversarial_vendor_async_queue` | Recovery/replay keeps tenant | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_recovery.py::test_acceptance_ten_recovery_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-RECOVERY @ TES | wrong tenant resume | FRZ-TEN-08 | GAP-COV-TEN-RECOVERY | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-STATE | pytest_nodeid | `tests/qualification/tenant_x/test_tenant_x_r4_fast.py::test_tx_r4_q05_adversarial_context_llm_cross_tenant` | Cross-tenant state blocked | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_state.py::test_acceptance_ten_state_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-STATE @ TES | allow cross-tenant state | FRZ-TEN-04 | GAP-COV-TEN-STATE | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| TEN-TRACE | pytest_nodeid | `tests/qualification/trace_x/test_trace_x_p4_r1_recorder_tenant.py::test_txp4r1_q04_recorder_binding_resets_tenant` | Trace/diagnostic tenant attribution | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_trace.py::test_acceptance_ten_trace_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=TEN-TRACE @ TES | mixed-tenant trace | FRZ-TEN-07 | GAP-COV-TEN-TRACE | C035 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| VBN-CONTRACT-LEAKAGE | pytest_nodeid | `tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes` | No vendor concrete in contracts | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_contract_leakage.py::test_acceptance_vbn_contract_leakage_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=VBN-CONTRACT-LEAKAGE @ TES | vendor type in contract | FRZ-CTR-04 | GAP-COV-VBN-CONTRACT-LEAKAGE | C036 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| VBN-CORE-HARDCODING | pytest_nodeid | `tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero` | PROHIBITED unresolved = 0 | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_core_hardcoding.py::test_acceptance_vbn_core_hardcoding_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=VBN-CORE-HARDCODING @ TES | inject prohibited | FRZ-RPL-02 | GAP-COV-VBN-CORE-HARDCODING | C036 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| VBN-INVENTORY-DRIFT | pytest_nodeid | `tests/qualification/ebh_5/test_ebh_5_g06_enterprise_integrity_gates.py::test_ebh5_g06_01_roadmap_31_domains_accounted` | Inventory/matrix vs baseline | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_inventory_drift.py::test_acceptance_vbn_inventory_drift_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=VBN-INVENTORY-DRIFT @ TES | stale inventory PASS | FRZ-RPL-01 | GAP-COV-VBN-INVENTORY-DRIFT | C036 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| VBN-OBS-OPS-COUPLING | pytest_nodeid | `tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py::test_g5_otlp_event_export_sink_passes_payload_to_transport` | Obs/diag vendor replaceability | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_obs_ops_coupling.py::test_acceptance_vbn_obs_ops_coupling_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=VBN-OBS-OPS-COUPLING @ TES | hardcoded transport | FRZ-OBS-03 | GAP-COV-VBN-OBS-OPS-COUPLING | C036 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| VBN-REACHABILITY | pytest_file | `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py` | Production-effective reachability | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_reachability.py::test_acceptance_vbn_reachability_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=VBN-REACHABILITY @ TES (shared file — wrapper scopes assertion to VBN-REACHABILITY) | unreachable claimed ext | FRZ-RPL-07 | GAP-COV-VBN-REACHABILITY | C036 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| VBN-REPLACEABILITY-MATRIX | pytest_file | `tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_operational_gates.py` | RPL-A..H integrity | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_replaceability_matrix.py::test_acceptance_vbn_replaceability_matrix_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=VBN-REPLACEABILITY-MATRIX @ TES (shared file — wrapper scopes assertion to VBN-REPLACEABILITY-MATRIX) | broken RPL row | FRZ-RPL-06 | GAP-COV-VBN-REPLACEABILITY-MATRIX | C036 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| VBN-SHADOW-SELECTION | pytest_file | `tests/qualification/config_x/test_config_x_final_reconciliation_gates.py` | No shadow provider selector | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_shadow_selection.py::test_acceptance_vbn_shadow_selection_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=VBN-SHADOW-SELECTION @ TES (shared file — wrapper scopes assertion to VBN-SHADOW-SELECTION) | shadow registry | FRZ-CFG-03 | GAP-COV-VBN-SHADOW-SELECTION | C036 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| VBN-SILENT-FALLBACK | pytest_file | `tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py` | No silent vendor fallback | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_silent_fallback.py::test_acceptance_vbn_silent_fallback_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=VBN-SILENT-FALLBACK @ TES (shared file — wrapper scopes assertion to VBN-SILENT-FALLBACK) | silent None fallback | FRZ-CFG-04 | GAP-COV-VBN-SILENT-FALLBACK | C036 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |
| VBN-SQLITE-EFFECTIVE | pytest_file | `tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py` | SQLite effective authority qualified | Wrapper `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_sqlite_effective.py::test_acceptance_vbn_sqlite_effective_semantic_gate` executes manifest entrypoint via sandbox; assert outcome PASS for check_id=VBN-SQLITE-EFFECTIVE @ TES (shared file — wrapper scopes assertion to VBN-SQLITE-EFFECTIVE) | unauthorized sqlite owner | FRZ-RPL-05 | GAP-COV-VBN-SQLITE-EFFECTIVE | C036 | manifest outcomes + junit @ TES | OPEN until certifying FULL @ TEST_EXECUTION_SHA |

## Matrix F — ADV-01..31

| ADV | Threat | Injection | Test | Finding | Exit | Task |
| --- | --- | --- | --- | --- | --- | --- |
| ADV-01 | manifest shrink | 84-check manifest | `test_adv01_manifest_too_few_rejected` | load error | 3 | C019 |
| ADV-02 | duplicate ID | dup row | `test_adv02_duplicate_check_id_rejected` | ValueError | — | C019 |
| ADV-03 | Annex C drift | unknown ID | `test_adv03_unknown_check_id_rejected` | closed-world | 3 | C019 |
| ADV-04 | DAG cycle | cycle | `test_adv04_cycle_rejected` | ValueError | — | C030 |
| ADV-05 | false FULL dry | dry_run FULL | `test_adv05_dry_run_coverage_not_full` | INCOMPLETE | 2 | C004 |
| ADV-06 | FAST as FULL | FAST run | `test_adv06_fast_never_full_label` | PARTIAL label | 0/2 | C004 |
| ADV-07 | FAST floor skip | non-cert filter | `test_adv07_fast_floor_incomplete_exit` | INCOMPLETE | 2 | C004 |
| ADV-08 | FREEZE no baseline | no accepted **and** no bootstrap | `test_adv08_freeze_without_baseline_blocked` | BLOCKER | 1 | C009 |
| ADV-09 | baseline path jail | bad path | `test_adv09_baseline_path_jail` | ValueError | — | C001 |
| ADV-10 | orphan dep | unknown dep | `test_adv10_orphan_dep_rejected` | ValueError | — | C030 |
| ADV-11 | mutating sandbox | mutating check | `test_adv11_sandbox_rejects_mutating` | ValueError | — | C014 |
| ADV-12 | evidence tamper | bit flip | `test_adv12_tampered_evidence_fails` | verify false | — | C026 |
| ADV-13 | CHANGED-SINCE no sha | CLI | `test_adv13_changed_since_requires_since_sha_cli` | error | 3 | C025 |
| ADV-14 | REGRESSED | severity bump | `test_adv14_regressed_on_severity_increase` | REGRESSED | 1/2 | C022 |
| ADV-15 | extra artifact | add file | `test_adv15_extra_artifact_fails_closed_verify` | verify false | — | C026 |
| ADV-16 | path traversal | `../` entry | `test_adv16_path_traversal_rejected` | ValueError | — | C027 |
| ADV-17 | shell | shell=True | `test_adv17_sandbox_never_shell_true` | rejected | — | C014 |
| ADV-18 | digest swap | wrong digest | `test_adv18_manifest_digest_mismatch_finding` | BLOCKER | 1 | C028 |
| ADV-19 | wrong SHA | fake HEAD | `test_adv19_report_sha_matches_git_head` | BLOCKER | 1 | C012 |
| ADV-20 | tenant category | strip category | `test_adv20_tenant_enrich_restores_category` | TEN | — | C015 |
| ADV-21 | VBN category | strip category | `test_adv21_vbn_enrich_restores_category` | VBN | — | C011 |
| ADV-22 | non-atomic report | interrupt write | `test_adv22_no_partial_report_without_status` | no torn file | — | C013 |
| ADV-23 | FRZ lie | all AUD-03 | `test_adv23_frz_map_varies_by_check` | BLOCKER | 1 | C020 |
| ADV-24 | bootstrap accepted | bootstrap | `test_adv24_bootstrap_never_writes_accepted` | no accepted | — | C002 |
| ADV-25 | fake VBN NA | forged domains | `test_adv25_domain_gap_fails_full` | VBN BLOCKER | 1 | C043 |
| ADV-26 | empty CHANGED-SINCE | since=HEAD | `test_adv26_empty_changed_since_blocked` | EMPTY | 1 | C025 |
| ADV-27 | bad mode | invalid mode | `test_adv27_invalid_mode_rejected` | argparse | 3 | C005 |
| ADV-28 | CI gap | readiness script | `test_adv28_mechanical_readiness_ci_section` | fail | — | C054 |
| ADV-29 | non-cert bundle | non-cert run | `test_adv29_evidence_refused_when_non_certifying` | build error | — | C006 |
| ADV-30 | L3 false qual | exit 3 | `test_adv30_l3_full_strict_exit_zero_only` | assert | — | C031 |
| ADV-31 | status mutation | PASS+findings | `test_adv31_mutated_status_not_clean` | internal | 3 | C044 |

## Matrix G — D01–D31

| Domain | Name | ENT check | Test | TaskID | Qual @ HEAD |
| --- | --- | --- | --- | --- | --- |
| D01 | RELATIONAL_PERSISTENCE | VBN-SQLITE-EFFECTIVE | `test_ebh_5_l2_integrity_gates` | C008,C011 | PLANNED @ HEAD |
| D02 | NOSQL_DOCUMENT | VBN-INVENTORY-DRIFT | g06 domain test | C008,C011 | PLANNED @ HEAD |
| D03 | KEY_VALUE | VBN-SHADOW-SELECTION | config_x gates | C008,C011 | PLANNED @ HEAD |
| D04 | CACHE | VBN-INVENTORY-DRIFT | `test_ebh5_g06_01_roadmap_six_na_domains_have_discovery_evidence` | C008,C011 | PLANNED @ HEAD |
| D05 | VECTOR | VBN-REPLACEABILITY-MATRIX | RPL gates | C008,C011 | PLANNED @ HEAD |
| D06 | GRAPH | VBN-REPLACEABILITY-MATRIX | ebh_5 factory rpl | C008,C011 | PLANNED @ HEAD |
| D07 | OBJECT_BLOB | VBN-REACHABILITY | behavioral package C | C008,C011 | PLANNED @ HEAD |
| D08 | EVENT_STREAMING | VBN-OBS-OPS-COUPLING | harness_01 gates | C008,C011 | PLANNED @ HEAD |
| D09 | QUEUES_WORKERS | VBN-SILENT-FALLBACK | W4 qual slice | C008,C011 | PLANNED @ HEAD |
| D10 | DISTRIBUTED_COORDINATION | VBN-INVENTORY-DRIFT | g06 NA test | C008,C011 | PLANNED @ HEAD |
| D11 | TRANSACTION_IDEMPOTENCY | VBN-INVENTORY-DRIFT | state_x qual | C008,C011 | PLANNED @ HEAD |
| D12 | CHECKPOINTS_EXECUTION | VBN-REACHABILITY | ee_a1 gate | C008,C011 | PLANNED @ HEAD |
| D13 | AGENT_MEMORY | VBN-CORE-HARDCODING | ebh6 hardcoding register | C008,C011 | PLANNED @ HEAD |
| D14 | RAG_RETRIEVAL | VBN-CONTRACT-LEAKAGE | ebh_2i gate | C008,C011 | PLANNED @ HEAD |
| D15 | LLM_EMBEDDING | VBN-REPLACEABILITY-MATRIX | ebh_5 dual slug rpl | C008,C011 | PLANNED @ HEAD |
| D16 | AGENT_RUNTIMES | VBN-SHADOW-SELECTION | harness gates | C008,C011 | PLANNED @ HEAD |
| D17 | HOSTING_SUBSTRATE | VBN-REACHABILITY | host replaceability tests | C008,C011 | PLANNED @ HEAD |
| D18 | EXTERNAL_MCP | VBN-CORE-HARDCODING | ebh6 register | C008,C011 | PLANNED @ HEAD |
| D19 | LOGGING | VBN-OBS-OPS-COUPLING | runtime_event_delivery_wiring | C008,C011 | PLANNED @ HEAD |
| D20 | METRICS | VBN-OBS-OPS-COUPLING | W5 gates | C008,C011 | PLANNED @ HEAD |
| D21 | TRACING | VBN-OBS-OPS-COUPLING | ADR-OBS-005 regressions | C008,C011 | PLANNED @ HEAD |
| D22 | DIAGNOSTICS | VBN-OBS-OPS-COUPLING | harness qual | C008,C011 | PLANNED @ HEAD |
| D23 | ALERTING | VBN-INVENTORY-DRIFT | g06 | C008,C011 | PLANNED @ HEAD |
| D24 | AUDIT_EVIDENCE | VBN-INVENTORY-DRIFT | trace_x qual | C008,C011 | PLANNED @ HEAD |
| D25 | SECRETS | VBN-SILENT-FALLBACK | config_x fail-closed | C008,C011 | PLANNED @ HEAD |
| D26 | AUTH_IDENTITY | VBN-CONTRACT-LEAKAGE | gov_x2 qual | C008,C011 | PLANNED @ HEAD |
| D27 | CONFIG_PROFILE | VBN-SHADOW-SELECTION | config_x final gates | C008,C011 | PLANNED @ HEAD |
| D28 | NETWORK_HTTP | VBN-INVENTORY-DRIFT | g06 NA | C008,C011 | PLANNED @ HEAD |
| D29 | TELEMETRY_OTLP | VBN-OBS-OPS-COUPLING | W5 batch tests | C008,C011 | PLANNED @ HEAD |
| D30 | EVAL_ANALYTICS | VBN-INVENTORY-DRIFT | g06 NA | C008,C011 | PLANNED @ HEAD |
| D31 | DISCOVERED | VBN-INVENTORY-DRIFT | inventory integrity | C008,C011 | PLANNED @ HEAD |

See [ANNEX_D_VBN_DOMAINS_1_31.md](./ANNEX_D_VBN_DOMAINS_1_31.md).

## Matrix A — FRZ-AUD-01..10

| FRZ | Requirement | Task mapping | Status @ audited HEAD | Formal PASS |
| --- | --- | --- | --- | --- |
| FRZ-AUD-01 | runner exists | import | DONE mechanical | operator post-B6 |
| FRZ-AUD-02 | modular checks | C019,C030,C034–C036 | OPEN @ TES | operator post-B6 |
| FRZ-AUD-03 | FRZ per check | C020,C021 | OPEN | operator post-B6 |
| FRZ-AUD-04 | reports | C013,C062 | PARTIAL | operator post-B6 |
| FRZ-AUD-05 | exact SHA | C059,C048 | PARTIAL | operator post-B6 |
| FRZ-AUD-06 | baseline drift | C001,C022 | PARTIAL | operator post-B6 |
| FRZ-AUD-07 | RESOLVED class | C071,C023 | PARTIAL | operator post-B6 |
| FRZ-AUD-08 | exit codes | C004,C067 | PARTIAL | operator post-B6 |
| FRZ-AUD-09 | modes L3 | C009,C031–C032 | NOT_STARTED | operator post-B6 |
| FRZ-AUD-10 | CI policy | C064,C047,C046 | OPEN | operator post-B6 |

## Matrix B — EA summary

P0: C031–C036,C066,C074; EA-04 external C047 (OD-ENT-02).

## §14 Task DAG

### 14.1 Dependencies

| TaskID | Depends on | Batch |
| --- | --- | --- |
| C001 | — | W-A |
| C002 | C001 | W-A |
| C003 | C001,C006 | W-A |
| C004 | — | W-A |
| C005 | C004 | W-A |
| C006 | — | W-A |
| C007 | C002 | W-E |
| C008 | — | W-A |
| C009 | C001,C002 | W-B |
| C010 | C008 | W-B |
| C011 | C008 | W-B |
| C012 | — | W-B |
| C013 | — | W-B |
| C014 | — | W-B |
| C015 | — | W-B |
| C016 | — | W-B |
| C017 | — | W-B |
| C018 | C011 | W-B |
| C019 | — | W-B |
| C020 | — | W-B |
| C021 | C020 | W-B |
| C022 | — | W-B |
| C023 | C022 | W-B |
| C024 | — | W-B |
| C025 | C024 | W-B |
| C026 | C006 | W-B |
| C027 | C026 | W-B |
| C028 | C019 | W-B |
| C029 | C024 | W-B |
| C030 | C019 | W-B |
| C031 | C004,C011 | W-D |
| C032 | C002,C009,C011,C031 | W-D |
| C033 | C004 | W-C |
| C034 | C033 | W-C |
| C035 | C033 | W-C |
| C036 | C033 | W-C |
| C037 | C004,C026 | W-D |
| C038 | C037 | W-D |
| C039 | C037 | W-D |
| C040 | — | W-A |
| C041 | C040 | W-D |
| C042 | C003,C026 | W-D |
| C043 | C008 | W-D |
| C044 | C026 | W-D |
| C045 | — | W-E |
| C046 | C007 | W-E |
| C047 | — | W-E |
| C048 | C031 | W-G |
| C049 | C026 | W-G |
| C050 | C026,C031 | W-G |
| C051 | C031,C032,C040 | W-D |
| C052 | C047 | W-G |
| C053 | C050 | W-G |
| C054 | C050,C059,C064 | W-G |
| C055 | — | W-G |
| C056 | C055 | W-G |
| C057 | — | W-G |
| C058 | C050,C053 | W-G |
| C059 | C004,C045 | W-E |
| C060 | C004,C005,C062 | W-E |
| C061 | C060,C062 | W-E |
| C062 | C004 | W-E |
| C063 | C059,C062 | W-E |
| C064 | C047,C059,C063 | W-G |
| C065 | C003,C026 | W-D |
| C066 | C031,C032,C051,C074 | W-G |
| C067 | C004 | W-A |
| C068 | C006,C067 | W-A |
| C069 | C006 | W-A |
| C070 | C060,C061 | W-E |
| C071 | C023 | W-B |
| C072 | C012,C024 | W-B |
| C073 | C006 | W-A |
| C074 | C031,C032 | W-D |
| C075 | C041,C048,C055 | W-G |

### 14.2 Topological order

**Acyclic:** YES · **Nodes:** 75

C001 → C004 → C006 → C008 → C012 → C013 → C014 → C015 → C016 → C017 → C019 → C020 → C022 → C024 → C040 → C045 → C047 → C055 → C057 → C002 → C005 → C033 → C062 → C067 → C003 → C026 → C069 → C073 → C010 → C011 → C043 → C028 → C030 → C021 → C023 → C025 → C029 → C072 → C041 → C059 → C052 → C056 → C007 → C009 → C034 → C035 → C036 → C060 → C068 → C027 → C037 → C042 → C044 → C049 → C065 → C018 → C031 → C071 → C063 → C046 → C061 → C038 → C039 → C032 → C048 → C050 → C064 → C070 → C051 → C074 → C075 → C053 → C054 → C066 → C058

**Wave F infra:** implement **C051** + **C074** (W-D) before Wave F qual; **C066** readiness only after TES artifacts.

## §9 Qualification and Wave F

1. Complete remaining W-A..W-E implementation deltas @ audited baseline.
2. Commit/push per wave; no production changes in SSOT-only repair sessions.
3. Clean checkout @ pin candidate; operator selects **TEST_EXECUTION_SHA**.
4. **L1/L2** pytest subsets (`ent_l1`, `ent_l2`, markers per C040).
5. **C051/C074** orchestrator must exist before final Wave F qual.
6. Real **85/85** semantic execution (Matrix E wrappers + manifest outcomes).
7. Certifying **FULL** (0 skipped, exit 0, FULL coverage label).
8. **ADV31** fault-injection suite @ TES.
9. First **FREEZE candidate** (no accepted baseline without OD).
10. **B5** evidence publication tree @ descendant of TES.
11. Independent **B6** (agent prepares dossier only).
12. **OD-ENT-01** → accepted baseline → normal FREEZE → **OD-ENT-08**.

## §13 Agent instruction

Execute **§14.2 topological order** within wave constraints (no backward cross-wave edges). **W-0:** C075-S1 inventory only. **W-A..W-E:** implementation deltas from Matrix C; **git commit and push** each wave. Pytest logs under `.tmp/session/ent-audit-x-*`. Before Wave F: **C031,C032,C051,C074** must be IMPLEMENTED. **C066** readiness script runs only after TES artifacts exist (fail-closed). **Operator only:** OD-ENT-00, OD-ENT-01, OD-ENT-02, OD-ENT-08. **GitHub admin:** branch protection (BLK-CI-ENFORCE). **Independent auditor:** Protocol 03 B6 (BLK-ENT-B6). Agent does **not** declare FRZ/B6/CLOSED/OD APPROVED.

## §18 Mechanical publication self-test

| Check | Result |
| --- | --- |
| Matrix C 75 rows | PASS |
| No backward cross-wave edges | PASS |
| Matrix E 85 non-generic oracles | PASS |
| GAP-RC 33 set equality | PASS |
| GAP-B 6 external blockers listed | PASS |
| ADV 31 / Matrix G 31 / FRZ-AUD 10 | PASS |
| FP-04 matches Matrix C impl counts | PASS |
| C075-S1/S2 documented | PASS |
| PL-07 collect-only 282 @ audited baseline | PASS |
| ECAP B5 before B6 in PL-08 | PASS |
| Validator exit code: 0 | PASS |

**Overall publication gate:** PASS

Runtime qualification @ TEST_EXECUTION_SHA: OPEN (not publication gate).

## Appendix B — C065..C075

| TaskID | Scope | Owner | Deps | Pos test | Neg test | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| C065 | Extend verify; wire promote; operator_records path | baseline.py::verify_operator_decision_record | C003,C026 | `promote w/ valid ladder fixture` | `forged JSON` | no promote without B6 pin |
| C066 | Rewrite checker; JSON report | mechanical_readiness_check.py::main | C031,C032,C051,C074 | `all gates true` | `missing L3` | TECHNICAL_READY only |
| C067 | Flag manifest; exit code | runner.py,modes.py | C004 | `dry-run exit 2` | `dry-run exit 0 cert` | no dry-run CLEAN certifying |
| C068 | Cross-check manifest | evidence.py::build_evidence_manifest | C006,C067 | `real FULL bundle` | `dry-run bundle fail` | bundle rejected |
| C069 | Add dependency; validate all artifacts | validators.py | C006 | `valid pass` | `extra props fail` | all artifacts |
| C070 | Replace OUT= pattern | audit.bat,audit.sh, CLI | C060,C061 | `parallel runs` | `collision` | unique paths |
| C071 | Filter in aggregate | modes.py::aggregate_status | C023 | `resolved PASS` | `blocker resolved` | no false BLOCK |
| C072 | Data file + tests | changed_since.py, impact JSON | C012,C024 | `mapped change` | `unmapped BLOCK` | fail-closed |
| C073 | Catalog versions | evidence_artifact_catalog.v1.json | C006 | `profile match` | `missing artifact` | mode closure |
| C074 | Harness + markers | tests/.../l3/test_l3_qual_wave.py (new) | C031,C032 | `85/85 PASS` | `partial fail` | strict exit 0 |
| C075 | Doc pass | this doc, Annex F/H, ladder, dossier | C041,C048,C055 | `link lint` | `stale pin` | SSOT match HEAD |

---
*Execution SSOT @ TEST_EXECUTION_SHA `b8bb88c9db1060153ec0a6a29bbb79ba16e00078`; documentation HEAD `60bf2f43b3d5c33b9864574c5296e1dd864fc7a5`.*

