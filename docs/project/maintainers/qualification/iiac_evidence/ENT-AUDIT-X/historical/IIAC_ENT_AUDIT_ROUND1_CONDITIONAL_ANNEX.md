> **HISTORICAL ANNEX — NOT A B6 ACCEPTANCE RECORD.** Round 1 independent audit @ audit_execution_sha `15347255ac62dbccfb7b64762b4c0f8af7ec38fa`. Verdict **CONDITIONAL** (2 HIGH / 11 MEDIUM). Corrective loop 1 published as TES `62d1c615` / EVIDENCE `163ef0c0`. Source dossier: `.tmp/session/ent-w2-audit/` (session id unknown).
# IIAC independent audit - ENT-AUDIT-X (formal B6 acceptance of the implementation)

Role: `independent-quality-auditor` | Policy: `IIAC-2026-11-05.1` (IN_FORCE, activation `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d`) | Profile: `iiac-v1`
Assurance level: `process_enforced` (see section 9). Auditor session id: not known to the auditor, left for the orchestrator.
Audit date (UTC): 2026-11-08. Machine: Windows 11 Home 10.0.26200, Python 3.12.11, uv 0.13.4, git 2.46.0.windows.1, gh 2.94.0, `uv.lock` sha256 `d9bef8dd6365f495e5cfbe286a9430df256295b7f2c9a0f453513baa05150519`.

## 0. Verdict

**CONDITIONAL.** The implementation itself reproduces: a fresh certifying replay at the audit SHA passes in full (qual wave PASS, FULL 85/85 certifying, FREEZE bootstrap-candidate 85/85, 293 passed / 0 failed in the unit+integration+l3 suite, Linux CI nightly FULL 85/85 at the same SHA). The verdict is not PASS or PASS_SCOPED because two HIGH defects sit in the evidence-integrity and operator-trust boundaries that formal B6 acceptance relies on (F-01 B5 evidence is verifiable only on a CRLF Windows checkout and is skipped in Linux CI; F-02 `promote-baseline` can be driven by a forged world in the current working directory - exploited in probe Q02), and eleven MEDIUM findings show that several qualification/status claims are stronger than the evidence behind them.

Counts of findings: BLOCKER 0, HIGH 2, MEDIUM 11, LOW 5, INFO 2 (20 total; `findings.json`). Nothing was repaired by the auditor.

Conditions that must be met (by an implementer loop, then a re-audit of the delta) before the certifier may publish a B6 acceptance:

1. F-01 - the B5 evidence tree must be verifiable on the bytes stored in git (it verifies only on a CRLF/autocrlf=true Windows checkout and is silently skipped in every Linux CI run).
2. F-02 - `promote-baseline` must not evaluate the OD-ENT-01 trust gate against the current working directory (and the unsigned human path must not remain reachable under IIAC).
3. Either fix F-10 and F-11 (the COMPENSATING and LIVE OD-ENT-02 routes cannot be satisfied as shipped) or have the operator record that these verifiers are unreliable and will be bypassed by a manual audit; this does not block B6 itself but blocks closure.
4. F-05 / F-13 - Matrix C / Matrix D status must stop citing unpublished artefacts and contradictory closure cells (status honesty), or the certifier must carry them as stated limitations.

What the audit supports: the implementation at `89c76eb7...` (unchanged product and qualification-sensitive code at the audit SHA) passes its own qualification on a fresh, independent execution (`run_qual_wave.py --profile certifying` exit 0, status PASS, all 11 stages exit 0, 179 minutes on a loaded machine), the closed-world 85-check manifest, Annex C, Annex G2, Matrix E and the 85 acceptance wrappers are identical sets, the six NOT_QUALIFIED Matrix C rows are correctly NOT_QUALIFIED, and every fail-closed probe on the manifest, runner, sandbox path jail and delegated-record verifier behaved fail-closed. What it does not support: treating B5 evidence integrity as platform independent (F-01), treating the OD-ENT-01 trust boundary as closed (F-02/F-03), and treating the "ADV-01..31" stage as an adversarial proof of 31 threats (F-04).

## 1. Scope lock (task 1)

### 1.1 Pins and ancestry

All commands run in `D:\Projekty\intergrax` (main checkout, read-only use) and in the audit worktree `D:\Projekty\_wt\intergrax\ent-w2-audit` created with `git worktree add --detach ... 15347255ac62dbccfb7b64762b4c0f8af7ec38fa`. `git rev-parse HEAD` of the worktree: `15347255ac62dbccfb7b64762b4c0f8af7ec38fa` (= `origin/development` after `git fetch origin development`, = main checkout HEAD, branch `development`).

| Role | SHA | `git cat-file -t` | `merge-base --is-ancestor <sha> 15347255...` |
|---|---|---|---|
| audited_implementation_sha = test_execution_sha | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` | commit | yes |
| evidence_publication_sha | `300859537f12a1de7bf3d75fd685bc3a5c727183` | commit | yes (and TES is an ancestor of it: yes) |
| audit_execution_sha | `15347255ac62dbccfb7b64762b4c0f8af7ec38fa` | commit | (HEAD) |
| IIAC activation | `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` | commit | yes |
| EVOL-X formal closure | `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca` | commit | yes |
| EVOL IIAC B6 | `1dd769331506149a1ee2f411c2bc62e85e5bbf25` | commit | yes |
| ENT handoff | `7a4f279fd8c0986431941c77283d872ea5b0c1ba` | commit | yes |
| technical re-audit (evidence only) | `0e538269a7188075003b9ff2c7cce2545d8f1ccb` | commit | yes |

Additional CI SHAs checked: `162acea31833a6d40d3a5e8e79638db7e443ce3e` (nightly 42015310879) and `618e234ee4dc41aa3673de5e1e81271f6039ca6c` (nightly 41800553851) are ancestors of the audit SHA and descendants of TES.

The B5 tree `qualification_at_89c76eb7...` was introduced by exactly one commit (`300859537`, `git log -- <tree>`) and is byte-identical between `300859537` and the audit SHA (`git diff --stat 300859537 15347255` on the tree: empty). Between TES and `300859537` the only non-doc change is `scripts/enterprise_audit/wbs_reconciliation_lib.py`.

### 1.2 Delta TES -> audit SHA (`git diff --name-only 89c76eb7...15347255ac62...` = 614 paths, `delta_paths.txt`, classified in `delta_classified.tsv`)

| Class | Paths |
|---|---|
| Product paths (`intergrax/`, `agents/`, `applications/`) | **0** |
| Qualification-sensitive (`intergrax/platform/enterprise_audit/`, `run_enterprise_audit.py`, `qual_wave_lib.py`, `run_qual_wave.py`) | **0** (also 0 for TES..`300859537`, TES..`618e234`, TES..`162acea`) |
| `scripts/enterprise_audit/` other tooling (audited here) | 5: `iiac_baseline_promotion.py` (new), `master_plan_ssot_lib.py`, `mechanical_readiness_check.py`, `reconcile_c075_s2.py`, `wbs_reconciliation_lib.py` |
| ENT qualification tests (audited here) | 6: `test_baseline_promotion_iiac.py`, `test_evidence_publication_strict.py`, `test_formal_closure_preflight_gates.py`, `test_od_ent_02_verifiers.py`, `test_sandbox_redaction_sensitive_markers.py`, `test_validate_post_implementation_master_plan.py` |
| ENT preflight and OD-ENT-02 verifiers (audited here) | 3: `scripts/qualification/ent_audit_x_formal_closure_preflight.py`, `scripts/governance/verify_ent_audit_branch_protection.py`, `scripts/governance/verify_od_ent_02_compensating_control.py` |
| Governance / IIAC tooling `scripts/governance/*` (other) | 32 |
| EVOL-X qualification scripts `scripts/qualification/*` (non-ENT) | 12 |
| Other scripts (`scripts/ci`, `scripts/dev`, `scripts/__init__`) | 8 |
| Non-ENT tests (`tests/unit/governance`, `tests/unit/qualification`, one fixture) | 18 |
| CI workflow `.github/workflows/unit-tests.yml` (adds `governance-fast`, `evol-x-b4-preflight`; no ENT workflow changed) | 1 |
| `.github/repo-management/repository-metadata.json` | 1 |
| Docs: ENT-AUDIT-X / IIAC evidence / other qualification / other | 33 / 217 / 116 / 32 |
| Governance tickets / policy / agent-env / adapters | 38 / 83 |
| Root metadata (`pyproject.toml` license text + ruff exclude only; `.gitattributes`; `.gitignore`; README; LICENSE; ...) | 9 |

`uv.lock` is unchanged. `pyproject.toml` changes only the license text and a ruff `extend-exclude` for `iiac_evidence`. The repository's own overlay statement "tooling only in scripts/enterprise_audit/ (3 files) and 2 ENT tests" describes the state at `ba07ebaa`; at the audit SHA the delta is 5 scripts and 6 tests (lane A/B/C merges `#400-#404`). No classification changes the conclusion: product code and qualification-sensitive code are identical to TES.

Delta reviewed line by line: `master_plan_ssot_lib.py` (D-1 fix: exact-line disposition check with mutation tests - sound), `iiac_baseline_promotion.py` (section 4.4), `formal_closure_preflight.py` (section 4.5), both OD-ENT-02 verifiers (section 4.6), `mechanical_readiness_check.py` and `wbs_reconciliation_lib.py` (section 6, F-05/F-13/F-16), the six tests.

## 2. Mode A - evidence integrity (task 2)

Tree: `docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/` (8 files).

- Index vs files: `sha256sum` of the 8 files in the worktree equals every digest in `publication_index.v1.json` (6 L3 artifacts incl. `evidence_manifest.v1.json`, plus `qual_wave_report.json` `fbffc784...`). The six schemas under `intergrax/platform/enterprise_audit/data/schemas/` are applied by the verifier (`validate_b5_handoff_bundle`, `verify_closed_evidence_bundle`).
- Repository verifier `verify_b5_publication_tree(repo_root, tree, expected_qualification_sha=TES)` on a faithful copy: `()` - no errors (probe A00). It is the same verifier the repo's `mechanical_readiness_check` uses.
- Important caveat (**F-01**): the digests are those of CRLF working-tree bytes. The git blobs are LF (`git ls-files --eol`: `i/lf w/crlf`; blob sha256 of `enterprise_audit_report.json` is `c0da90e5...`, indexed digest `87f56662...`). Probe A01 (the identical tree materialised from `git cat-file blob` bytes, i.e. what any Linux/CI checkout has): **8 errors**, every artifact digest mismatches. In CI the two strict-B5 tests are skipped (run 42467338307 at the audit SHA and nightly 42015310879).
- Tamper probes on COPIES in a scratch repo (alternates to the main object store, detached at the audit SHA; `probes/modeA_probe.py`, `probes/modeA_results.json`):

| Case | Tamper | Verifier result |
|---|---|---|
| A02/A03 | one byte appended to report.json / PASS->FAIL in report.md (digests untouched) | rejected (artifact digest + evidence-manifest mismatch) |
| A04 | missing `vbn_domains.v1.json` | rejected (missing file, mandatory artifact missing, manifest mismatch) |
| A05/A22/A21 | missing index / invalid JSON / schema violation (`qualification_sha` removed) | rejected |
| A06 | one hex char altered in an index digest | rejected |
| A07/A08 | wrong `qualification_sha` (nonexistent object) / wrong expected pin (stale `87203c1b`) | rejected (pin mismatch, dir-name mismatch, not ancestor) |
| A09 | tree re-labelled to a real commit that is not an ancestor of HEAD | rejected ("not an ancestor of HEAD") |
| A10/A11 | extra file in the tree / in the L3 bundle | rejected |
| A12/A13/A14 | duplicate indexed path / `../` traversal path / qual_wave path redirect | rejected |
| A15 | `qual_wave_report.json` PASS->FAIL, index not updated | rejected |
| A19 | report `status=FAIL` with `exit_code=0`, all digests re-sealed | rejected (semantic check) |
| A23 | symlink substitution | NOT RUN (symlink creation not available to the unprivileged session) |
| **A16** | qual_wave_report `status=FAIL`, a failing stage, index re-sealed | **accepted** |
| **A17** | evidence_manifest `qualification_sha`/`audit_execution_sha`/`check_manifest_digest` replaced, re-sealed | **accepted** |
| **A18** | report `current_sha` + `execution_manifest.qualification_sha` replaced, re-sealed | **accepted** |
| **A20** | `execution_manifest.checks_executed=84, checks_skipped=1`, re-sealed | **accepted** |

Un-resealed tampering is detected in every tested case; a re-sealing adversary is not detected for content bindings (**F-08**). Integrity of the published bytes therefore rests on the git commit identity `300859537`, which was verified (tree unchanged since). A defect in the verifier that I hit unintentionally while building the harness: an empty/invalid `evidence_manifest.v1.json` makes `verify_evidence_manifest` raise `JSONDecodeError` (evidence.py:147) instead of returning an error (fail-closed by crash; part of F-08).

Content checks of the retained evidence itself (read, not trusted): `qual_wave_report.json` status PASS, `pinned_sha == qualification_sha == 89c76eb7`, `clean_tree: true`, stages l1 111, l2 42, semantic_acceptance 63, semantic_matrix 85, adv31 33, ten13 13, vbn9 9, d31 2, l3_cli_full exit 0, l3_validate "FULL 85/85", first_freeze_candidate exit 0 (detail "retry after stale baseline removal ENT-W8-B6-001"); report: mode FULL, certifying, `exit_code 0`, `findings []`, 2 limitations; all 85 outcomes executed, none skipped, none with "skipped"/"no tests ran" in stdout_tail; no secret-pattern hit in the whole tree (grep for `ghp_`, `AKIA`, `Bearer`, PEM headers, `sk-`, URL credentials: none).

## 3. Mode B - retained CI evidence read with `gh` (task 3)

Commands: `gh run view <id> --json ...`, `gh run view <id> --log` (logs saved to `ci/`), `gh run list`, `gh api .../commits/<sha>/check-runs`.

| Run | Workflow / event | Head SHA | Conclusion | What it actually ran |
|---|---|---|---|---|
| 42015310879 | Enterprise audit nightly, `workflow_dispatch` | `162acea31833a6d40d3a5e8e79638db7e443ce3e` (not TES) | success | `uv run pytest tests/qualification/ent_audit_x -q` = **185 passed, 179 skipped** (the 85 acceptance wrappers, the 85 matrix tests, L3 and B5-verifier tests skip); then `run_enterprise_audit.py --mode FULL` real: `Mode FULL certifying_run=True`, Git SHA 162acea3, **85/0/85 executed/skipped/planned, 0 failed, PASS exit 0**, 850 s. It reused TES via a "doc-only" delta (0 product or sensitive paths TES..162acea, verified). |
| 41800553851 | nightly `workflow_dispatch` | `618e234ee4dc...` | success | cited as L3 in the trust chain; ancestor of audit SHA, TES..618e234 has 0 product/sensitive paths |
| 41804655934 | Regression gate push | `0e538269a718...` | success | cited as L4 |
| 42467338307 | Enterprise audit push (`development`) | **`15347255ac62...` = audit SHA** | success | `ent-audit-fast`: pytest unit+integration+l3 = **290 passed, 94 skipped**; FAST CLI non-certifying: PASS, 85/0/85, exit 0; `ent-audit-pr-gate` skipped (push event) |
| 42467338306 | Regression gate push | audit SHA | success (run level) | jobs `ci-smoke`, `governance-fast`, `evol-x-b4-preflight` success; **`governance-policy-pin` failed at "Install uv"** (advisory job, infrastructure); the other jobs are conditional skips |
| 42468352524 | nightly `schedule` | audit SHA | in_progress at the time of reading | **completed success** (09:43-09:56Z): `pytest tests/qualification/ent_audit_x` = 293 passed, 179 skipped (same two strict-B5 skips); then FULL real, certifying, Git SHA = audit SHA, **85/0/85, PASS, exit 0** - a Linux certifying FULL at the audit SHA |
| 42200464814 | nightly `schedule` | `e1bd9eb11bbc` | success (2026-11-06) | shows the overlay's "last nightly success 42015310879" is incomplete |
| 42359724126, 42462029231, 42462046714 and PR runs | nightly 2026-11-07 / pushes at 08:13-08:14Z / PRs | various | failure | all D-1 (`test_validate_post_implementation_master_plan`, 2 tests) or `setup-uv` HTTP 504; fixed at the tip (F-19) |

Observations: (1) no retained CI run executes the 85 acceptance wrappers, the 85 matrix test or the L3 FULL/FREEZE tests (all skip by env gate); the only certifying evidence of those is the local `qual_wave_report` at TES and my replay (section 5). (2) The CI nightly FULL real run is a genuine independent 85/85 certifying execution on Linux at `162acea3`. (3) CI does not verify B5 integrity (F-01). (4) Check-run names are bare job names (relevant to F-11).

## 4. Mode C - adversarial (task 4)

All probes are in `probes/`, logs in `logs/`, machine-readable results in `probes/modeC_probe*_results.json`.

### 4.1 Repository adversarial suite
`tests/qualification/ent_audit_x/integration/test_adversarial_*.py` and the sandbox/redaction/path-escape/status-mutation unit tests all pass in the fresh run (section 5). Assessment of their strength: **F-04** (23/29 Matrix F names missing; most ADV tests are existence checks). They are not wrong, but they do not prove what Matrix F says they prove.

### 4.2 Closed world, sandbox, runner (probe 1; 40 records: 32 as expected, 5 unexpected, 3 informational)
Fail-closed as expected: 84- and 86-check manifests (C01/C02), duplicate id (C03), orphan dependency (C04), cycle (C05), unknown mode (C06), unknown entrypoint kind executes to a BLOCKER, exit 3 (C07b), `..`, absolute, UNC and directory-junction escapes (C08a-f), nonexistent script (C09, exit 2), mutating check (C10), failing test and empty module (C12b/c), real 3 s timeout killing a grandchild (C16), FULL dry-run INCOMPLETE exit 2 / PARTIAL (C17/C18), FREEZE without baseline BLOCKED (C19), certifying `--check-id` BLOCKED (C20/C27), empty CHANGED-SINCE BLOCKED (C21), FREEZE vs accepted baseline whose SHA is not in history BLOCKED `FREEZE:BASELINE_ANCESTRY` (C22), baseline dir outside the jail (C23), CLI invalid mode exit 2 (C24), CHANGED-SINCE without or with a nonexistent `--since-sha` exit 3 (C25/C26), evidence bundle on a non-certifying run refused (C28), `promote-baseline` with missing record exit 1 (C29), PASS relabel of a report with a BLOCKER finding rejected (C30b).
Unexpected (findings): C11 option injection `--version` counted PASS; C12a all-skipped and C12d xfail-only counted PASS (**F-07**); C13 10/15 secret shapes survive `_redact` (**F-06**); C30a an INCOMPLETE dry-run report relabelled PASS/exit 0 passes the semantic validator (**F-08**). Informational: C14 environment inherited, C15 `mutating=false` check wrote a file inside the repo (**F-06**).

### 4.3 Secrets in logs, argv, bundles
argv: the sandbox argv contains only `uv run pytest|python <entrypoint>` (sandbox.py:75-80); no secret passes through argv. Logs/bundles: stdout/stderr tails are persisted in `enterprise_audit_outcomes.json`; `_redact` is a 4-marker line filter (F-06); no secret pattern exists in the published B5 tree. The report-level test named "contains_no_environment_secrets" is vacuous (F-04).

### 4.4 OD-ENT-01 promotion records (probe 2, P00-P14; probe 3 for the CLI)
Fail-closed (REJECTED with the expected reason): agent/cursor-origin and case variants (P02/P03/P05), `operator_origin=false` (P04), record for another decision `OD-ENT-08` (P06), blank signature_line (P07), delegation without operator signature line (P08), template-blank timestamp (P09), designated session equal to the implementation session (P12), manifest verdict CONDITIONAL (P14). The 27 negative cases in `test_baseline_promotion_iiac.py` also pass.
Not fail-closed: **P01** a human-path record `{decision_id, operator_origin:true, 4 pins}` with no signature is ACCEPTED; **P10** a delegated record whose delegation and record commits are authored by `claude-agent-implementer` is ACCEPTED (no authorship/signature binding; process_enforced); P11 designated == certifier ACCEPTED (info); P13 the promoter reads only 7 manifest fields and does not run `verify_iiac_audit_manifest` (info) (**F-03**).
**Probe 3 (cwd confusion, F-02; `probes/modeC_probe3.py`, `logs/modeC_probe3.log`)**: a forged OD-ENT-01 record (`operator_origin:true`, ladder B6 pin set to an unrelated real ancestor commit, dossier listing the SHAs) placed in an attacker-built directory with its own git repo, plus a copy of the real B5 bundle and a candidate baseline. Q01: `run_enterprise_audit.py promote-baseline ...` with cwd = the audited repo -> exit 1 "OD-ENT-01 record must live under operator_records/", nothing written. **Q02: the same command with cwd = the forged directory -> exit 0, `ENTERPRISE_AUDIT_BASELINE.json` written under the real repo's jailed baseline dir.** The scratch baseline dir was removed afterwards; the real `tests/qualification/ent_audit_x/` was never given an accepted baseline.

### 4.5 Formal-closure preflight (Gate A, Gate B) with unset (`*unset*`) ladder pins
- Real tree, `--gate A` and `--gate B`: exit 1, 18 and 19 errors (F01/F02): missing pass record, missing manifest, `formal_b6_status`, `frz_aud_status OPEN`, ladder pins unset, `od_ent_02_mode` unset, FRZ-AUD-01..10 OPEN, ... Fail-closed.
- Pin cell variants (F13): `*unset*`, blank, `TBD`, 39-hex, uppercase hex, hex + trailing text all give "unset or malformed" for both B6 and ACCEPTED_BASELINE; B6 set but status JSON null -> mismatch error (F11); B6 = a real non-ancestor commit -> "not an ancestor of HEAD" (F12); Gate A with `formal_closure_publication_sha` set -> error (F14). The old fail-open on `*unset*` is gone.
- Not fail-closed: **F15** a fabricated manifest + one-line pass record produces no manifest-related error (**F-09**); **O01** the compensating route cannot pass by default wiring with a ruleset control (**F-10**).

### 4.6 OD-ENT-02 verifiers
Live: real GitHub state `branches/development/protection` = 404, `rules/branches/development` = `deletion`, `non_fast_forward` only (ruleset 26771061): `verify_ent_audit_branch_protection.py` FAIL as it must (B04). But B01/B02: it also fails a correctly configured protection that names the jobs as GitHub does (**F-11**). Compensating: draft mode PASS on the package (B05), `--published --live-rulesets` FAIL because the sub-manifest does not exist (B06); no equivalence claim anywhere (verified by test and text). Branch protection on `development` is not live (known, F-19).

## 5. Fresh execution at the audit SHA (task 5)

Clean worktree, `uv run --frozen --extra dev-ci` (mirrors CI sync), one process at a time.

| ID | Command | Result |
|---|---|---|
| 5a-1 | `uv run --frozen --extra dev-ci pytest tests/qualification/ent_audit_x/unit tests/qualification/ent_audit_x/integration tests/qualification/ent_audit_x/l3 -q -rsx --junitxml=...` | **exit 0, 293 passed, 91 skipped, 0 failed, 164.32 s** |
| 5a-2 | `scripts/enterprise_audit/validate_post_implementation_master_plan.py` | exit 0 "PASS" |
| 5a-3 | `scripts/enterprise_audit/check_manifest_annex_c_alignment.py` | exit 0 |
| 5a-4 | `scripts/enterprise_audit/mechanical_readiness_check.py` before the wave | FAIL on `.tmp`-dependent lines (evidence_integrity, qual-wave lines, nightly in_progress), OK on the rest (F-16) |
| 5a-4b | same, after the wave (captured, `logs/mech_after_wave.log`) | exit 1: now `ci_nightly_green_at_head`, `pr_nightly_ci_proof`, `full_85_85_executed`, `dossier_complete`, `wbs_*`, `qualification_current_post_tes` are OK; the remaining FAILs (`pre_qual_or_certifying_wave pin_ok=False`, `adv_31`, `ten_13`, `vbn_9`, `vbn_domains_31 stale`, `l1_l2_l3_logs`, `evidence_integrity`) are caused by the script requiring the qual-wave `qualification_sha` to equal the ladder TES `89c76eb7` while my replay is deliberately at the audit SHA `15347255` - expected, not a defect of the replay |
| 5a-5 | `verify_iiac_policy_activation.py --require-in-force` | exit 0, "IN_FORCE @ IIAC-2026-11-05.1 from f8c29fb8..." |
| 5b/5c | `TEST_EXECUTION_SHA=15347255... uv run --frozen python scripts/enterprise_audit/run_qual_wave.py --profile certifying --log-dir .tmp/session/ent-audit-x-certifying-final` (2026-11-08 10:00:26Z to 12:59:52Z) | **exit 0, `qual_wave_report.json` status PASS**, `clean_tree: true`, `pinned_sha == qualification_sha == 15347255ac62dbccfb7b64762b4c0f8af7ec38fa`. Stages (exit 0 each): l1 219 passed (111 at TES: the marker now also selects the post-TES tests), l2 42, semantic_acceptance 63, semantic_matrix 85, adv31 33, ten13 13, vbn9 9, d31 2, l3_cli_full (certifying FULL, `--evidence-bundle`, 2670 s), l3_validate "FULL 85/85 + evidence integrity validated", first_freeze_candidate (FREEZE bootstrap candidate, 85 executed / 0 skipped, only `CANDIDATE_ENTERPRISE_AUDIT_BASELINE.json` produced, no accepted baseline). Replayed artefacts are in `replay/`. |

Comparison of my replayed FULL bundle (`replay/l3-cli-full`, qualification_sha = audit SHA) with the B5 bundle published at TES (jq/diff): the 85-row outcome vector (check_id, executed, skipped, exit_code, findings) is **identical**; `stdout_tail` of all 85 outcomes identical modulo durations; `frz_coverage.v1.json` byte-identical; `vbn_domains.v1.json` identical except `qualification_sha`/`run_id`; report `status PASS`, `exit_code 0`, `findings []`, `checks 85/85/0`, `coverage_label FULL`, `certifying_run true`, 0 outcomes with "skipped"/"no tests ran". Product and sensitive code being unchanged, this is the expected result and it shows that the 2026-11-04 evidence is reproducible in behaviour (not in bytes - see F-01).

Why the qual wave instead of three separate commands: `run_qual_wave.py --profile certifying` is the repository's own certifying procedure; it requires a clean tree and `TEST_EXECUTION_SHA == HEAD`, runs l1, l2, 63 ENT-A semantic acceptance wrappers (`ENT_AUDIT_X_SEMANTIC=1`), the 85-test matrix (`ENT_AUDIT_X_GATE_MATRIX=1`), adv31, ten13, vbn9, d31, then `run_enterprise_audit.py run --mode FULL --output-dir ... --evidence-bundle` (task 5b, certifying FULL 85/85, validated against the closed-world bundle) and a FREEZE bootstrap-candidate run through the runner with `require_clean_repo=True` and `bootstrap_baseline=True` (task 5c; never writes an accepted baseline - the stage asserts that). `ENT_AUDIT_RUN_L3_FULL` is only the pytest wrapper around the same CLI run.

### 5.1 Skips (91 local, 94 on Linux CI), classified (F-20)

| Count | Reason | Hides a mandatory check? |
|---|---|---|
| 85 | `test_semantic_gate_execution.py:28` needs `ENT_AUDIT_X_GATE_MATRIX=1` | No for the audit: executed by the qual wave stage `semantic_matrix`. Yes for CI: no CI job runs them. |
| 2 | `test_l3_freeze_real_85.py:21,47` need `ENT_AUDIT_RUN_L3_FREEZE=1` | The FREEZE 85/85 candidate is covered by the wave stage; the lab "normal FREEZE vs accepted fixture" pytest scenario (`test_l3_freeze_real_85.py:47`) was **NOT EXECUTED** by me (about 45 minutes per FREEZE run on the loaded machine); NOT VERIFIED |
| 1 | `test_l3_freeze_real_85.py:85` "Production FREEZE requires OD-ENT-01" | Yes, by design: FREEZE against a real accepted baseline has never been executed; same root as the six NOT_QUALIFIED rows. Mandatory for formal closure (W4). |
| 1 | `test_l3_full_real_85.py:29` `ENT_AUDIT_RUN_L3_FULL=1` | No: covered by the wave stage `l3_cli_full` |
| 1 | `test_l3_qual_wave.py:64` `ENT_AUDIT_RUN_PREQUAL=1` | No: this is the wave itself |
| 1 | `test_audit_launchers_cross_platform.py:45` "requires bash" | Partly: the SH launcher is non-certifying FAST; GAP-RC-LAUNCHER-SH remains PARTIAL |
| Linux only +2 | `test_b5_publication_verifier.py:79,181` "no strictly valid B5 tree" | **Yes** - B5 integrity is not checked in CI (F-01) |
| Linux only +2 | Windows BAT launcher tests | Environmental |

## 6. Coverage reconciliation (task 6)

Method: `jq` over the manifest and `grep` over the documents (`/tmp/rec`, commands in `test_matrix.txt`).

- 85 manifest check ids (`enterprise_audit_check_manifest.v1.json`: 52 pytest_nodeid, 26 pytest_file, 7 python_script, 0 mutating) = 85 ids parsed from the 85 acceptance wrappers (`assert_semantic_gate_executes(repo_root, "<id>")`) = 85 ids in Annex C = 85 in Annex G2 = 85 in master-plan Matrix E: **five identical sets**. Every entrypoint file exists. The wrappers (21 areas x 3 = 63, TEN 13, VBN 9) call one shared oracle that runs the same `execute_manifest_check` as the FULL run and asserts exit 0; the "negative mutation" column of Matrix E is not implemented as a test (the wrapper proves the gate passes today, not that it can fail) - consistent with F-07.
- Annex G2: all 85 rows still read "(TO_IMPLEMENT)" and name non-existing files (F-14). Annex B test names do not exist (F-14).
- **Matrix C (75 TaskIDs)**: parsed from the master plan: 69 QUALIFIED + 6 NOT_QUALIFIED (C001, C002, C003, C042, C057, C065) = 75. Evidence basis of the 69: 26 cite the B5/qual-wave at TES (durable), 2 cite scripts at HEAD, **41 cite untracked `.tmp` artefacts** (F-05). The six NOT_QUALIFIED rows are correct: C001-C003 (candidate/accepted FSM, no auto-promotion, promote-baseline trust boundary), C042 (unauthorised promote tests), C057 (operator record doc), C065 (extended verify, wired promote). Their unit tests exist and pass (D-2 is implemented in `iiac_baseline_promotion.py`), but the production closure object does not exist: no B6 acceptance pin in the ladder, no accepted-baseline pin, no OD-ENT-01 record, no accepted baseline, no FREEZE against it, and no independent replay of that path. What exactly closes them: (i) this audit's B6 acceptance published and `B6_ACCEPTANCE_PUBLICATION_SHA` set; (ii) a delegation record signed by the operator plus a delegated OD-ENT-01 record under `iiac_evidence/ENT-AUDIT-X/operator_decisions/` with `accepted_baseline_sha`; (iii) candidate baseline from a certifying FREEZE bootstrap at that SHA, closed-bundle verification, promotion through `iiac_baseline_promotion.py` (not through `promote-baseline`, which is human-only and cwd-bound: F-02/F-03); (iv) a certifying FREEZE against the accepted baseline; (v) a later independent audit that replays promotion with forged/blank/other-decision/agent records. Today GAP-RC-OD-TRUST is correctly OPEN.
- **VBN domains 1-31**: 31 rows, 25 QUALIFIED, 6 NA_WITH_EVIDENCE (4, 10, 19, 20, 28, 30), `inventory_error_count` 0, all pointing to one EBH-5 g06 test (F-15).
- **GAP register (Annex F / Matrix D)**: GAP-RC 33 = 22 CLOSED/CLOSED, 10 CLOSED with Closure PARTIAL, 1 OPEN; GAP-B 6 external (BLK-*) honestly external; Annex F carries no status columns; redaction and others closed from unrelated booleans (F-13).
- **FRZ coverage matrix**: 85 rows but 30 map to the catch-all FRZ-AUD-03 and only 12 distinct wildcard targets (F-12). FRZ-AUD-01..10 are all OPEN in the checklist and the overlay; this audit does not issue row verdicts.

## 7. Trust chain re-expressed for IIAC (task 7)

| Level | Original (human Protocol 03) | IIAC meaning | Satisfied by evidence I saw? |
|---|---|---|---|
| L0 | W8 corrective implementation `89c76eb7` | audited_implementation_sha | Yes - commit, ancestor of audit SHA, no product/sensitive delta (git) |
| L1 | TES certifying qual_wave PASS | test_execution_sha | Retained report PASS (read); **independently replayed at the audit SHA**: PASS (section 5), outcome vector identical |
| L2 | EVIDENCE_PUBLICATION_SHA `30085953` B5 | evidence_publication_sha | Partly - digest chain verified on a CRLF Windows checkout, tamper detection demonstrated; **not** portable and not enforced in CI (F-01); content bindings not verified (F-08) |
| L3 | frozen ENT audit head, nightly FULL | CI replay on Linux | Yes for `162acea3` (85/85 FULL, run 42015310879); tip nightly `42468352524` also success (FULL 85/85 at the audit SHA); acceptance wrappers/matrix not run in CI |
| L4 | technical re-audit `0e538269` (recommendation) | superseded by this audit | n/a - read only as evidence |
| L5 | B6_ACCEPTANCE_PUBLICATION_SHA (human signer) | auditor manifest (this audit) + certifier publication gate | **Not satisfied** - verdict CONDITIONAL; ladder pin unset (verified) |
| L6 | ACCEPTED_BASELINE_SHA (OD-ENT-01) | delegated OD-ENT-01 + accepted baseline | Not satisfied - unset; promotion path defects F-02/F-03 |
| L7 | FORMAL_CLOSURE_PUBLICATION_SHA (OD-ENT-08, roadmap, FRZ-AUD) | certifier + operator final publication | Not satisfied - unset; preflight Gate A/B fail closed (verified) |

## 8. Neighbouring surfaces (task 8)

- `.github/workflows/enterprise-audit.yml` (FAST real on push/PR, `ent-audit-pr-gate` path-independent on PR) and `enterprise-audit-nightly.yml` (pytest suite + FULL real; FREEZE/bootstrap only manual): behave as documented in the logs above. Nightly runs `pytest tests/qualification/ent_audit_x` with no semantic/matrix env, so 179 tests skip; FULL real is executed. No `permissions:` block, actions pinned by tag (v4/v5/v6), no change to ENT workflows after TES. Nothing enforces them on `development` (F-19).
- `unit-tests.yml` regression gate: delta adds `governance-fast` and `evol-x-b4-preflight`; the `enterprise` profile is smoke+governance, ENT audit is not part of the regression gate. `governance-policy-pin` advisory job failed on infrastructure at the tip.
- Ladder parser `baseline.parse_sha_ladder_table` and `preflight.ladder_pin`: agree on the real ladder and on 13 variants, fail closed on malformed cells, last-row-wins (F-18).
- `verify_ent_audit_branch_protection.py`: F-11. `verify_od_ent_02_compensating_control.py`: marker + file-existence + sub-manifest checks are real (not only strings); wiring defect F-10.

## 9. Known limits recorded (task 9)

1. Independence is `process_enforced`: the auditor is a sub-agent launched by the orchestrating session; no cryptographic or telemetry proof; the auditor session id is unknown to the auditor and is not invented.
2. The operator's handoff attestation line is not signed (as instructed).
3. Branch protection on `development` is not live; only ruleset 26771061 (deletion + non-fast-forward). OPEN-BP-01 stands. This audit does not rule on OD-ENT-02 sufficiency or on any FRZ-AUD row.
4. Mode B is a replay on one platform (Windows) plus retained Linux CI logs; Linux outcomes of the 85 checks are not retained, so skip-hiding on Linux is NOT VERIFIED.
5. Not done / NOT VERIFIED: symlink substitution (A23); a behavioural negative test for each of the 31 ADV threats (F-04); execution of a certifying FULL on a dirty tree (F-17 is code-read); FREEZE against a real accepted baseline (does not exist); the lab "normal FREEZE vs accepted fixture" L3 pytest test (`ENT_AUDIT_RUN_L3_FREEZE=1`, not run); Linux outcomes of the 85 checks other than the pass/fail summary line in CI logs; whether any FRZ-AUD row, OD-ENT-02 route or accepted-baseline promotion would be sufficient (out of scope).
6. Harness defects I made and discarded (transparency): first `pytest` start used `-p no:cacheprovider` (exit 4, `logs/a0_failed_start_cacheprovider.log`); Mode A probe v1 wrote the alternates file with CRLF and v2 used `git show` with a path over MAX_PATH (both invalid, kept as `logs/modeA_probe.v1_*.log`, `v2_*.log`); one trivial `json` one-liner and one empty python heredoc ran while a pytest process was active (not a pytest/uv run). Results quoted come from the final runs only.

## 10. Findings (see `findings.json`)

BLOCKER 0 | HIGH 2 (F-01, F-02) | MEDIUM 11 (F-03 ... F-13) | LOW 5 (F-14 ... F-18) | INFO 2 (F-19, F-20).

| ID | Sev | Title (short) |
|---|---|---|
| F-01 | HIGH | B5 digests line-ending dependent; strict B5 tests skipped in Linux CI |
| F-02 | HIGH | promote-baseline trust gate bound to cwd |
| F-03 | MEDIUM | human path accepts unsigned boolean record; delegated signature is textual |
| F-04 | MEDIUM | ADV-01..31 stage mostly vacuous; Matrix F names missing |
| F-05 | MEDIUM | 41 QUALIFIED Matrix C rows cite untracked evidence |
| F-06 | MEDIUM | sandbox is not isolation; redaction misses 10/15 shapes; W8 cites wrong test |
| F-07 | MEDIUM | exit code 0 = PASS even if all tests skipped |
| F-08 | MEDIUM | B5 verifier structural only; re-sealed forgeries accepted |
| F-09 | MEDIUM | preflight accepts fabricated manifest/pass record |
| F-10 | MEDIUM | compensating route unsatisfiable by default wiring |
| F-11 | MEDIUM | live verifier expects strings GitHub does not emit |
| F-12 | MEDIUM | FRZ coverage is a prefix heuristic (30 checks -> FRZ-AUD-03) |
| F-13 | MEDIUM | GAP register contradictions and unrelated closure rules |
| F-14 | LOW | Annex B/G2 drift |
| F-15 | LOW | VBN domain accounting thin; D30 circular deferral |
| F-16 | LOW | mechanical readiness not reproducible from a clean clone |
| F-17 | LOW | certifying FULL does not require a clean tree |
| F-18 | LOW | ladder last-wins without duplicate detection |
| F-19 | INFO | no live enforcement; red ENT runs on development |
| F-20 | INFO | skips classified |

## 11. Conclusion

Verdict **CONDITIONAL** under IIAC-B6. Reasoning:

- Technical reproducibility of ENT-AUDIT-X is demonstrated by execution (qual wave PASS at the audit SHA; identical 85-row outcome vector; Linux CI FULL 85/85 at the same SHA; 293/0 in the suite; all fail-closed probes on manifest, runner, mode, path-jail, baseline-jail, report-mutation, preflight pins and the delegated-record verifier behaved fail-closed). Product and qualification-sensitive code are bit-identical to the audited implementation.
- A PASS is not justified because (a) the published B5 evidence cannot be verified on the bytes git stores and the only automatic check of it is silently skipped in CI (F-01); (b) the operator trust boundary that C001-C003/C042/C057/C065 and GAP-RC-OD-TRUST depend on has a demonstrated bypass in the still-public human CLI path (F-02, F-03); (c) the "ADV-01..31", "sandbox isolation", GAP-register and Matrix C qualification statements are stronger than the tests and artefacts behind them (F-04..F-07, F-13); (d) the two OD-ENT-02 routes cannot currently be satisfied by their own verifiers (F-10, F-11).
- None of the findings is a BLOCKER (no product defect, no false PASS of the 85 gates observed, no secret in published evidence), which is why the verdict is not FAIL. Closing F-01 and F-02 (and restating F-04..F-07, F-13 honestly or accepting them as scoped limitations in the certifier manifest) should allow a re-audit of the delta only. Because the fixes to F-02 touch a qualification-sensitive path (`run_enterprise_audit.py`), the ladder rule requires a new TES for that fix; the audit of that new TES can reuse this report for the unchanged remainder.
- This audit does not rule on OD-ENT-02 sufficiency, FRZ-AUD-01..10, OD-ENT-00/01/08 or closure publication.

Counts in section 0 and in the manifest are those of `findings.json` (BLOCKER 0, HIGH 2, MEDIUM 11, LOW 5, INFO 2).

