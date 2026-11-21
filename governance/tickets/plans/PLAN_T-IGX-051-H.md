# PLAN T-IGX-051-H — RW train, lane 2: dependency batch and the dependency-ownership gate (A4-005)

Draft, 2026-11-17. Nothing was changed outside `.tmp`; the only runs were throwaway runs of the gate script and of `uv lock` (check, dry run, relock, one package upgrade) on copies under the session scratchpad (section 12); no pytest process was started. No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this plan. Evidence notation `path@D:line` and `path@T:line`; `D` = `2727a2c2364a97492202b931375772a400279a9b` (`origin/development`, fetched 2026-11-17), `T` = `05699b633cdb1318c215f6529e449420efc1382c` (`origin/arch-freeze/rw-train`, merge of PT-WA), `B2` = `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (AUDIT_BASELINE_2). The five repository files this plan reads or edits (`pyproject.toml`, `uv.lock`, `scripts/maintenance/check_dependency_ownership.py`, `tests/unit/test_dependency_governance.py`, `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py`) have the same git blob at `B2`, `D` and `T`. `FACT_CHECK.md` has the claim-by-claim table.

## 1. Purpose and findings

Train node `T-AF2-DEPS-BATCH` (RW-PLAN v13 appendix A1 block `T-AF2-DEPS-BATCH (T-IGX-051-H)`; lane 2, after PT-WA and PT-WB). Sensitivity NONPRODUCT: none of the four write-set paths starts with `intergrax/`, `agents/` or `applications/` (`scripts/governance/iiac_common.py@D:22`).

| Item | Statement (labels quoted from RW-PLAN v13 and the Campaign A dossier) | What this ticket does | What it cannot close alone |
|---|---|---|---|
| EQOA-A4-005 (MEDIUM, CONFIRMED_DEFECT per `campaign_a/findings/FINDINGS.jsonl@D:26`) | "Dependency ownership gate FAILS at baseline: CORE_OWNERSHIP_VIOLATION for jsonschema and tzdata" | Part A: both names are declared in the reviewed `CORE_ALLOWLIST`; new negative test `NT-RW-DEP-01` | closure needs the exact-SHA wave audit (`PT-Q02`; audit flag "yes" in RW-PLAN v13 `audit_flags`) |
| FRZ-DEBT-02, FRZ-DEBT-03 (checklist rows "unresolved transitional authority = 0", "temporary compatibility seam in frozen scope = 0", status OPEN) | RW-PLAN v13 dispositions: owner `T-AF2-DEPS-BATCH`, "dependency part; EDER part stays with PT-R01..R03" | the dependency part only: the same Part A change and the same test row | the EDER part (`PT-R01..R03`); this ticket does not touch any register |
| D-06 (RW-PLAN dedupe) | one defect: dependency-ownership gate red (jsonschema, tzdata), carried by EQOA-A4-005, FRZ-DEBT-02, FRZ-DEBT-03 and PDC handoff 5.9 | one fix, one test, three rows | — |
| Q-RW-09 / PDC handoff 5.9 | "dependency-bot / `uv.lock` batch" held until the train cut; batch = the dependency-bot PRs open when `arch-freeze/rw-train` is cut plus the A4-005 declarations | Part B: a rule for the held batch, and the facts about the four PRs named in the brief (section 4) | whether any PR is carried at all: **none is open at drafting** (section 4, question Q-DEP-01) |

Dependencies on other nodes: PT-WA and PT-WB merged into the train first (lane 2 rule). Overlap: none (section 9).

## 2. Design rules applied to every part

1. **No change of gate logic.** The gate script gets two data lines in `CORE_ALLOWLIST`; no function, no policy table other than that map, no `FORBIDDEN_CORE_NAMES`, `HIGH_RISK_POLICY` or `INTENTIONAL_CORE_EXTRA_SHARING` change. The script's own comment says a core addition "requires changing this map and the qualification evidence in the same PR" (`scripts/maintenance/check_dependency_ownership.py@D:18-19`): the ticket, this plan and `NT-RW-DEP-01` are that evidence.
2. **A declaration follows a fact.** Both names are already core requirements at `D` and `T` (`pyproject.toml@D:76,81`) and the code uses them (section 3); the allowlist is made to match the declared and used set, not the reverse.
3. **A range change and its lock change land in one commit.** CI's `LOCK_CONSISTENCY` check is `uv lock --check --offline` (`scripts/maintenance/check_environment_conformance.py@D:321-323`); the three pyproject-only bot PRs failed it (section 4).
4. **The new test runs in the gate selection.** It carries `pytest.mark.unit` and `pytest.mark.gate`, so the CI selector `tests/unit -m "gate and not no_ci"` (`.github/workflows/unit-tests.yml@D:286`) picks it up; the existing `tests/unit/test_dependency_governance.py` carries only `unit` (`@D:13`).
5. **In-process, not by exit code.** The new test imports `check_project` (`from scripts.maintenance.check_dependency_ownership import ...`, as the existing test does, `tests/unit/test_dependency_governance.py@D:8-11`; the import root comes from `pythonpath = [".", ...]`, `pyproject.toml@D:401`) and never runs the script as a subprocess.
6. **No bot PR is merged, commented on, reopened or closed by this ticket.** Dependabot PRs target `development`, which is on hold until FCS; the train re-creates a change by hand, only if the batch is non-empty (Part B).

## 3. Part A — A4-005: ownership declarations (firm)

**Fact.** `check_project()` of the unmodified script on `pyproject.toml` returns `CORE_OWNERSHIP_VIOLATION: jsonschema` and `CORE_OWNERSHIP_VIOLATION: tzdata` and nothing else. `[project].dependencies` has 17 entries; `CORE_ALLOWLIST` has 15 and lacks those two (`scripts/maintenance/check_dependency_ownership.py@D:20-36`, emission at `@D:180-183`). Probe at `D` (script and `pyproject.toml` extracted to scratch, run with the repository's interpreter): exit 1 and the two lines above (section 12).

**Why the names are core.**
- `jsonschema>=4.23,<5` (`pyproject.toml@D:76`): imported unconditionally at module level by `intergrax/platform/enterprise_audit/validators.py@D:12` (`from jsonschema import Draft202012Validator`); added to core by commit `332b2bf02` ("ENT-AUDIT-X C069: fail-closed JSON Schema for all evidence artifacts").
- `tzdata; sys_platform == 'win32'` (`pyproject.toml@D:81`, with its comment at `@D:80`): `zoneinfo` is used by `intergrax/runtime/long_running/recurrence.py@D:54,215-218`; added by commit `cc1c0efb6` ("fix(pdc): B-P6-R integration repairs (T-IGX-036-T)").

**Change (default option a).** Two entries in `CORE_ALLOWLIST`, both `"CORE_FOUNDATION"`, placed after `"chardet"` (`@D:35`):
```
    "jsonschema": "CORE_FOUNDATION",
    "tzdata": "CORE_FOUNDATION",
```
`DEPENDENCY_OWNERSHIP.md` requires every core entry to have a `CORE_FOUNDATION` or `CORE_SERVER` owner (`docs/project/architecture/DEPENDENCY_OWNERSHIP.md@D:55-56`); both fit `CORE_FOUNDATION` (a schema validator and a platform time-zone database, neither a server component). Scratch simulation: with exactly these two lines the extracted script prints `dependency governance: OK` on the current `pyproject.toml` and on the variant with the three bot range changes of section 4 (section 12).

**Rejected by default (option b): move both out of core.** The assertion would become "jsonschema and tzdata are not core requirements" (RW-PLAN row text). It needs a new optional extra or a dependency-group move, a `uv.lock` change, and a decision that `intergrax/platform/enterprise_audit/validators.py` and the Windows recurrence path may fail at import or first use on a default install. That is a change of the install contract, not a declaration; it is raised as question Q-DEP-04, not drafted.

**Not changed:** the docs. `DEPENDENCY_OWNERSHIP.md` and the DEP4 document do not name `jsonschema` or `tzdata` and do not enumerate the allowlist (grep at `D`: no match), and no docs path is in appendix A1 (question Q-DEP-05).

## 4. Part B — the held dependency-bot batch (Q-RW-09)

### 4.1 State at drafting (2026-11-17, read-only `gh` calls)

`gh pr list --state open` on the repository returns no pull request. The four PRs named in the brief and RW-PLAN v13 were all **closed without merge** on 2026-11-17 between 10:03:49Z and 10:03:57Z by the repository owner account (`gh api .../issues/<n>/events`: event `closed`, actor `jakbuczarnecki`; Dependabot then commented that it will not notify about "this release" again and deleted the four head branches). So at drafting the batch of "dependency-bot PRs open at the train cut" is **empty**. RW-PLAN v13 names only #631 (note on ticket row T05, line 178); #637, #638 and #639 were opened on 2026-11-16 at 21:25Z and are not in that text.

| PR | Dependency | Declared where | Old range -> new range (the bot's change) | Effect on `uv.lock` | Bot CI at the time | Current state |
|---|---|---|---|---|---|---|
| #631 | aiohttp (title: 3.13.3 -> 3.14.4) | not core; only extra `integrations-slack`, `aiohttp>=3.9` (`pyproject.toml@D:372-374`, unchanged by the PR); also required by `langchain-community` in the lock | no range change | the PR edited only `uv.lock` (37 added, 33 deleted lines): `aiohttp` locked `3.13.3` -> `3.14.4` (`uv.lock@D:99-100`), `yarl` `1.22.0` -> `1.25.1` (`uv.lock@D:6094-6095`), and `aiohttp` gains a `typing-extensions` dependency. A scratch `uv lock --upgrade-package "aiohttp==3.14.4"` produced the same two package updates | `ci-smoke` SUCCESS | CLOSED, unmerged |
| #637 | sentence-transformers | extras `rag-local-embeddings` (`pyproject.toml@D:236`) and `dev-unit-cert` (`@D:344`) | `>=3.0,<6` -> `>=3.0,<7` (two lines) | the PR edited only `pyproject.toml`, so the lock was stale: `ci-smoke` failed at `LOCK_CONSISTENCY`. A relock changes two metadata lines (`uv.lock@D:2199-2200`) and no locked version (`5.2.0`, `uv.lock@D:5060-5061`, stays) | `ci-smoke` FAILURE | CLOSED, unmerged |
| #638 | cryptography | **core**, `pyproject.toml@D:58`; allowlisted `CORE_FOUNDATION` (`@D:27`); `HIGH_RISK_POLICY` `BOUNDED_MAJOR` (`@D:95`) | `>=42.0,<47` -> `>=42.0,<51` | `pyproject.toml` only; same `LOCK_CONSISTENCY` failure. A relock changes one metadata line (`uv.lock@D:2103`) and no locked version (`46.0.5`, `uv.lock@D:766-767`, stays). The range stays bounded, so the policy check adds no finding | `ci-smoke` FAILURE | CLOSED, unmerged |
| #639 | PyMuPDF (`pymupdf`) | extra `parsing-pdf` (`pyproject.toml@D:203`); `FORBIDDEN_CORE_NAMES` member, so it must stay out of core (`@D:62`) | `>=1.23,<2` -> `>=1.23,<3` | `pyproject.toml` only; same failure. A relock changes one metadata line (`uv.lock@D:2176`) and no locked version (`1.26.7`, `uv.lock@D:4408-4409`, stays) | `ci-smoke` FAILURE | CLOSED, unmerged |

Combined scratch run (section 12): with the three range changes applied, `uv lock` rewrote exactly four metadata lines (cryptography 1, pymupdf 1, sentence-transformers 2) and no package version; the gate script reported only the two A4-005 findings, i.e. no new one. The failing `ci-smoke` step on #637, #638, #639 is "Environment conformance (CI profile)", check `LOCK_CONSISTENCY`; the log line is truncated by the check itself, so the exact `uv` message is not quoted here.

Labels in the PR bodies (the bot's text, not verified here): #638's changelog excerpt names a security fix in cryptography 50.0.0; #639's text describes PyMuPDF 2.0 and 2.1 releases; #637's text describes sentence-transformers 6.3.0. Whether the code works with those majors is UNVERIFIED and is not part of this plan.

### 4.2 Rule for the batch at the train cut (draft default)

1. Step 0 (read-only, before anything is edited): list open dependency-bot PRs (`gh pr list --state open --author "app/dependabot"`); record the list in the PR body. Do not merge, comment on, reopen, rebase or close any PR.
2. Empty list (the state at drafting): Part B adds nothing; the ticket is Part A alone. `pyproject.toml` and `uv.lock` are firm-listed paths that this ticket **may** edit; a zero diff on them is a valid outcome.
3. Non-empty list: each PR is carried as a separate commit group after Part A. The train re-creates the PR's change by hand: for a range change, edit the same lines in `pyproject.toml` and run `uv lock` (no `--upgrade`) in the same commit; for a lock-only PR (as #631), `uv lock --upgrade-package "<name>==<version>"`. Then `uv lock --check --offline`, the gate script and the regression scope of section 7. A carried item that moves a locked version needs its own targeted regression scope, which this plan does not define (question Q-DEP-02).
4. The four closed PRs are **not** carried by default (question Q-DEP-01); section 4.1 is precomputed input for that ruling, not scope.

## 5. Write set

Branch rule: product-tree rules do not apply (no product path); commits of this ticket go only to branch `arch-freeze/rw-train` (through a PR into the train). Every path below equals RW-PLAN v13 appendix A1 block `T-AF2-DEPS-BATCH (T-IGX-051-H)` (`RW_PLAN.md:1139-1146`).

**Firm (4):**

| Path | Kind | Part | Verified |
|---|---|---|---|
| `pyproject.toml` | edited, only if Part B is non-empty | B | exists at `D` and `T` |
| `uv.lock` | edited, only if Part B is non-empty | B | exists at `D` and `T` |
| `scripts/maintenance/check_dependency_ownership.py` | edited (two allowlist lines) | A | exists at `D` and `T` |
| `tests/unit/architecture/test_rw_deps_core_ownership_declared.py` | new | A | absent at `D` and `T`; parent `tests/unit/architecture/` exists at `T` |

**Conditional (0).** **Product paths: 0.** **Docs: none.**

**Forbidden (a need for any of these stops the work and becomes an amendment):** every other path, in particular `tests/unit/test_dependency_governance.py` and `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py` (regression only), `.github/` (including `dependabot.yml` and the workflows), `docs/`, `intergrax/`, `agents/`, `applications/`, every other gate script, the agent and application `pyproject.toml` files, the known-red register, IIAC and PDC-X records, the roadmap and the FRZ checklist.

## 6. Negative tests

One new file, one row id `NT-RW-DEP-01` serving three RW-PLAN rows: `NT-ROW-003` (A4-005), `NT-ROW-059` (FRZ-DEBT-02, `NT-DEBT-02-1`), `NT-ROW-060` (FRZ-DEBT-03, `NT-DEBT-03-1`). Type B (red-first at the pre-fix SHA) with one control. Assertion text of the rows: for every requirement in `pyproject.toml` `[project].dependencies`, `check_project()` imported in process reports no `CORE_OWNERSHIP_VIOLATION`; a seeded undeclared core requirement in a temporary copy of `pyproject.toml` is reported (sensitivity control).

| ID | Configuration | Expected result | Node id | Type |
|---|---|---|---|---|
| NT-RW-DEP-01a | for each of the 17 requirements of `[project].dependencies` (read with `tomllib` and `packaging.requirements.Requirement`; test id = normalized name), `check_project(DEFAULT_PROJECT)` | `f"CORE_OWNERSHIP_VIOLATION: {name}"` is not in the findings. At the pre-fix SHA the cases `jsonschema` and `tzdata` fail, 15 pass | `tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01a_core_requirement_has_declared_owner` | B (2 red, 15 green controls) |
| NT-RW-DEP-01b | the same findings as one aggregate | no finding starts with `CORE_OWNERSHIP_VIOLATION:` (red at the pre-fix SHA, naming both packages) | `...::test_rw_dep_01b_project_reports_no_core_ownership_violation` | B |
| NT-RW-DEP-01c | temporary copy of the repository `pyproject.toml` with one extra undeclared requirement (for example `"rw-dep-seed-undeclared>=1,<2"`) inserted into `[project].dependencies` | the findings contain `CORE_OWNERSHIP_VIOLATION: rw-dep-seed-undeclared` (sensitivity control; green before and after, so it proves the check is live, not that the fix works) | `...::test_rw_dep_01c_gate_reports_seeded_undeclared_core_requirement` | C |

Rules: the test never asserts `check_project(...) == []` (other finding kinds are other gates' business, and the existing `test_current_project_passes_dependency_governance` already does that); the script's exit code does not count; no `Any` or `object` is needed. **Red-first (S0):** the new file runs on a detached worktree at the pre-change commit (train tip before this ticket's first edit); expected 3 failures (01a `jsonschema`, 01a `tzdata`, 01b), 16 passes (15 of 01a, 01c); the log is kept for the auditor. If option b of Q-DEP-04 is ruled, 01a/01b change to "not core" with the same red-first run (RW-PLAN row text).

Counts: 3 test functions, 19 test items; 3 RW-PLAN rows (`NT-ROW-003`, `NT-ROW-059`, `NT-ROW-060`).

## 7. Regression scope and test commands

One process at a time under the lock `D:/Projekty/_wt/_locks/pytest.lock` (`mkdir` must succeed, `rmdir` right after, also on failure; only the creator removes it), logs under `.tmp/session/arch-freeze/w3-impl/deps-batch/`, `uv run --frozen` so that no run can rewrite `uv.lock`, no broad or diagnostic suite, no Docker, no vendor integration.

| Step | Command | Purpose / expectation |
|---|---|---|
| S0 red-first | `uv run --frozen pytest tests/unit/architecture/test_rw_deps_core_ownership_declared.py -p no:cacheprovider` at the pre-change commit (detached worktree) | 3 expected failures as in section 6 |
| S1 new test | the same command after Part A | all 19 items expected green |
| S2 gate regression | one `uv run --frozen pytest tests/unit/test_dependency_governance.py tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py -p no:cacheprovider` | `test_current_project_passes_dependency_governance` (`@D:18-19`) is the existing mirror of NT-RW-DEP-01 and is red at `D`/`T` by the scratch probe, expected green after Part A; the other 5 items of that file and the 9 test functions of the EBH-3 file are regression, expected unchanged. No pass is claimed by this plan |
| S3 L1 script | `uv run --frozen python scripts/maintenance/check_dependency_ownership.py` | expected exit 0 after Part A; the exit code goes into the PR body (RW-PLAN planned expected-red row "L1 local scripts": not run by `unit-tests.yml`) |
| S4 lock | `uv lock --check --offline` (the command of `check_environment_conformance.py@D:323`) and, in CI, `uv sync --extra dev-ci --frozen` | expected exit 0; with a zero `pyproject.toml`/`uv.lock` diff it is unchanged from the base |
| S5 static | `uv run --frozen ruff check` and `uv run --frozen pyright` on the script and the new test | typing and style |
| CI | the full L2 gate suite in CI because the lockfile may change (RW-PLAN regression column) | read from the PR, not run locally |

**Contract-surface ratchet (OD-AF-11).** This ticket edits nothing under `intergrax/contracts/`; if an amendment ever does, the wave-packet contract check (`rw-plan/work/typ05/measure_typ05.py` plus `typ_measure.py`) must report nothing new before the PR merges.

**Naming note for S2.** `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py` is named in the RW-PLAN rows and the planned expected-red table as a dependency-ownership gate node. Its content is the EBH-3 dependency-direction and integrations-composition gates (`@D:3`); it never reads `pyproject.toml` and never imports the ownership script (`git grep` at `D`). It stays in the regression scope as written, but it is not evidence for A4-005 (question Q-DEP-06).

## 8. Order of work inside the ticket

0. Read-only: open bot PRs list (section 4.2), `uv --version` recorded, `git rev-parse HEAD` of the train tip.
1. Write the new test file; S0 on the pre-change commit; keep the log.
2. Part A commit group: the two allowlist lines; S1, S2, S3, S5.
3. Part B commit groups (only if step 0 found open bot PRs or the Operator rules some of the four in): one commit per dependency, `pyproject.toml` and `uv.lock` together; S4 after each.
4. PR into `arch-freeze/rw-train` with the L1 exit code, the S0 log reference and the bot-PR list in the body; no tool signature in commits or PR text; every commit carries `Ref: governance/tickets/T-IGX-051-H.md` (the commits touch `scripts/`).

## 9. Overlaps and merge order

Lane 2, parallel with PT-FC2, PT-FC4, PT-E05, PT-D08, after PT-WA and PT-WB (RW-PLAN v13 lanes table). None of the four paths is listed in any other appendix A1 block: a search of `RW_PLAN.md` for `pyproject.toml` and `uv.lock` hits only this ticket's own rows (178, 363, 419, 420 and the A1 lines 1143-1144), and `check_dependency_ownership.py` appears elsewhere only as a regression or L1 item (lines 343-346, 599, 644), not as a write path. Risk R-6 covers a later ticket that needs a new dependency. Merge: independent review, then merge commit into `arch-freeze/rw-train` (OD-AF-02); the merge of the train into `development` stays the Operator's decision.

## 10. ACT-N11 / IIAC consequence

None of the four paths is a product path (`PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")`, `scripts/governance/iiac_common.py@D:22`), so by reading this ticket does not break the IIAC activation. The script is listed in `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json@D:89366-89369` as category 2, `owner_hint` "inventory/neutral". Not run: no IIAC validator was executed.

## 11. ARCH-FREEZE impact and what closes each finding

| Finding | Closure condition (nothing is closed by this plan) |
|---|---|
| A4-005 | `NT-RW-DEP-01` green at the exact wave SHA, the L1 script exit 0 recorded, the existing `test_current_project_passes_dependency_governance` green, and the independent audit (`PT-Q02`); the ticket file carries the audit line (RWA-05) |
| FRZ-DEBT-02, FRZ-DEBT-03 | dependency part: the same evidence; the EDER part stays with `PT-R01..R03`; the checklist rows are changed only by the orchestrator through the promotion path |
| Q-RW-09 / PDC handoff 5.9 | the PR body lists the open bot PRs found at step 0 and what was carried; an empty list is a recorded result |

Independent audit design: a separate session, not the implementer and not a reviewer of this plan, at the exact wave head (`PT-Q02`). Mutation checks, in a detached worktree only: remove `tzdata` from `CORE_ALLOWLIST` (01a `tzdata` and 01b must fail); add an undeclared requirement to `[project].dependencies` (01a/01b/01c behaviour); widen a high-risk range to unbounded (the existing policy test must still fail). The auditor also re-runs the scratch probes of section 12.

## 12. Evidence of this draft (session artefacts, not repository records)

Scratch under `C:/Users/XPS/AppData/Local/Temp/claude/a6d338a1be78d1c4/d7ad588d-b317-4857-b7e3-52fa771ad644/scratchpad/` (`probe/`, `fixprobe/`, `lockprobe/`):
- Gate probe at `D`: `check_dependency_ownership.py` and `pyproject.toml` extracted with `git show`, run with the repository interpreter in isolated mode: `dependency governance: FAIL`, `CORE_OWNERSHIP_VIOLATION: jsonschema`, `CORE_OWNERSHIP_VIOLATION: tzdata`, exit 1. Same two lines with the three bot range changes applied.
- Fix simulation: the same script plus the two allowlist lines: `dependency governance: OK`, 17 core direct dependencies, on both `pyproject.toml` variants.
- Lock probes (`uv 0.13.4`, a copy of the workspace `pyproject.toml` files and `uv.lock` from `D`): `uv lock --check` exit 0 on the copy; after the three range edits `uv lock` rewrote four lines (section 4.1) and changed no version; `uv lock --upgrade-package "aiohttp==3.14.4"` updated `aiohttp` and `yarl`. These used the network (PyPI) and ran in the scratchpad only; nothing in the repository was touched.
- No pytest process was started.

## 13. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | The bot batch is empty, or the Operator expects the closed PRs to be applied; Dependabot will not re-open them for the same releases | question Q-DEP-01; Part B is a rule, not a fixed content; the PR body records the empty result |
| R-2 | A range edit without a lock edit turns `ci-smoke` red (seen on #637-#639) | design rule 3; S4 after each carried item |
| R-3 | `uv lock` at implementation time resolves differently from the scratch run (index changes, another `uv` version) | no `--upgrade` except for a named package; record `uv --version`; review the lock diff line by line; any version change not named in the ticket stops the work |
| R-4 | Wider ranges (cryptography `<51`, PyMuPDF `<3`, sentence-transformers `<7`) let a fresh resolve without the lock pick majors the code was never qualified on | question Q-DEP-02; the policy check keeps cryptography bounded; compatibility is UNVERIFIED |
| R-5 | The allowlist is a review surface; two more names widen it | each name has a usage fact (section 3); `NT-RW-DEP-01c` keeps the gate live for any other name |
| R-6 | A later train ticket needs a dependency and edits `pyproject.toml` or `uv.lock` without being in appendix A1 | the plan of that ticket must list the paths and be serialized with this one; none does at v13 |
| R-7 | The known-red register PR names the wrong nodes for this gate (section 7 naming note) | question Q-DEP-06 |
| R-8 | `uv run` without `--frozen` rewrites `uv.lock` as a side effect | all commands use `--frozen` |

## 14. Rollback

Part A: revert the allowlist commit and delete the new test file; no data, setting or persisted-format change. Part B: each carried dependency is its own commit group and reverts alone (`pyproject.toml` and `uv.lock` together). A revert of Part A returns the gate to the red state recorded in section 3.

## 15. Open questions with draft defaults

| ID | Question | Draft default | Who decides |
|---|---|---|---|
| Q-DEP-01 | The four PRs named for the batch (#631, #637, #638, #639) were closed unmerged by the repository owner account today; Q-RW-09's premise ("open at the train cut") currently yields an empty batch. Carry any of them anyway? | carry none; batch = A4-005 plus PRs open at the cut; section 4.1 is the precomputed effect if the Operator rules some in | Operator, or the orchestrator if the closure was already a ruling |
| Q-DEP-02 | If an item is carried: range-only (as #637-#639, locked versions unchanged) or a version upgrade (as #631, and #638 if the security fix of cryptography 50 is wanted)? An upgrade needs its own regression scope | range-only items need only S2-S4; a version-moving item is an amendment with a named test scope | Operator |
| Q-DEP-03 | Owner class for the two allowlist entries and whether a two-line data change to a gate map counts as a "gate meaning" change for the wave packet | `CORE_FOUNDATION` for both; not a gate-meaning change (no logic changes, the gate gets stricter for every other name) | orchestrator |
| Q-DEP-04 | Option b (move both out of core) instead of declaring them | option a; option b is the install-contract change of section 3 | Operator |
| Q-DEP-05 | `DEPENDENCY_OWNERSHIP.md` says the checker runs in the governance smoke job of `unit-tests.yml` (`@D:72-73`); no workflow at `D` references it (`git grep` over `.github/` finds nothing), which agrees with RW-PLAN F-55. The finding's scope lists `DEPENDENCY_OWNERSHIP.md` but appendix A1 lists no doc. Edit docs or wire the script into CI here? | neither: both are outside appendix A1; a docs/CI follow-up is for the orchestrator | orchestrator |
| Q-DEP-06 | RW-PLAN names `test_ebh_3_dependency_ownership_gate.py` and `test_dependency_governance.py` as dependency-ownership nodes of job `gate-tests`. The first does not test the pyproject gate; the second carries only the `unit` marker and so is not selected by `-m "gate and not no_ci"`. Correct the known-red register entry? | keep both in the regression scope as written; the register PR should name `tests/unit/test_dependency_governance.py::test_current_project_passes_dependency_governance` and be re-measured; this ticket's new test (marked `gate`) is the node that the gate-tests job will see | orchestrator |

## 16. Authorization requests

| ID | Request | Scope |
|---|---|---|
| AR-DEP-1 | Edit the four write-set paths and open a PR into `arch-freeze/rw-train` | section 5 only |
| AR-DEP-2 | One-process `uv run --frozen pytest`, `ruff` and `pyright` runs under the shared lock, plus one detached worktree under `<parent>/_wt/<repo>/` for S0 | sections 7 and 8 |
| AR-DEP-3 | `uv lock`, `uv lock --check --offline` and network access to the package indexes (PyPI, `pytorch-cpu`) | only if Part B carries an item; no vendor integration |
| AR-DEP-4 | Read-only `gh pr list` and `gh pr view` | step 0; no merge, comment, close, reopen or rebase of any PR |

Not requested: any product path, `.github/`, docs, Docker, vendor or production configuration, the IIAC re-pin, a register edit.
