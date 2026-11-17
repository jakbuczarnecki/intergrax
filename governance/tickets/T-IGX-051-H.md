# T-IGX-051-H — RW train, lane 2: dependency batch and the dependency-ownership gate (A4-005)

## Scope

Non-product ticket of the RW product train (RW-PLAN v13 node `T-AF2-DEPS-BATCH`, appendix A1 block `T-AF2-DEPS-BATCH (T-IGX-051-H)`; lane 2, after PT-WA and PT-WB; approval route OD-AF-22: one wave packet approved by the Operator, then signed by the delegate). Drafted from EQOA-A4-005 ("Dependency ownership gate FAILS at baseline: CORE_OWNERSHIP_VIOLATION for jsonschema and tzdata", Campaign A baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`), the dependency part of FRZ-DEBT-02 and FRZ-DEBT-03 (the EDER part stays with `PT-R01..R03`), and the held dependency-bot batch of PDC handoff 5.9 (Q-RW-09). Facts re-read at `origin/development` `2727a2c2364a97492202b931375772a400279a9b` and at the train tip `05699b633cdb1318c215f6529e449420efc1382c` (merge of PT-WA); the five repository files involved are identical at both and at AUDIT_BASELINE_2 `7960ae7f6c82da60ca9acdc0ceebf63810002a42`.

- Part A (firm): the two names that are declared core requirements and used by the code, `jsonschema` and `tzdata`, are added to `CORE_ALLOWLIST` in `scripts/maintenance/check_dependency_ownership.py` (owner `CORE_FOUNDATION`, data lines only, no gate logic change); new test `tests/unit/architecture/test_rw_deps_core_ownership_declared.py` with `NT-RW-DEP-01` (RW-PLAN rows `NT-ROW-003`, `NT-ROW-059`, `NT-ROW-060`), red-first at the pre-fix SHA on `jsonschema` and `tzdata`, plus a sensitivity control.
- Part B (conditional content of the firm paths `pyproject.toml` and `uv.lock`): the dependency-bot PRs that are open when `arch-freeze/rw-train` is cut are re-listed at the cut and carried one commit group each, by re-creating the change on the train (a bot PR itself is never merged, commented on, reopened or closed by this ticket). At drafting no such PR is open: #631 (aiohttp), #637 (sentence-transformers), #638 (cryptography) and #639 (PyMuPDF) were closed unmerged on 2026-11-17. The default is to carry none of them; the plan lists each with its dependency, old and new range and the effect on `uv.lock`, as input for the question Q-DEP-01.

Write set (exact list in the plan, section 5; it equals RW-PLAN appendix A1 block `T-AF2-DEPS-BATCH`, no difference, see `.tmp/session/arch-freeze/w3-drafts/deps-batch/INPUT_T-IGX-051-H.md`): 4 firm paths (`pyproject.toml`, `uv.lock`, `scripts/maintenance/check_dependency_ownership.py`, one new test file), 0 conditional, 0 product paths, no docs. No edit of `intergrax/`, `agents/`, `applications/`, `.github/`, any other gate script or the two existing dependency test files. Commits go only to branch `arch-freeze/rw-train`; every commit carries the trailer `Ref: governance/tickets/T-IGX-051-H.md`.

Dependencies and overlaps stated in the plan: PT-WA and PT-WB merged into the train first; no other train ticket lists any of the four paths. None of the four paths is a product path, so by reading the ticket does not break the IIAC activation (no validator was run).

Independent audit (RWA-05, RW-PLAN v13 `audit_flags`): required for finding A4-005, auditing node `PT-Q02` (exact-SHA wave audit at G2, a separate session that is neither this ticket's implementer nor a plan reviewer). This ticket claims no audit outcome.

No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this ticket or its plan. A4-005, FRZ-DEBT-02 and FRZ-DEBT-03 are not closed by the draft; closure conditions are in plan section 11.

## Plan digest

Plan digest: 62053d90e6201d1fcf383e995e773bdd13b08ee1a471c6d3bc8c7b04261e4b7d

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-051-H.md` (drafted at `.tmp/session/arch-freeze/w3-drafts/deps-batch/PLAN_T-IGX-051-H.md`; the plan is copied unchanged into `governance/tickets/plans/` when the ticket is committed). The digest was computed with `scripts/governance/plan_digest.py` extracted from `origin/development`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-17T13:28:10+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_DELEGATION_RECORD_3.md@62dcc99faf7dc2ab95b83e8e5dae2ce845366ec2
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

Nothing was run as a test when this ticket was drafted (only throwaway runs of the gate script and of `uv lock` on copies in a scratch directory, recorded in plan section 12). One process at a time, one `uv run --frozen pytest` per verification step with multiple paths, under the lock `D:/Projekty/_wt/_locks/pytest.lock`; exact commands in plan section 7:

- Red-first (S0): `tests/unit/architecture/test_rw_deps_core_ownership_declared.py` on a detached worktree at the pre-change commit. Expected: 3 failures (`jsonschema` and `tzdata` cases of the per-requirement test, and the aggregate test) and 16 passes; the log is kept for the auditor.
- After the change (S1): all 19 items green.
- Regression (S2): `tests/unit/test_dependency_governance.py` and `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py` in one command. `test_current_project_passes_dependency_governance` is the existing mirror of `NT-RW-DEP-01` and is expected to turn green; the EBH-3 file does not test the pyproject gate and is regression only.
- L1 and lock (S3, S4): `uv run --frozen python scripts/maintenance/check_dependency_ownership.py` (expected exit 0; the exit code goes into the PR body) and `uv lock --check --offline`; the full L2 gate suite runs in CI because the lockfile may change.
- S5: `uv run --frozen ruff check` and `uv run --frozen pyright` on the changed Python files.
- This ticket edits nothing under `intergrax/contracts/`; if an amendment does, the OD-AF-11 wave-packet contract check (`measure_typ05.py` plus `typ_measure.py`) must report nothing new.
- No broad or diagnostic suite, no Docker, no vendor integration, no production configuration (C10/C15 freeze). The only network use is `uv lock` against the package indexes, and only if Part B carries an item.

## Out of scope

- Every product path (`intergrax/`, `agents/`, `applications/`), `.github/` (including `dependabot.yml` and the workflows), `docs/` (including `DEPENDENCY_OWNERSHIP.md` and the DEP4 document; question Q-DEP-05), every other gate script, the two existing dependency test files (regression only), the agent and application `pyproject.toml` files.
- Moving `jsonschema` and `tzdata` out of core (an install-contract change; question Q-DEP-04), a version upgrade of any locked package (question Q-DEP-02), the wiring of the ownership script into CI, the correction of the known-red register entry (questions Q-DEP-05, Q-DEP-06), the EDER part of FRZ-DEBT-02 and FRZ-DEBT-03 (`PT-R01..R03`).
- Any merge, comment, close, reopen or rebase of a dependency-bot PR; the IIAC re-pin, the EBH-7 pin, FRZ status promotion, stage closure, GitHub settings and rulesets, the roadmap, the FRZ checklist, PDC-X and IIAC record directories, Docker or vendor work.

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_734e59f8-2a58-4487-aae2-b39c0fc8bfd1
