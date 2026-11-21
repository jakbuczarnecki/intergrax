# T-IGX-051-I — RW train: thread the A4-007 import allowlist through the product callers (seams b1 and b2)

## Scope

Product-path ticket of the RW product train (node `T-AF2-A4007-CALLERS`, added after RW-PLAN v15; earliest lane by dependency 3, capacity-placed lane 4, after `PT-FC2`; approval route OD-AF-22: one wave packet approved by the Operator, then signed by the delegate). It closes, as far as callers can, the A4-007 seams b1 and b2 that `T-IGX-043-H` (PT-WB, merged into `arch-freeze/rw-train` as `6b94d202f`) left open as a recorded residual: PT-WB landed the loader capability (`allowed_prefixes`, `fail_closed`, typed refusals, the helper `import_allowlist.py`) but no product caller passes either parameter, so the check is a no-op on production paths (PT-WB ticket "Implementer checkpoint"; PR #654 review finding H-1, disposition "recorded residual (orchestrator ruling)"; the ARCH-FREEZE orchestrator ruled on 2026-11-17 that the residual is owned by this separate ticket, because `T-IGX-043-B` was approved at digest `14e4de95` and must not be amended again).

What it does (plan sections 4 to 6): Part A, one new module `intergrax/applications/_shared/import_policy.py` that derives the permitted package sets from the registered product manifests (no hand-typed list; an empty registry permits nothing in a guard posture); Part B, the agent-import callers pass the Tier-2 and factory sets and `fail_closed` in the guard postures, with the posture decided only by `wiring_fails_closed(env)` from `wiring_posture.py` (`wiring.py`, `agent_resolution.py`, `environment_conformance.py`, `roster_agent_contract_authority.py`; conditional `package_wiring.py`); Part C, the two callers of `check_application_migrations` (scripts) pass the application-package set and `fail_closed`. LAB and every non-guard posture stay unrestricted as today. Typed rosters (`AgentBinding.mount`) never import and are unaffected.

Write set (exact list in the plan, section 7): 10 firm paths (5 product paths under `intergrax/`, 1 of them new; 2 scripts; 3 new tests) and 2 conditional paths (`package_wiring.py`, `environment_diff_wiring.py`); 10 negative tests (9 firm, 1 conditional) that are red by assertion at the red-first base, 6 controls (two of them pin the revision-bound shapes found by the step-0 dry run), 5 unit tests of the new module; regression scope 25 files (the two AC3 production-registry files were added, not edited). The census found that `environment_diff_wiring.py:108` cannot reach an import (a control test pins it), and listed two callers outside the census that stay residuals (`load_agent_class`, the reference-roster compat call).

Dependencies and overlaps: `PT-FC2` merged first (it creates `wiring_posture.py`; absent at the train heads `6b94d202f` and `881428fb2`, and not on `origin`: **the ticket cannot start before PT-FC2 merges**); `PT-WA` and `PT-WB` merged. No other train ticket lists a write-set path (the shared paths belong to the merged PT-WB; serialization, not conflict). RW-PLAN v18 carries the block and the lane entry; the A4-007 b1/b2 ownership move remains (plan section 9, question Q-A7C-07). Plan revision 2 (section 0 of the plan) answers the step-0 dry run: installed distribution agents need no second source because they are identified by digest and carry no `import_path`; the rehydration context stays a residual (RES-5, Q-A7C-09).

Product commits of this ticket go only to the branch `arch-freeze/rw-train`. The five `intergrax/` paths are product paths: every merge of them breaks the IIAC activation until the single wave re-pin (ACT-N11), and a revert does not restore it; the two scripts are not product-prefix paths. After the wave: independent audit at the exact wave SHA by a session that did not implement or review it. No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this ticket or its plan; A4-007 is not claimed closed.

## Plan digest

Plan digest: 646fdfb58226876a55fc643f6cc90a977a6b4d1485d109e2f532c96025e33c70

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-051-I.md`](plans/PLAN_T-IGX-051-I.md) (drafted at `.tmp/session/arch-freeze/w3-drafts/a4007-callers/PLAN_T-IGX-051-I.md`; the plan is copied unchanged into `governance/tickets/plans/` when the ticket is committed). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-18T08:24:46+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_STANDING_DELEGATION_RECORD.md@881428fb231f34215a67364aa5788a4f2f986e83
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

Nothing was run as a test when this ticket was drafted. One process at a time, one `uv run pytest` per verification step with multiple paths, under the shared test lock `D:/Projekty/_wt/_locks/pytest.lock` (only the creator of the lock removes it); no broad or diagnostic suite. The commands V1..V6 of plan section 11 are the contract:

- Red-first (plan section 10, step 1): the three new test files on a detached worktree at the train head at the start of this ticket (PT-WB and PT-FC2 merged; the SHA is recorded in the ticket checkpoint; the b1/b2 defect is unchanged from `a7f023615`, but the tests need PT-WB code); expected: NT-A7C-01..07, 09 and 10 fail by assertion (and 08 when `package_wiring.py` is in), the controls pass; the derivation file is red by the missing module and is not counted as negative. The log is kept under `.tmp/session/arch-freeze/T-IGX-051-I/red-first/` for the auditor (PT-Q02 re-runs from that path).
- V1 new tests: `uv run pytest tests/unit/applications/test_a4007_caller_import_refusal.py tests/unit/applications/test_a4007_import_policy_derivation.py tests/unit/scripts/test_a4007_migration_gate_allowlist.py -p no:cacheprovider`
- V2 regression: the 25 existing files of plan section 11 in one command (199 test definitions; node expansion is step-0 work). Two of them, `test_production_registry_startup_ac3_final.py` and `test_ac3_phase5_production_lifecycle_e2e.py`, are expected green by reading; red at step 0 is a stop. A mechanical lint commit on the touched files comes first (plan section 10, step 1a).
- V3 `uv run python scripts/maintenance/check_application_migrations.py` (exit code to the PR body).
- V4 step-0 and final `git grep` census of the callers (no pytest), repeated by the auditor.
- V5 `uv run ruff check` and `uv run pyright` on the changed Python files; V6 `uv run python -m scripts.governance.igx_gov ticket` before push.
- The ticket edits nothing under `intergrax/contracts/`; if an amendment does, the OD-AF-11 wave-packet contract check (`measure_typ05.py` plus `typ_measure.py`) must report nothing new.
- Reviewer check: `git diff --stat` lists exactly the plan's write set; commits carry the trailer `Ref: governance/tickets/T-IGX-051-I.md` and no AI-tool attribution.
- Independent wave audit at the exact RW SHA is a separate session and is not part of this contract.

## Out of scope

`wiring_posture.py` and every `T-IGX-043-B` path; the loaders and the helper `import_allowlist.py`, `migration_wiring.py`, `mcp_import_guard.py`, `scoreboard.py` and every other `T-IGX-043-H` path not listed in the write set; `load_agent_class` and the reference-roster staging helper (recorded residuals); the other importlib loaders outside A4-007 (`context_engine_resolver.py` and five more; `PT-G01`); the standing gates, the seam ledger and any gate-meaning change; `intergrax/contracts/**`; the registry, the manifest contract and the Tier-3 catalog (read only); existing tests (regression only); docs, the roadmap, the FRZ checklist, the known-red and EDER registers; RW-PLAN itself (v16 is the orchestrator's); IIAC activation and the RB-2 re-pin; GitHub settings; `.github/`, `pyproject.toml`, `uv.lock`; production configuration, Docker migration, new vendor integrations, S1 or diagnostic suites, PROD-Q node runs; FRZ promotion.

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_eaf9ef72-d1c1-44d0-acbc-fac05f02edf7
