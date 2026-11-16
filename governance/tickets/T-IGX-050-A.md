# T-IGX-050-A — ARCH-FREEZE program W0: decision register, SSOT reconciliation (part A) and plan publication

## Scope

Documentation-only. Three MEP-AF2 v2.1 nodes in one batch under `docs/project/maintainers/qualification/`:

- **T-AF2-OD-RECORD:** `ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md` converted to the table `OD_ID | Question | Decision | Date | Authority / record | Status`; original rows kept with corrected statuses; OD-AF-00, packet-1 and packet-2 rows (including OD-AF-01a, the renumbering to T-IGX-050/051/052), OD-REM-01..08, C1–C10 and one summary row per decision session DS-1..DS-6 (DP-01..DP-68) added as labelled English renderings citing the session records by path; OD-ARCH-FREEZE-01 stays OPEN.
- **T-AF2-SSOT-RECONCILE-A:** `ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md` (new `V21_QUALIFICATION` column, MCP-SERVER-X and PDC-X gate notes, task-count supersession pointer), `ARCH_FREEZE_SHA_LADDER.md` (`AUDIT_BASELINE_2`, placeholders G2/R/FCS/M3), `ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md` (current-state addendum), `ARCH_FREEZE_INDEPENDENT_AUDIT_AND_CLOSURE_MATRIX.md` (OD-ENT-08 row), `ARCH_FREEZE_EXECUTION_WAVE1_INDEPENDENT_AUDIT_HANDOFF.md` and `ARCH_FREEZE_COMPLETE_IMPLEMENTATION_AND_CERTIFICATION_MASTER_PLAN.md` (header notes).
- **T-AF2-PLAN-PUBLISH:** new folder `docs/project/maintainers/qualification/arch_freeze_program/` with the MEP-AF2 v2.1 plan copy, four WP-12 planning records, `README.md` and `SOURCES.sha256`.

Out of scope: every product path (`intergrax/`, `agents/`, `applications/`), `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, `ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md` and `ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md` (edited by PR #606, then by T-AF2-SSOT-RECONCILE-B), `qualification/README.md` (edited by PR #607), IIAC records and scripts, PDC-X record directories, GitHub settings. No PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 925de1a46c813c145a4d3dce754b8f341242c1468a83aa6f6b4ac221808ec925

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-050-A.md`](plans/PLAN_T-IGX-050-A.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

PENDING. <<OPERATOR: the approval line goes here after plan review, written by the Operator or a commissioned session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).>>

## Test contract

One process at a time, under the shared lock (`mkdir <worktree base>/_locks/pytest.lock`), logs under `.tmp/session/arch-freeze/W0-ssot-a/`:

- `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` (the WBS table gained a column and the pre-admission report was edited; the validator reads both).
- `uv run python scripts/docs/check_docs_domain_pairs.py`.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the files in the plan write set, plus this ticket and its plan.
- Reviewer check: the WBS qualification counts are 6 / 12 / 1 / 1 / 7 / 1 / 26 (total 54).

## Out of scope

Code, tests, scripts, workflows, roadmap, FRZ checklist, the two registers owned by PR #606, `qualification/README.md`, EDER, product paths, GitHub rulesets, Docker, vendor integration, production configuration, FRZ promotion.

## Lifecycle state

`plan_pending`
