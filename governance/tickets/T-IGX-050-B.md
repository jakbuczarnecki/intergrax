# T-IGX-050-B — ARCH-FREEZE program W0: traceability validator amendment (T-AF2-VALIDATOR-AMEND)

## Scope

Amend `scripts/governance/verify_arch_freeze_plan_traceability.py` as specified by MEP-AF2 v2.1 section C items 1-4 and C.4, and add `tests/unit/governance/test_arch_freeze_plan_traceability.py`:

- every hard DAG rule unchanged; one additive rule (pre-lock ancestors `T-B5-EVOL-FRZ` and `T-B6-FRZ-AUD` never follow `T-B7-ECAP-2C` / `T-B7-LOCK-FREEZE-SHA`);
- `SUPERSEDED_BY_CURRENT_AUTHORITY` is a satisfied dependency and requires a non-empty `authority_ref`;
- counters computed from the files (checklist counts, WBS = 54 historical + `T-AF2-*` / `PT-*` rows, qualification runs), with two invariants kept (225 FRZ rows in 26 families; at least 16 qualification runs);
- matrix-vs-checklist cross-check opt-in via the matrix header line `**MATRIX_CHECKLIST_CROSSCHECK:** ENABLED`;
- word-boundary task-ID regex that accepts `PT-` and `T-AF2-` IDs;
- `FIND-PROD-REACH-DEFERRED` at severity INFO;
- DAG self-test (cycle check, the 20 COORD prerequisites, every hard rule, and a fixture rejecting the edge `T-B5-EVOL-FRZ <- T-B7-ECAP-2C`);
- `--sync-wbs` and `--sync-matrix` forbidden: exit code 2, message pointing to plan section C, no file touched.

The change is tooling only. It certifies nothing, promotes no FRZ criterion and changes no document it reads. The independent review required by IIAC for a validator that a certifier uses is a separate node (T-AF2-VALIDATOR-INDEP-REVIEW).

## Plan digest

Plan digest: 8903bd6e98785cd7b33fa8718d642e520d49b54833b446539d7609877d1b74f0

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-050-B.md`](plans/PLAN_T-IGX-050-B.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

PENDING. The approval line is added after plan review by the Operator or a commissioned session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

One process at a time, under the shared lock (`mkdir <worktree base>/_locks/pytest.lock`), logs under `.tmp/session/arch-freeze/T-AF2-VALIDATOR-AMEND/`:

- `uv run pytest tests/unit/governance/test_arch_freeze_plan_traceability.py` (positive cases, negative cases, current repository state).
- `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` on the branch head (current repository state passes).
- `uv run ruff check` and `uv run ruff format --check` on the two changed Python files.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the four files in the plan write set.

## Out of scope

Edits of the WBS, matrix, gap register, checklist or any other document; IIAC material scripts; product paths; the roadmap; `.github/`; GitHub settings; the independent validator review; FRZ promotion; any certification claim.

## Lifecycle state

plan_pending
