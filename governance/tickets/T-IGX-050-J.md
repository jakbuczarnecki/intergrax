# T-IGX-050-J — ARCH-FREEZE W2 node T-AF2-RED-CLASSES: known-red register and closure-regex lint

## Scope

Documentation, one lint script with its unit test, and one comment fix, drafted for node T-AF2-RED-CLASSES of the merged ARCH-FREEZE plan (MEP-AF2 v2.1). Adds a known-red register (`KNOWN_RED_REGISTER.md` and `known_red_register.v1.json`, classes FIX / ENV / OUT / UNKNOWN with evidence, owner node and how to prove ENV), a lint `scripts/governance/lint_iiac_closure_regex.py` that imports the `_CLOSURE` pattern from `iiac_activation_gate.py` and flags matches in activation records and operator decision texts, its test, and a comment fix in the header of `.github/workflows/unit-tests.yml` (lines 10-11; the brief said 9-10, the stale text is on 10-11). Baseline of the measurements: `AUDIT_BASELINE_2 = 7960ae7f6c82da60ca9acdc0ceebf63810002a42`. Out of scope: every product path, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, every `IIAC_*` file and script, the PDC-X record directories. No PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 861f1e47453958c8e2aba6af9352543fbe62b3e7462efb0bb7ab2b7cebf00b8c

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-050-J.md`](plans/PLAN_T-IGX-050-J.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T18:52:40+01:00
Delegation: docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md@24b9c9e72b68de61c36523738e02677d318b1f40

## Test contract

One process at a time, under the shared lock `D:/Projekty/_wt/_locks/pytest.lock`:

- `uv run pytest tests/unit/governance/test_lint_iiac_closure_regex.py`
- `uv run python -m scripts.governance.lint_iiac_closure_regex` (one run, default targets; result reported as measured)
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the seven paths of the plan's write set.

## Out of scope

Code under `intergrax/`, `agents/`, `applications/`; dependency files; IIAC scripts and records; installing missing CI dependencies; enumerating node ids the sources do not give; assigning owners the plan does not name; Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
