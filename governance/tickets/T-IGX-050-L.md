# T-IGX-050-L — ARCH-FREEZE program W2: fixes for the independent review of the traceability validator (T-AF2-VALIDATOR-FIX)

## Scope

Fix the findings of the independent review of the validator amended by T-IGX-050-B (review verdict `VALIDATOR_REVIEW_FINDINGS`: 2 MAJOR, 2 MINOR, 3 LOW, 0 BLOCKER), in `scripts/governance/verify_arch_freeze_plan_traceability.py` and `tests/unit/governance/test_arch_freeze_plan_traceability.py`:

- F1: a malformed task ID (a dependency token or a WBS row) is an error instead of being dropped silently;
- F2: the new edges of MEP-AF2 v2.1 section C.3 are validator rules (26 edges, active once any post-DAG-REGEN row exists) with one self-test mutation each; ROADMAP/ECAP-16/VBN-HEAD edges stay covered by the existing hard rules;
- F3: every `V21_QUALIFICATION` state must belong to the section C vocabulary;
- F4: the matrix-vs-checklist cross-check always runs;
- F6: an `authority_ref` must contain an identifier-like token;
- F5 is documented in the plan and not changed; F7 (documents outside the write set) is not included.

The change is tooling only. It certifies nothing, promotes no FRZ criterion and changes no document the validator reads.

## Plan digest

Plan digest: 18a38d7170e9fa33f59061a3400466896549707e8664c945f055245fd02d43c6

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-050-L.md`](plans/PLAN_T-IGX-050-L.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

PENDING. <<OPERATOR: the approval line goes here after plan review, written by the Operator or a commissioned session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).>>

Implementer session: local_5b1805e3-955e-4e76-81a6-c175799d8908

## Test contract

One process at a time, under the shared lock (`mkdir <worktree base>/_locks/pytest.lock`), logs under `.tmp/session/arch-freeze/T-AF2-VALIDATOR-FIX/`:

- `uv run pytest tests/unit/governance/test_arch_freeze_plan_traceability.py` (new tests fail on the unfixed validator and pass on the fixed one; one red run of the new tests, one green run of the module).
- `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` on the branch head (the current repository state passes).
- `uv run ruff check` and `uv run ruff format --check` on the two changed Python files.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the four files in the plan write set.

## Out of scope

Edits of the WBS, matrix, gap register, checklist or any other document; F7 (documentation); DAG-REGEN itself; IIAC material scripts; product paths; the roadmap; `.github/`; GitHub settings; a new independent review; FRZ promotion; any certification claim.

## Lifecycle state

plan_pending
