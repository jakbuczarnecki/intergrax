# T-IGX-050-C — ARCH-FREEZE W0: PDC-X residual qualification package (T-AF2-DS-EXC-PACKAGE)

## Scope

Documentation-only. Adds `docs/project/maintainers/qualification/PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md`: Table A (60 PDC-X declared exceptions), Table B (13 `PARTIAL` rows) and Table C (other handoff residuals) with the 13 columns of plan MEP-AF2 v2.1 section E. It is a qualification of residual admissibility for ARCH-FREEZE, not a re-certification of PDC-X; no row is accepted or closed, and final acceptance stays with the W7 independent audit. Baseline: `7960ae7f6c82da60ca9acdc0ceebf63810002a42`. Out of scope: every product path, `scripts/`, `tests/`, `.github/`, the PDC-X record directories, the roadmap, the FRZ checklist, IIAC files. No PASS, FRZ promotion, stage closure or independent-verification claim.

## Plan digest

Plan digest: 11922d786cd8896a8c33b4be05aeae10e31f89a5d111cb8c5b0038afdc54833b

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-050-C.md`](plans/PLAN_T-IGX-050-C.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T16:59:46+01:00
Delegation: docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md@31ba3aa84490ad7984106bdeb410163e873b0fc1

## Test contract

One process at a time, no product test:

- `python -I` exactly-once script over the new file and the PDC registers (60/60 EX ids, 13/13 `PARTIAL` ids); output recorded in the PR body.
- `uv run python scripts/docs/check_docs_domain_pairs.py` only if the file location triggers a pair rule.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the new qualification file, this ticket and its plan.

## Out of scope

Code, tests, scripts, workflows, roadmap, FRZ checklist, PDC-X records, reachability measurement (T-AF2-CLOSED-WORLD), FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
