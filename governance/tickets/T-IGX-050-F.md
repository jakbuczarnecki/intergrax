# T-IGX-050-F — ARCH-FREEZE W2: RM@499 entry map (T-AF2-RM499-ENTRY)

## Scope

Documentation-only. Adds `docs/project/maintainers/qualification/arch_freeze_program/RM499_ENTRY_MAP.md`: every mechanical entry condition of the roadmap ARCH-FREEZE row (L499) and of its Path B row (L267), quoted verbatim, each ending `OPEN` with the plan node that closes it or `EVIDENCED` with a `path@sha`. It covers the roadmap OPEN/BLOCKED count with the TRACE-X rows at L401 and L410, the Harness, Top-Tier, compatibility-seam, frozen-contract, reopened-stage and replay-finding counters, the 7 VBN artefacts, and the independence of SKILL-X, TOOL-LIFE-X, DIST-SCALE-X, EBH-7 and ROADMAP-REPLAY-X; it adds an interpretation note for the post-freeze PLANNED and BLOCKED rows. Baseline: `332c5217c390202a911ad5592e88a8f366845587`. Out of scope: the roadmap and the FRZ checklist (Operator-only), every product path (`intergrax/`, `agents/`, `applications/`), `scripts/`, `tests/`, `.github/`, the PDC-X record directories, IIAC files, the FRZ matrix and the gap register. No PASS, FRZ promotion, stage closure or independent-verification claim.

## Plan digest

Plan digest: dc90800c5e6ffc93e31d0bc358f56cbaa4c25b9a23c48879ceef7e43614c3761

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-050-F.md`](plans/PLAN_T-IGX-050-F.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T18:44:15+01:00
Delegation: docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md@24b9c9e72b68de61c36523738e02677d318b1f40

## Test contract

One process at a time, taken under the pytest lock, no product test:

- The map generator script exits 0: every quoted condition is verbatim in the roadmap at the base SHA, every cited record path exists.
- `python scripts/governance/verify_arch_freeze_plan_traceability.py` (expected: exit 0, unchanged counters).
- `python scripts/docs/check_docs_domain_pairs.py`.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the map, this ticket and its plan.

## Out of scope

Code, tests, scripts, workflows, roadmap, FRZ checklist, other qualification records, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
