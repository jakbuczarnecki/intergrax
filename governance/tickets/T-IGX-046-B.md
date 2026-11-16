# T-IGX-046-B — EQO-A remediation node PT-D06 (batch DB-E): PDC-X document corrections, legacy Critic tombstone

## Scope

Documentation-only implementation of the surviving hunks of batch DB-E of the EQO-A Campaign A remediation (baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589), re-verified at `origin/development` `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2` (7 hunks, 6 files). Findings: EQOA-A4-043, EQOA-A8-001, EQOA-A8-005 (A4-038, A8-004 attached), EQOA-A8-017. Out of scope: `PDC_INDEX.md` and every other PDC-owned record, `PDC_GAP_REGISTER.md` (generated), the PDC P7 audit reports, the hunks with owner placeholders or blocked on open decisions (DP-47), the roadmap, the FRZ checklist, every product path, `scripts/`, `tests/`, `.github/`. No PASS, FRZ promotion, stage closure or independent-verification claim.

## Plan digest

Plan digest: 248aeb4205fdaa62cd1d64304c61c6bc140121d721f64fba57096b7a4f4db388

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-046-B.md`](plans/PLAN_T-IGX-046-B.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Write set

```text
docs/project/architecture/CRITIC_VERIFICATION.md
docs/project/architecture/PROOF_RECEIPTS.md
docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md
docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md
docs/project/maintainers/plans/CRITIC_VERIFICATION.md
docs/project/maintainers/qualification/platform_domain_completeness/PDC_OWNERSHIP_MATRIX.md
```

## Approved

PENDING. The approval line is written by the Operator or a commissioned session after plan review; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it.

## Test contract

One process at a time, under the shared pytest lock:

- `uv run pytest tests/unit/docs/` (without `-p no:cacheprovider`: that flag conflicts with the repo pytest temp plugin and the run aborts with exit 4)
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Reviewer check: `git diff --name-only origin/development...HEAD` lists exactly the six Write set paths plus this ticket and its plan; every link added resolves.

## Out of scope

Edits of PDC-owned records, the PDC-P9-01 reference amendment of DP-52, owner assignments, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

`plan_pending`
