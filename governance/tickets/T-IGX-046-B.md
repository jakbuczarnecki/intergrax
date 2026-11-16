# T-IGX-046-B — EQO-A remediation node PT-D06 (batch DB-E): PDC-X document corrections

## Scope

Documentation-only implementation of the surviving hunks of batch DB-E of the EQO-A Campaign A remediation (baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589), re-verified at `origin/development` `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2` (5 hunks, 4 files). Findings: EQOA-A4-043, EQOA-A8-005 (A4-038, A8-004 attached), EQOA-A8-017. Out of scope: `PDC_INDEX.md` and every other PDC-owned record, `PDC_GAP_REGISTER.md` and `PDC_CLOSED_WORLD_INVENTORY.json` (generated), the legacy `CRITIC_VERIFICATION.md` tombstone (it would change the generated inventory), the PDC P7 audit reports, the hunks with owner placeholders or blocked on open decisions (DP-47), the roadmap, the FRZ checklist, every product path, `scripts/`, `tests/`, `.github/`. No PASS, FRZ promotion, stage closure or independent-verification claim.

## Plan digest

Plan digest: fb85c2b6f39843f2fc12d483b8454065d9e1923cd845711b1c81d4b24909e058

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-046-B.md`](plans/PLAN_T-IGX-046-B.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Write set

```text
docs/project/architecture/PROOF_RECEIPTS.md
docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md
docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md
docs/project/maintainers/qualification/platform_domain_completeness/PDC_OWNERSHIP_MATRIX.md
```

## Approved

Approved: operator-delegate @ 2026-11-16T18:14:01+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@743087d57c674100dc9119959dc875271f70e72c

## Test contract

One process at a time, under the shared pytest lock:

- `uv run pytest tests/unit/docs/` (without `-p no:cacheprovider`: that flag conflicts with the repo pytest temp plugin and the run aborts with exit 4)
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `python -I scripts/qualification/pdc_x_inventory.py --check` and `uv run pytest tests/unit/governance/test_pdc_x_inventory.py`
- Reviewer check: `git diff --name-only origin/development...HEAD` lists exactly the four Write set paths plus this ticket and its plan.

## Out of scope

Edits of PDC-owned records, the PDC-P9-01 reference amendment of DP-52, owner assignments, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
