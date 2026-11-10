# T-IGX-034-F — PRE-ENT Wave 1: EBH-7 B6-F10 normative matrix on clean execution tree (no B6 PASS)

## Scope

Wave 1 batch **`f10`** of the PRE-ENT predecessor certification campaign. Recompute EBH-7 B6-F10 C1–C5 normative certification via `scripts/qualification/render_ebh7_f10_normative_matrix.py`, classify **TREE_DRIFT** vs **SEMANTIC** obligation failures, and publish status in:

`docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE1_F10_STATUS.md`

Distinguish PROFILE_BOUND failures (executed tree ≠ `TEST_EXECUTION_SHA` outside B5 allowlist) from genuine missing obligations. **Do not weaken** F10 gates or allowlists for campaign convenience.

No independent B6 acceptance, no G6 PASS, no FRZ promotion, no ECAP PASS claims in prose.

## Plan digest

Plan digest: (pending — PRE-ENT campaign plan `pre-ent_certification_closure`)

## Approved

(pending — operator acceptance of PRE-ENT campaign plan)

## Test contract

```text
# Clean tree @ TEST_EXECUTION_SHA from MANIFEST (detached worktree recommended)
uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py
uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py --check

# Read-only gate (when uv available)
uv run --extra dev-ci pytest tests/qualification/ebh_7/test_ebh_7_b6_f10_f11_corrective_gates.py -q --tb=line
```

Wave 1 dirty-`development` render @ `4a932d0cf` (2026-11-10): disposition **BLOCKED**; 639 identity violations vs `dc18e36d`; 182/182 obligation FAILs attributed **TREE_DRIFT** on regenerated matrix row scan.

## Out of scope

- Weakening P03 identity rules or `TEST_EXECUTION_SHA` without governed MANIFEST ladder update
- Independent auditor session / B6 sign-off
- T-IGX-034-I matrix publication closure (Wave 4+)
- Product implementation owned by 034-B..E (except regeneration-driven doc updates)

## Lifecycle state

draft

## Checkpoint

| Item | Status |
|------|--------|
| `WAVE1_F10_STATUS.md` | Published 2026-11-10 (dirty-tree classification + clean-checkout requirements) |
| Render @ `4a932d0cf` | Complete (2× exit 0, ~8–10 min) |
| Clean worktree replay @ `dc18e36d` | **Open** — required for semantic FAIL inventory |
| Live `development` @ `f97d286b8+` | BND-03B merged after render; **re-run generator** when operator scopes F10 to current tip |

Full gate table and routing: [`WAVE1_F10_STATUS.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE1_F10_STATUS.md).
