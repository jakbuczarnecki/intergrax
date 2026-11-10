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

Execution @ `development` `9fafaa288` (2026-11-10). **Not closed.** **No B6 PASS.** Lifecycle remains `draft`.

| Item | Status |
|------|--------|
| `WAVE1_F10_STATUS.md` | Published — dirty-tree @ `4a932d0cf` (639 identity paths; 182/182 TREE_DRIFT) + pinned replay rollup |
| `WAVE1_F10_PINNED_REPLAY.md` | Published — detached `f10-pin` replay SSOT |
| Render @ `4a932d0cf` (live tree) | Complete (2× exit 0); BLOCKED by PROFILE_BOUND drift |
| **Pinned replay @ `dc18e36d`** | **Complete** — generator exit 0; matrix **not** merged to `development` |
| In-tree F10 `TEST_EXECUTION_SHA` @ pin | **`496016c0`** (`MANIFEST.json` on tree @ `dc18e36d` — not weakened; differs from PRE-ENT campaign pin field on live `development`) |
| Pin `identity_violations` | **4 paths only** (`496016c0` → `dc18e36d`, outside B5 allowlist) — see pinned replay doc |
| Pin obligation FAIL attribution | **182 TREE_DRIFT** / **0 SEMANTIC** — C1–C5 **BLOCKED**; **not** independent B6 acceptance |
| `EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md` | Worktree/pinned render — **intentionally uncommitted** on `development` |
| F10 manifest ladder advance | **Open** — operator policy after Wave 1+ batches; no gate weakening |

**Pin violation paths (4):** `scripts/qualification/ebh7_junit_evidence.py`, `tests/qualification/ebh_7/_ebh7_skip_disposition_register.py`, `tests/qualification/ebh_7/_semantic_disposition_registry.py`, `tests/qualification/ebh_7/test_ebh_7_b6_junit_skip_disposition_gates.py`.

Full gate table and routing: [`WAVE1_F10_STATUS.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE1_F10_STATUS.md) · [`WAVE1_F10_PINNED_REPLAY.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE1_F10_PINNED_REPLAY.md) · campaign rollup [`PRE_ENT_CAMPAIGN_PROGRESS.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_CAMPAIGN_PROGRESS.md).
