# PRE-ENT Wave 1 — EBH-7 B6-F10 pinned replay (T-IGX-034-F)

**Campaign:** `pre_ent_enterprise_certification`  
**Date:** 2026-11-10  
**Executor:** T-IGX-034-F — **no B6 PASS**, no independent acceptance

## Execution context

| Role | SHA | Note |
|------|-----|------|
| **Worktree** | `D:/Projekty/_wt/intergrax/f10-pin` | Detached HEAD |
| **Executed HEAD** | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` | P03 / operator pin checkout (Wave 1 ladder reference on live `development`) |
| **`MANIFEST.json` `TEST_EXECUTION_SHA` @ this commit** | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` | F10 `build_canonical_f10_inputs()` reads this field — not the checkout SHA while ladder on `development` names `dc18e36d` |
| **Generator** | `uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py` | Exit 0 @ `dc18e36d`; matrix refreshed in worktree only (**not** merged to `development`) |

Contrast: [`WAVE1_F10_STATUS.md`](WAVE1_F10_STATUS.md) **f10-clean** section @ `f97d286b8` uses live `development` `MANIFEST` pin `dc18e36d` → **662** identity violations.

## Tree identity (B5 allowlist)

| Signal | Value |
|--------|------:|
| **`identity_violations`** | **4** (`496016c0` → `dc18e36d` diff outside B5 allowlist) |
| **`in_process_failures`** | **0** |
| **Paths** | `scripts/qualification/ebh7_junit_evidence.py`, `tests/qualification/ebh_7/_ebh7_skip_disposition_register.py`, `tests/qualification/ebh_7/_semantic_disposition_registry.py`, `tests/qualification/ebh_7/test_ebh_7_b6_junit_skip_disposition_gates.py` |
| **Tree identity satisfied?** | **No** — four qualification paths between manifest TES and executed HEAD |

B5 allowlist **unchanged** (no weakening for PRE-ENT).

## Computed disposition @ pinned replay

**`EBH-7 — B6-F10 BLOCKED / EXACT MISSING OBLIGATIONS`** — not qualified, not B6 acceptance.

| Gate | Verdict | Obligations | PASS | N/A WITH EVIDENCE | FAIL |
|------|---------|-------------|------|-------------------|------|
| **C1** | BLOCKED | 32 | 0 | 4 | 28 |
| **C2** | BLOCKED | 43 | 0 | 0 | 43 |
| **C3** | BLOCKED | 105 | 0 | 6 | 99 |
| **C4** | BLOCKED | 9 | 0 | 0 | 9 |
| **C5** | BLOCKED | 3 | 0 | 0 | 3 |

**Total obligation FAIL rows:** 182.

## TREE_DRIFT vs SEMANTIC

| Class | Count | Basis |
|-------|------:|-------|
| **TREE_DRIFT** | **182** | Every FAIL row cites `executed tree differs from TEST_EXECUTION_SHA` (PROFILE_BOUND attestation) |
| **SEMANTIC** | **0** | Row scan — no FAIL without tree-drift attestation string |

**Read:** Even with only **four** identity paths between manifest TES and `dc18e36d` HEAD, F10 still attributes **all** obligation FAILs to tree drift. **Semantic** gaps are **not** isolated until `identity_violations` = 0 (e.g. detached replay @ `496016c0` with matching HEAD, or governed MANIFEST / ladder alignment). Gates unchanged.

## Operator summary

| Gate | Verdict |
|------|---------|
| C1–C5 | BLOCKED (same FAIL counts as dirty / `f97d286b8` ladder render) |
| **B6-F10** | **BLOCKED** — **not PASS** |

## Related SSOT

- Clean ladder replay: [`WAVE1_F10_STATUS.md`](WAVE1_F10_STATUS.md) (incl. `f10-clean` @ `f97d286b8`)
- Ticket: `governance/tickets/T-IGX-034-F.md`
