# PRE-ENT Wave 1 — EBH-7 B6-F10 pinned replay (T-IGX-034-F)

**Campaign:** `pre_ent_enterprise_certification`  
**Date:** 2026-11-10  
**Executor:** T-IGX-034-F — **no B6 PASS**, no independent acceptance

## Execution tree

| Field | Value |
| --- | --- |
| **Worktree** | `D:/Projekty/_wt/intergrax/f10-pin` (detached HEAD) |
| **Executed HEAD** | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` |
| **PRE-ENT ladder pin (development SSOT)** | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (`MANIFEST.json` `TEST_EXECUTION_SHA` on current `development`) |
| **F10 attestation `TEST_EXECUTION_SHA` @ execution** | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` (in-tree `MANIFEST.json` at `dc18e36d` — not weakened) |
| **Generator** | `uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py` (exit 0) |
| **Disposition** | `EBH-7 — B6-F10 BLOCKED / EXACT MISSING OBLIGATIONS` |

**Publication rule:** matrix and checklist outputs were regenerated **only inside the worktree** — **not** merged to `development`.

## Tree identity (B5 allowlist)

| Signal | Value |
| --- | --- |
| **`identity_violations`** | **4 only (not zero)** — `496016c0` → `dc18e36d`, outside B5 allowlist |
| **`in_process_failures`** | **0** |
| **Tree identity satisfied for F10?** | **No** — executed HEAD ≠ in-tree F10 `TEST_EXECUTION_SHA` |

**Violation paths (full list):**

- `scripts/qualification/ebh7_junit_evidence.py`
- `tests/qualification/ebh_7/_ebh7_skip_disposition_register.py`
- `tests/qualification/ebh_7/_semantic_disposition_registry.py`
- `tests/qualification/ebh_7/test_ebh_7_b6_junit_skip_disposition_gates.py`

## C1–C5 @ pinned replay

| Gate | Verdict | Obligations | PASS | N/A WITH EVIDENCE | FAIL |
| --- | --- | ---: | ---: | ---: | ---: |
| **C1** | BLOCKED | 32 | 0 | 4 | 28 |
| **C2** | BLOCKED | 43 | 0 | 0 | 43 |
| **C3** | BLOCKED | 105 | 0 | 6 | 99 |
| **C4** | BLOCKED | 9 | 0 | 0 | 9 |
| **C5** | BLOCKED | 3 | 0 | 0 | 3 |

**Total obligation FAIL rows:** 182.

## TREE_DRIFT vs SEMANTIC

| Class | Count | Basis |
| --- | ---: | --- |
| **TREE_DRIFT** | **182** | Every FAIL row cites `executed tree differs from TEST_EXECUTION_SHA` (automated scan post-`certify_f10_for_render`) |
| **SEMANTIC** | **0** | No FAIL row without tree-drift attestation string |

## Read vs ladder `f10-clean`

| Replay | HEAD | Campaign `identity_violations` vs PRE-ENT pin `dc18e36d` | SEMANTIC |
| --- | --- | ---: | ---: |
| **`f10-clean`** (`f2682d651` / ladder) | Wave 1 product HEAD | 662 | 0 |
| **`f10-pin`** (this record) | P03 replay checkout `dc18e36d` | 4 (F10 manifest pin `496016c0` vs HEAD) | 0 |

Pinned replay **reduces** PROFILE_BOUND drift versus ladder HEAD but **does not** clear F10 attestation: in-tree manifest at `dc18e36d` still anchors F10 to `496016c0`, and four qualification paths differ. **Gates and B5 allowlist unchanged.**

## Related SSOT

- Status rollup: [`WAVE1_F10_STATUS.md`](WAVE1_F10_STATUS.md)
- Ticket: `governance/tickets/T-IGX-034-F.md`
