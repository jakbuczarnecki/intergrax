# DUP-X — Responsibility Delta: context dedup wrappers

| Field | Value |
| --- | --- |
| **Symbols** | `intergrax/context/dedup.py` → `intergrax/context/policy/exact_dedup.py` |
| **Candidate** | DUPX-CAND-06 |
| **Disposition** | **THIN TYPED ADAPTER** |

## 1. Exact responsibility

Stable public import surface for fragment exact-dedup policy used by context assembly callers.

## 2. Authority / truth owned

**None** — delegates semantic dedup rules to `exact_dedup_fragments` / policy module.

## 3. I/O contracts + lifecycle

Pass-through to policy functions; no parallel store.

## 4. Caller graph

Context pipeline modules importing `dedup` shim.

## 5. Why canonical owner cannot absorb

Preserve backward-compatible import path without duplicating policy logic.

## 6. Convergence point

`intergrax/context/policy/exact_dedup.py`

## 7. Negative bypass proof

Mechanical import graph — single policy implementation file owns algorithms.
