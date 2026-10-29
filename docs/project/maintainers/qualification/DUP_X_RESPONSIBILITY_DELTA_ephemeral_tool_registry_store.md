# DUP-X — Responsibility Delta: EphemeralToolRegistryStore

| Field | Value |
| --- | --- |
| **Symbol** | `intergrax/runtime/codecraft/ephemeral_registry.py` |
| **Candidate** | DUPX-CAND-13 |
| **Disposition** | **DISTINCT BOUNDED LIFECYCLE** |

## 1. Exact responsibility

In-process ephemeral tool registration for codecraft / lab execution scopes with explicit teardown — not global platform ToolRegistry.

## 2. Authority / truth owned

Ephemeral session store only; **does not** replace marketplace or qualified tool catalog.

## 3. I/O contracts + lifecycle

Created per codecraft run; discarded after session — bounded lifecycle distinct from persistent registry.

## 4. Caller graph

Codecraft tests and UCA flows (`test_uca6c_r4_real_codecraft_execution.py`).

## 5. Why canonical owner cannot absorb

Different lifecycle and isolation requirements vs platform ToolRegistry.

## 6. Convergence point

Qualified tool activation still routes through platform resolvers for production paths.

## 7. Negative bypass proof

Unit tests assert `EphemeralToolRegistryStore` type isolation from `ToolRegistry` catalog authority.
