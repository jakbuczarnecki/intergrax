# DUP-X — Responsibility Delta: `registry_v2`

| Field | Value |
| --- | --- |
| **Symbol** | `intergrax/runtime/integrations/registry_v2.py` |
| **Candidate** | DUPX-CAND-02 |
| **Disposition** | **THIN TYPED ADAPTER** — catalog projection |

## 1. Exact responsibility

Build read-only integration registration snapshots and contract registry views from **catalog truth** (`integrations/registry/catalog`, contract specs) — not alternate plugin registration authority.

## 2. Authority / truth owned

**None** for vendor selection; **projection only** for runtime contract typing.

## 3. I/O contracts + lifecycle

Inputs: catalog entries, `IntegrationContractSpec`, category contracts. Outputs: immutable snapshots (`build_contract_registry_snapshot`, `build_integration_registration`). No env slug merge.

## 4. Caller graph

Runtime integration wiring, unit tests (`test_canonical_registry_projection.py`). Composition converges on `resolve_from_profile` @ `integrations/registry/factory.py`.

## 5. Why canonical owner cannot absorb

Runtime needs typed contract facades decoupled from catalog module graph; absorption would invert tier boundaries.

## 6. Convergence point

`intergrax/integrations/registry/factory.py` — `resolve_from_profile` for materialized integrations.

## 7. Negative bypass proof

`tests/unit/runtime/integrations/test_canonical_registry_projection.py` — catalog authority guards.
