# DUP-X — Responsibility Delta: harness registry projection

| Field | Value |
| --- | --- |
| **Symbols** | `HarnessHostRegistryAuthority`, `registry_projection.py` |
| **Candidate** | DUPX-CAND-04 |
| **Disposition** | **READ-ONLY PROJECTION** — not catalog truth |

## 1. Exact responsibility

Project platform agent/tool registry views into hosted application harness assembly without mutating platform catalog state.

## 2. Authority / truth owned

**None** for registration mutations; platform `AgentRegistry` / tool registry remain truth.

## 3. I/O contracts + lifecycle

Read APIs only at assembly time; no parallel registration side effects.

## 4. Caller graph

`intergrax/applications/_shared/*_wiring.py`, host factory paths.

## 5. Why canonical owner cannot absorb

Tier-3 assembly boundary requires localized projection types without importing application modules into platform registry core.

## 6. Convergence point

Platform registry modules under `intergrax/` wired via sanctioned composition roots (TRACE-X).

## 7. Negative bypass proof

Architecture gates on tier boundaries + EBH-2* harness typing seams (read-only registry interfaces).
