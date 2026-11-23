# EQO B-1 — Critical invariants (Pre-FCS)

**Parent:** `EQO-W3` · **Pin:** `TRAIN_PIN` (`28c27cb`) · **Band:** C (isolated worktree `eqo-w3`).

## Executable contract

1. For each `WB01`–`WB12`, run eight tasks (`DES` → `EXE` → `EVD` → `DIS` → `AUD` → optional `COR`/`DRV` → `CLS`). Template IDs: `EQO-W3-WBnn-TYP`.
2. **Acceptance (campaign):** all `EQO-W3-WB*-CLS` complete **or** exact blockers published; plus `EQO-X-01`, `EQO-X-02`, `EQO-X-03` disposition (plan §4.4).
3. **Write set:** `eqo_closure/`, `eqo_closure/runs/<wb>/`. Product tree untouched (band C).
4. **Matrices:** `EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl`, `EQO_B1_PRODUCTION_REACHABILITY_MATRIX.jsonl`.

## Workstream → finding ownership (51 exclusive)

| WB | Domain | Finding IDs | Ticket refs |
|----|--------|-------------|-------------|
| WB01 | Governance, execution authority | A1-021, A4-015, A4-044, A5-003, A7-030 | T-IGX-043-A, T-IGX-044-F |
| WB02 | Tenant isolation | A4-006, A4-027, A7-009, A7-028, A7-031, A8-006 | — (OD-EQO-08 if no write set) |
| WB03 | Fail-closed, HITL | A4-026, A4-031, A4-037, A3-027 | T-IGX-043-A, T-IGX-043-B, T-IGX-043-E |
| WB04 | Security | A2-047, A7-013, A5-015 | T-IGX-051-B |
| WB05 | Architecture boundaries | A2-009, A4-007, A4-020, A3-011, A7-021 | T-IGX-043-G, T-IGX-043-H, T-IGX-051-I |
| WB06 | Configuration | A4-025, A4-042, A5-002 | T-IGX-043-H |
| WB07 | Recovery, durable execution | A2-043, A2-049, A4-034, A4-045 | T-IGX-051-E |
| WB08 | Context / memory (cross-domain) | — (obligation from 043-A/B) | T-IGX-043-A, T-IGX-043-B |
| WB09 | MCP, tools, skills | A2-027, A4-040 | T-IGX-043-D, T-IGX-051-D |
| WB10 | Trace, evidence | A2-019, A2-030, A5-012 | — |
| WB11 | Assurance engine | A6-007, A7-037, A1-008, A1-012, A1-015, A2-023, A2-024, A2-051, A7-003, A7-007 | T-IGX-043-I |
| WB12 | Cross-domain | A3-019, A4-039, A4-047, A5-010, A8-020, A2-031 | — |

**Count check:** 5+6+4+3+5+3+4+0+2+3+10+6 = **51**.

## Per-workstream execution checklist

| Step | Task pattern | Done when |
|------|----------------|-----------|
| 1 | `*-DES` | SHA256 prereg + hypotheses on `TRAIN_PIN` |
| 2 | `*-EXE` | Band-C run log; no mock of mechanism under test |
| 3 | `*-EVD` | Manifest in `eqo_closure/runs/` |
| 4 | `*-DIS` | Each owned finding has taxonomy status |
| 5 | `*-AUD` | Session E verdict on same SHA |
| 6 | `*-COR` | `CL-01` if defect; else `NOT_APPLICABLE_WITH_EVIDENCE` |
| 7 | `*-DRV` | `RV-01` if pin moved |
| 8 | `*-CLS` | DIS+AUD complete or exact blocker ID |

## Finding table (Campaign B scope)

| Finding ID | WB | Severity (dossier) | Primary probe row |
|------------|----|--------------------|-------------------|
| EQOA-A1-021 | WB11 | MEDIUM | `EQO_B1_ADVERSARIAL_TEST_MATRIX.jsonl` |
| EQOA-A4-015 | WB01 | MEDIUM | same |
| EQOA-A4-044 | WB01 | MEDIUM | same |
| EQOA-A5-003 | WB01 | LOW | same |
| EQOA-A7-030 | WB01 | MEDIUM | same |
| EQOA-A4-006 | WB02 | MEDIUM | same |
| EQOA-A4-027 | WB02 | MEDIUM | same |
| EQOA-A7-009 | WB02 | MEDIUM | same |
| EQOA-A7-028 | WB02 | MEDIUM | same |
| EQOA-A7-031 | WB02 | LOW | same |
| EQOA-A8-006 | WB02 | LOW | same |
| EQOA-A4-026 | WB03 | MEDIUM | same |
| EQOA-A4-031 | WB03 | LOW | same |
| EQOA-A4-037 | WB03 | MEDIUM | same |
| EQOA-A3-027 | WB03 | MEDIUM | same |
| EQOA-A2-047 | WB04 | MEDIUM | same |
| EQOA-A7-013 | WB04 | MEDIUM | same |
| EQOA-A5-015 | WB04 | LOW | same |
| EQOA-A2-009 | WB05 | MEDIUM | same |
| EQOA-A4-007 | WB05 | LOW | same |
| EQOA-A4-020 | WB05 | MEDIUM | same |
| EQOA-A3-011 | WB05 | LOW | same |
| EQOA-A7-021 | WB05 | MEDIUM | same |
| EQOA-A4-025 | WB06 | MEDIUM | same |
| EQOA-A4-042 | WB06 | LOW | same |
| EQOA-A5-002 | WB06 | MEDIUM | same |
| EQOA-A2-043 | WB07 | MEDIUM | same |
| EQOA-A2-049 | WB07 | MEDIUM | same |
| EQOA-A4-034 | WB07 | LOW | same |
| EQOA-A4-045 | WB07 | LOW | same |
| EQOA-A2-027 | WB09 | MEDIUM | same |
| EQOA-A4-040 | WB09 | MEDIUM | same |
| EQOA-A2-019 | WB10 | MEDIUM | same |
| EQOA-A2-030 | WB10 | MEDIUM | same |
| EQOA-A5-012 | WB10 | LOW | same |
| EQOA-A6-007 | WB11 | MEDIUM | same |
| EQOA-A7-037 | WB11 | MEDIUM | same |
| EQOA-A1-008 | WB11 | LOW | same |
| EQOA-A1-012 | WB11 | LOW | same |
| EQOA-A1-015 | WB11 | LOW | same |
| EQOA-A2-023 | WB11 | MEDIUM | same |
| EQOA-A2-024 | WB11 | MEDIUM | same |
| EQOA-A2-051 | WB11 | MEDIUM | same |
| EQOA-A7-003 | WB11 | MEDIUM | same |
| EQOA-A7-007 | WB11 | MEDIUM | same |
| EQOA-A3-019 | WB12 | MEDIUM | same |
| EQOA-A4-039 | WB12 | LOW | same |
| EQOA-A4-047 | WB12 | MEDIUM | same |
| EQOA-A5-010 | WB12 | MEDIUM | same |
| EQOA-A8-020 | WB12 | MEDIUM | same |
| EQOA-A2-031 | WB12 | MEDIUM | same |

## Merge dependency (#680)

Findings A4-044, A4-037, A4-015: probe **before** merge allowed on `TRAIN_PIN`; after merge → `EQO-W3-WB01-DRV` / `WB03-DRV`, not a new campaign.

## Closure criterion

`EQO-W3` campaign B-1 closes when every row above has `DIS` status ≠ `NOT_STARTED` and `CLS` tasks exist with evidence refs or exact blockers in the closure matrix.