# EQO-X-03 — IIAC A0-007 / RB-2 block read

**Task:** `EQO-X-03` · **Finding:** EQOA-A0-007 (HIGH)  
**Acceptance:** `BLOCKED until RB-2` with preserved artefacts (WB01 pytest, #680 track).

## Pins

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## Read-only evidence chain

| Artefact | Observation @ TRAIN_PIN |
|----------|-------------------------|
| `runs/wb01/GOVERNANCE_EVIDENCE.md` | `test_iiac_activation_gate.py` — IIAC policy **FAIL** |
| `EQO_C_DRIFT.jsonl` layer `IIAC` | `drift_class=FULL_REQUAL_REQUIRED` |
| `pr680_governance_fast.md` | Stale gate expectation class (W0-03); #680 not on pin |
| `EQO-W2-01` | T-IGX-043-A open — track-only |

## Interim disposition

`BLOCKED` — IIAC **not** `IN_FORCE` on train tree; downgrade requires operator RB-2 / band G (per `EQO_GOVERNANCE_ASSURANCE_COMPLETION_PLAN.md`), not EQO implementer assertion.

**Audit:** session E pending.

**Recorded:** 2026-11-21 · lane `eqo/wave-4/x` · `Ref: governance/tickets/T-IGX-046-F.md`## Wave-8 read pass (EQO-X-03)

**Land:** 569254840. English pr680_governance_fast.md refresh; IIAC not IN_FORCE on train unchanged. #680 remains track-only (C16).

**Recorded:** 2026-11-21 | lane eqo/wave-8/x

## Wave-12 disposition read (EQO-X-03)

**Land:** `292ca3348`. IIAC **not** `IN_FORCE` on train; RB-2 / FCS preconditions in `EQO_W7_W8_DEFERRED_BLOCKERS.md` unchanged.

**Recorded:** 2026-11-21 · lane `eqo/wave-12/x` · `Ref: governance/tickets/T-IGX-046-F.md`
