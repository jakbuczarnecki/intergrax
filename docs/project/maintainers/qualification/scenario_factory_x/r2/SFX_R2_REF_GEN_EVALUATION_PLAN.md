# SFX R2 — REF / GEN Evaluation and Performance Plan (PROPOSED)

**GEN seal:** Do **not** read sealed content. Use [`SFX_GEN_SET_SEAL.md`](../SFX_GEN_SET_SEAL.md) digest only until independent reveal (SFX-P12-04).

## REF-1 (invoice automation) — priority vertical slice

### Business oracles (independent)

| Oracle ID | Assertion | Failure class |
| --- | --- | --- |
| REF1-O-01 | Valid invoice → PO created with correct totals | BUSINESS_FAIL |
| REF1-O-02 | Duplicate invoice → no second PO | IDEMPOTENCY |
| REF1-O-03 | Missing PO reference → explicit HITL/escalation, no silent approve | SAFETY |
| REF1-O-04 | Malformed input → reject without side effects | SECURITY |

### Autonomy effectiveness (R2)

| Metric | Source | Pass threshold |
| --- | --- | --- |
| AUTO-02 time-to-first business-logic candidate | mission events | PROPOSED — OD-SFX-07 + AUTO targets |
| AUTO-04 iteration/escalation counts | mission + SEF refs | recorded, not fabricated |
| AUTO-06 champion delta | compare view | non-regression security suite |

### Modes under test

1. `DETERMINISTIC_SCAFFOLD` baseline (R1 requirement).
2. `ASSISTED` + human HO-2 submits.
3. `BOUNDED_AUTONOMOUS` only after grant OD — **NOT_RUN** until authorized.

### Evidence layout

`.tmp/proof/sfx/ref-1/<run-id>/` — oracles, verdicts, costs, SHA, grant_ref (if any).

## REF-2 / REF-3

Reuse R1 §12 acceptance; add R2 oracles for API idempotency (REF-2) and citation grounding (REF-3). Subject to OD-SFX-08.

## GEN set (six sealed)

| Rule | Action |
| --- | --- |
| Pre-reveal | Zero tuning against GEN content |
| Reveal authority | Independent session SFX-P12-04 only |
| Post-change | MS-6 full rerun REF-1..3 + GEN @ new SHA |

## Performance plan

Record G-1..G-5, M-1..M-9, AUTO-01..12 per [`design/SFX_METRICS.md`](../design/SFX_METRICS.md) R2 appendix. No SLA claims without measured artifacts.