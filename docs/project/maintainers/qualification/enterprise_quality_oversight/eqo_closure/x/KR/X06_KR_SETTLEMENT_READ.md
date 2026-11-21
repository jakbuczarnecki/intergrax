# EQO-X-06 — KR-01..28 settlement read refresh

**Parent:** `EQO-X-06` · **Machine rows:** `kr_settlement.jsonl` · **Plan:** `EQO_TEST_RED_GREEN_SETTLEMENT_PLAN.md`

## Pins

| Pin | SHA |
|-----|-----|
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |
| KR register draft | `7960ae7f` |

## Mechanical count (wave-2 read)

Source: `kr_settlement.jsonl` @ HEAD (post wave-1 merge).

| `settlement_status` | Count |
|---------------------|------:|
| `BLOCKED` | 25 |
| `WAIT_MEP_NODE` | 3 |
| `SETTLED` | 0 |

All 28 KR rows remain non-settled for freeze purposes; no row promoted to green in this wave.

## Notable clusters (unchanged)

- **KR-08:** governance-fast / ACT-N11 cluster — tracks **#680** gap (`pr680_governance_fast.md`); monitor only, no waiver.
- **KR-03..04, KR-13..14:** `T-AF2-RW-GATES` owner nodes — product train scope, out of EQO write set.

## Interim disposition

`NOT_CLOSED` — X-06 acceptance (`EQO_TEST_RED_GREEN_SETTLEMENT_PLAN.md` §Acceptance) **not** met; matrix row C14 remains open until owner/MEP path or AUD wave-3.

**Audit:** session E pending.

**Recorded:** 2026-11-21## Wave-8 read pass (EQO-X-06)

**Land:** 569254840. Settlement counts unchanged (SETTLED 0 / 28); KR-08 still links #680 track-only memo (English refresh this wave).

**Recorded:** 2026-11-21 | lane eqo/wave-8/x | Ref: governance/tickets/T-IGX-046-F.md

## Wave-12 disposition read (EQO-X-06)

**Land:** `292ca3348`. KR settlement plan unchanged; **#680** remains track-only (C16).

**Recorded:** 2026-11-21 · lane `eqo/wave-12/x` · `Ref: governance/tickets/T-IGX-046-F.md`
