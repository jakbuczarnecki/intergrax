# EQO-X-04 — ROADMAP-REPLAY-X counter read (both pins)

**Finding:** EQOA-A5-007 (HIGH, `REQUALIFY`)  
**Owner (disposition):** ROADMAP-REPLAY-X / PDC-X  
**Task:** pre-FCS read-only; counter zeroing remains **PT-Q05** post-FCS.

## Pins (`EQO_PIN_BIND.json`)

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## Method

Read-only `all_counters()` from `tests/qualification/roadmap_replay_x/_closure_counters.py` (no pytest EXE).

- TRAIN: worktree `eqo-w3` @ `28c27cb`
- DOC: worktree `_wt/intergrax/_scratch/s3-docpin` @ `8cee3c5`

Command: `uv run python -c "… all_counters() …"` (wave-1/x session, 2026-11-21).

## Results (2026-11-21, refreshed)

| Counter | DOC_PIN | TRAIN_PIN |
|---------|--------:|----------:|
| `historical_evidence_invalidated_by_later_changes` | **2** | **3** |
| `reopened_historical_stages` | 0 | 0 |
| `unresolved_replay_blockers` | 0 | 0 |
| `roadmap_drift` | 0 | 0 |
| `unclassified_replay_findings` | 0 | 0 |
| `vendor_neutrality_replay_failures` | 1 | 1 |

**Interpretation:** Both pins remain **non-zero** on own-invalidation and vendor-neutrality counters; pins **differ** on invalidation count (2 vs 3). Consistent with open A5-007 / PT-Q05 ownership. EQO records the read only.

## Disposition (interim)

`NOT_CLOSED` — evidence read captured; counter remediation owned by ARCH-FREEZE **PT-Q05** after FCS.

**Audit:** session E pending.

## Wave-4 refresh

`ROADMAP-REPLAY-X` layer remains `NO_PRODUCT_DELTA` on both pins; counter read idempotent @ wave-4 pins.

**Recorded:** 2026-11-21 · lane `eqo/wave-4/x`