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

Read-only `compute_historical_evidence_invalidated()` and sibling counters from  
`tests/qualification/roadmap_replay_x/_closure_counters.py` (no pytest EXE in this pass).

- TRAIN: worktree `eqo-par-s3` @ `28c27cb`
- DOC: detached scratch worktree `@ 8cee3c5` (no tracked edits)

## Results (2026-11-21)

| Counter | DOC_PIN | TRAIN_PIN |
|---------|--------:|----------:|
| `historical_evidence_invalidated_by_later_changes` | **2** | **2** |
| `reopened_historical_stages` | 0 | 0 |
| `unresolved_replay_blockers` | 0 | 0 |
| `roadmap_drift` | 0 | 0 |
| `unclassified_replay_findings` | 0 | 0 |
| `vendor_neutrality_replay_failures` | 1 | 1 |

**Interpretation:** Both pins agree — the own-invalidation counter is **non-zero** (2), consistent with dossier A5-007. EQO records the read; it does **not** claim closure or PT-Q05 completion.

## Disposition (interim)

`NOT_CLOSED` — evidence read captured; full requalification and counter remediation owned by ARCH-FREEZE **PT-Q05** after FCS.

**Audit:** session E pending.