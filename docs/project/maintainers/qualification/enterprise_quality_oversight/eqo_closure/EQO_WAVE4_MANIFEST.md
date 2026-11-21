# EQO wave 4 — parallel non-AUD execution manifest

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/` + ticket checkpoint line only.

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `bb4738c943f8aba2676125a1e9734fdd2e70a976` (post PR #692 waiver; lanes rebased ahead of commits).

## Parallel lanes

| Lane | Branch | Worktree | Deliverable |
|------|--------|----------|-------------|
| **G-W2** | `eqo/wave-4/w2` | `eqo-w4-w2` | W2-02..11 wave-4 read; `#680` track-only |
| **G-W4** | `eqo/wave-4/w4` | `eqo-w4-w4` | `W4-04_REQUALIFY_BATCH.md`, `W4-05_NONE_JUSTIFICATION.md` |
| **G-X** | `eqo/wave-4/x` | `eqo-w4-x` | X-01,02,03,05,07,08 read memos / refreshes |
| **G-MATRIX** | `eqo/wave-4/matrix` | `eqo-w4-matrix` | Partial CLS § in `EQO_MASTER_CLOSURE_MATRIX.md` (no AUD verdicts) |
| **G-INT** | `eqo/wave-4/integ` | `eqo-wv-w4-integ` | This manifest, README index, lane merges, PR → `development` |

**Out of scope:** pytest EXE, `EQO-C-04` rerun, E-02 apply, product/FRZ edits, AUD session E verdicts.

## #680 (track only)

No new ticket; gap documented in `EQO_W2_TRAIN_VERIFICATION.md` + `pr680_governance_fast.md`. C16 remains open.

## W4-02 queue

Tranches 01–07 **complete** (prior waves); wave-4 W4 lane executed W4-04 + W4-05 only.

## Merge ladder

```text
eqo/wave-4/w2      ──┐
eqo/wave-4/w4      ──┼──► eqo/wave-4/integ ──► development (PR)
eqo/wave-4/x       ──┤
eqo/wave-4/matrix  ──┘
```

## Wave-5 triggers

- AUD session E per `runs/wbNN/AUD_PACKET.md` (separate chat).
- `EQO-C-04` targeted replay under pytest lock (when operator schedules).
- W4-03 operator OD responses for OWNER_DECISION batch.