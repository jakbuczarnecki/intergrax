# EQO wave 2 — parallel execution manifest

**Generated:** 2026-11-21 (autonomous executor)  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/` only.

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `84a7a76ece64b3641da03a567e7893571e87aa2b` (post PR #689).

## Parallel lanes (wave-2)

| Lane | Branch | Worktree | Deliverable |
|------|--------|----------|-------------|
| **G-W4** | `eqo/wave-2/w4` | `eqo-par-w2-w4` | `W4-02_TRANCHE_05` … `_07` (W4-02 queue complete) |
| **G-X** | `eqo/wave-2/x` | `eqo-wv-w2-x` | X-01, X-06, X-08 read memos |
| **G-DIS/AUD-prep** | `eqo/wave-2/dis` | `eqo-wv-w2-dis` | `AUD_PREP_CHECKLIST.md` per WB + `runs/c/C04_TARGETED_REPLAY_PREP.md` |
| **G-INT** | `eqo/wave-2/integ` | `eqo-wv-w2-integ` | This manifest, README phase index |

**Out of scope this wave:** E-02 apply on `register.jsonl`, full AUD session E, matrix CLS, product/FRZ edits.

## #680 (track only)

PR #680 **merged** on train path; governance-fast classification documented in `pr680_governance_fast.md` (regresja IIAC/ACT-N11). **Does not block** wave-2 doc band. KR-08 cluster monitored; no waiver asserted.

## W3 EXE status @ TRAIN_PIN (carried from wave-1)

| WB | pytest exit | AUD prep |
|----|------------:|----------|
| WB01 | 1 | yes |
| WB02 | 0 | yes |
| WB03 | 0 | yes |
| WB04 | 1 | yes |
| WB05 | 0 | yes |
| WB06 | 4 | yes |
| WB07 | 1 | yes |
| WB09 | 1 | yes |
| WB10 | 1 | yes |
| WB11 | 1 | yes |
| WB12 | 0 | yes |

Non-zero exits retained as **BLOCKED** manifests; lanes not blocked.

## Optional checks

| Item | Result |
|------|--------|
| WB06 `uv sync` @ `eqo-w3` TRAIN_PIN | OK (~0.14s) |
| EQO-C-04 targeted replay | `DEFERRED` (prep doc only) |

## Merge ladder

```text
eqo/wave-2/w4  ──┐
eqo/wave-2/x   ──┼──► eqo/wave-2/integ ──► development (PR)
eqo/wave-2/dis ──┘
```

## Wave-3 triggers

- Session **E** AUD per `runs/wbNN/AUD_PREP_CHECKLIST.md`
- `EQO-C-04` targeted replay under pytest lock
- `EQO-W2-02` gate replay for A4-003 @ TRAIN_PIN
- E-02 steward apply (operator ticket choice)
- W4-03 OWNER_DECISION batch (X-08 five IDs + OD packet)