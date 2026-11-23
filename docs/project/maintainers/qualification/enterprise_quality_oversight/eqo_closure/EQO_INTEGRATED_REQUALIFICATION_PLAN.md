# EQO — Integrated requalification (RV + W7 + C-05)

**Parents:** `EQO-RV-01`, `EQO-W7`, `EQO-C-05` · **Does not replace** ARCH-FREEZE PT-Q05..Q13.

## Triggers

| Event | Action |
|-------|--------|
| Train or DOC pin moves | `EQO-RV-01` — diff, semantic impact, task risk class |
| FCS merge exists | `EQO-W7-01`..`03` — validate FCS, replay invalidated evidence only |
| MEP runs PT-Q05..Q10 | `EQO-C-05` — bind SHA to `FULL_REQUAL_REQUIRED` rows |

## RV-01 procedure

1. `git diff old_pin..new_pin`
2. Semantic impact (EVOL-X / changed-paths, not file count alone)
3. Tasks whose read set intersects diff → `UNAFFECTED` / `DOC_ONLY` / `TARGETED` / `FULL_STREAM`
4. Re-run **TARGETED** and **FULL_STREAM** only
5. Append `EQO_SHA_REVALIDATION.jsonl` (one edge per affected task)

Full W0 repeat only on topology change (merge-base, FCS/G2/M3 appearance, train rewrite).

## W7 post-FCS replay

| Task | Purpose |
|------|---------|
| W7-01 | Confirm FCS commit: parents G2 + development; `git diff G2 FCS` has no product tree |
| W7-02 | Impact of 12 DOC-only commits on development absent from train + merge commit |
| W7-03 | Evidence recon: `STILL_VALID` / `REPLAYED` / `INVALIDATED` |
| W7-04 | Cite PT-Q02, Q03, Q05..Q13, lock tasks — **observe only** |

**Key rule:** If W3 ran only on `28c27cb` and G2 is later, replay `28c27cb..G2`, not entire W3.

## C-05 integration matrix

For each `FULL_REQUAL_REQUIRED` row in C-03:

| Layer | ARCH-FREEZE node | Evidence field |
|-------|------------------|----------------|
| ROADMAP-REPLAY-X | PT-Q05 | `pt_q05_sha` |
| STATE-X / TRACE-X | PT-Q06 | `pt_q06_sha` |
| TENANT-X | PT-Q07 | `pt_q07_sha` |
| TOOL-LIFE-X | PT-Q08 | `pt_q08_sha` |
| DIST-SCALE-X | PT-Q09 | `pt_q09_sha` |
| (others) | PT-Q10..Q13 as mapped | TBD in C-03 row |

Missing node execution → exact blocker; EQO does not run those PT jobs.

## X-02 L499 linkage

A2-001, A2-002: DOC_PIN ledger read now; closure waits PT-Q08/Q09 (`WAIT_FOR_INTEGRATION`).

## Acceptance

Integrated requalification band met when: RV registry current for active pin; W7-03 complete after FCS; C-05 has no orphan `FULL_REQUAL_REQUIRED` without SHA or blocker; C09 matrix reflects state.