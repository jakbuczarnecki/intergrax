# EQO-C-04 — formal blocked record (wave-11)

**Task ID:** `EQO-C-04` · **Status:** `BLOCKED`  
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Disposition:** `BLOCKED` — targeted replay **attempted**; non-zero exit retained; no re-run in wave-11.

## Custody chain

| Wave | Artefact | Role |
|------|----------|------|
| 2 | `C04_TARGETED_REPLAY_PREP.md` | Prep under pytest lock |
| 3 | `c04_targeted_replay.log`, `C04_RUN_MANIFEST.json` | Single execute @ TRAIN_PIN |
| 7 | `C04_STATUS.md` | Log-derived summary refresh |
| 11 | This file | Formal blocked record for C-CLS gate |

## Pytest outcome (authoritative log)

| Field | Value |
|-------|-------|
| Exit code | `1` |
| Failed | `47` |
| Passed | `395` |
| Duration | `70.90s` |
| Source | `eqo_closure/runs/c/c04_targeted_replay.log` |

## Blocker classification

| Band | Interpretation |
|------|----------------|
| Product / train drift @ TRAIN_PIN | Dominates failure surface (TRACE-X, MCP P5, broad `tests/unit/docs/` contracts) |
| Slice width | Full targeted replay is wider than HIST three-row dispute alone |
| EQO scope | No product fix, no log redaction, no pytest re-execution in wave-11 |

## Downstream impact

- **C-01 / C-02:** disputed rows and manifest join remain open until a narrower replay or owner tooling succeeds.
- **C-CLS:** `NOT_MET` for campaign terminal close — see `c/C_CLS.md`.
- **Matrix:** master row C04 stays `PARTIALLY_MET` (replay attempted, exit 1).

## Unblock routes (documentation only)

1. Operator narrows pytest surface to HIST / layer slice aligned with C-03 `TARGETED_REPLAY` rows.  
2. Train or product delta cleared on a **new pin**, then one replay under shared lock.  
3. Steward records per-row `FULL_REQUAL_REQUIRED` exact blocker if replay is abandoned.

**Ref:** `governance/tickets/T-IGX-046-F.md`