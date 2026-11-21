# EQO wave 5 — W3 CLS disposition manifest

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/runs/wbNN/W3_CLS.md` (11 lanes), this manifest, closure matrix § and README index row.

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `569254840f8f2b96b48063044572b29773ed31c0` (PR #698 post-AUD matrix + `CLS_PRE`).

## Parallel lanes

| Lane | Branch | Worktree | WB | AUD session E | Deliverable |
|------|--------|----------|-----|---------------|-------------|
| batch A | `eqo/wave-5/cls-wb02` | `eqo-w5-cls-02` | WB02 | `FAIL` | `runs/wb02/W3_CLS.md` |
| batch A | `eqo/wave-5/cls-wb03` | `eqo-w5-cls-03` | WB03 | `FAIL` | `runs/wb03/W3_CLS.md` |
| batch A | `eqo/wave-5/cls-wb05` | `eqo-w5-cls-05` | WB05 | `BLOCKED` | `runs/wb05/W3_CLS.md` |
| batch A | `eqo/wave-5/cls-wb12` | `eqo-w5-cls-12` | WB12 | `BLOCKED` | `runs/wb12/W3_CLS.md` |
| batch B | `eqo/wave-5/cls-wb01` | `eqo-w5-cls-01` | WB01 | `FAIL` | `runs/wb01/W3_CLS.md` |
| batch B | `eqo/wave-5/cls-wb04` | `eqo-w5-cls-04` | WB04 | `BLOCKED` | `runs/wb04/W3_CLS.md` |
| batch B | `eqo/wave-5/cls-wb06` | `eqo-w5-cls-06` | WB06 | `FAIL` | `runs/wb06/W3_CLS.md` |
| batch B | `eqo/wave-5/cls-wb07` | `eqo-w5-cls-07` | WB07 | `FAIL` | `runs/wb07/W3_CLS.md` |
| batch B | `eqo/wave-5/cls-wb09` | `eqo-w5-cls-09` | WB09 | `FAIL` | `runs/wb09/W3_CLS.md` |
| batch B | `eqo/wave-5/cls-wb10` | `eqo-w5-cls-10` | WB10 | `FAIL` | `runs/wb10/W3_CLS.md` |
| batch B | `eqo/wave-5/cls-wb11` | `eqo-w5-cls-11` | WB11 | `FAIL` | `runs/wb11/W3_CLS.md` |
| **G-INT** | `eqo/wave-5/integ` | `eqo-wv-w5-integ` | — | — | This manifest, matrix §, README index, lane merges, PR → `development` |

**Deferred this wave:** **WB08** — session E `BLOCKED` and no EXE log @ pin; `CLS_PRE.md` only (no `W3_CLS.md` lane).

**Out of scope:** product/FRZ edits, pytest EXE replay, terminal `EQO_CLOSED`, G6/IIAC independence claims.

## W3 CLS summary (@ TRAIN_PIN)

| WB | AUD E | `W3_CLS.md` disposition |
|----|-------|-------------------------|
| WB01 | FAIL | `PARTIAL` — not EQO_CLOSED |
| WB02 | FAIL | `PARTIAL` |
| WB03 | FAIL | `PARTIAL` |
| WB04 | BLOCKED | `PARTIAL` |
| WB05 | BLOCKED | `PARTIAL` |
| WB06 | FAIL | `PARTIAL` |
| WB07 | FAIL | `PARTIAL` |
| WB08 | BLOCKED | *(deferred — no wave-5 lane)* |
| WB09 | FAIL | `PARTIAL` |
| WB10 | FAIL | `PARTIAL` |
| WB11 | FAIL | `PARTIAL` |
| WB12 | BLOCKED | `PARTIAL` |

**Published aggregate (11 lanes):** 0 `PASS`, 7 `FAIL`, 4 `BLOCKED` in `W3_CLS.md` text; campaign aggregate @ pin remains 8 `FAIL` / 4 `BLOCKED` including WB08.

## Merge ladder

```text
eqo/wave-5/cls-wb02 ─┐
eqo/wave-5/cls-wb03 ─┤
… (11 cls lanes) ────┼──► eqo/wave-5/integ ──► development (PR)
```

## Wave-6 triggers

- WB08 `W3_CLS` when EXE evidence exists or operator schedules DES scope replay.
- Targeted pytest EXE reruns for OPEN packets (waves 6+).
- Campaign C / W4 operator batches per prior manifests.