# EQO wave 14 — W3 terminal CLS + matrix manifest

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `runs/wb01`–`wb12/W3_CLS.md`, `COR_DRV.md` DRV sections, matrix C01–C25, this manifest, README row 40, ticket checkpoint.

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `9a673b1b2` (PR #711 post wave-13).

## Parallel lanes

| Lane | Branch | Worktree | WB / scope | AUD E | Deliverable | Commit |
|------|--------|----------|------------|-------|-------------|--------|
| batch | `eqo/wave-14/cls-wb01` | `eqo-w14-cls-01` | WB01 | `FAIL` | `runs/wb01/W3_CLS.md` | `a682e289d` |
| batch | `eqo/wave-14/cls-wb02` | `eqo-w14-cls-02` | WB02 | `FAIL` | `runs/wb02/W3_CLS.md` | `d6b115ee4` |
| batch | `eqo/wave-14/cls-wb03` | `eqo-w14-cls-03` | WB03 | `FAIL` | `runs/wb03/W3_CLS.md` | `1595ba7e5` |
| batch | `eqo/wave-14/cls-wb04` | `eqo-w14-cls-04` | WB04 | `BLOCKED` | `runs/wb04/W3_CLS.md` | `8d7d1d832` |
| batch | `eqo/wave-14/cls-wb05` | `eqo-w14-cls-05` | WB05 | `BLOCKED` | `runs/wb05/W3_CLS.md` | `989228046` |
| batch | `eqo/wave-14/cls-wb06` | `eqo-w14-cls-06` | WB06 | `FAIL` | `runs/wb06/W3_CLS.md` | `c6d3d689d` |
| batch | `eqo/wave-14/cls-wb07` | `eqo-w14-cls-07` | WB07 | `FAIL` | `runs/wb07/W3_CLS.md` | `eea9146db` |
| batch | `eqo/wave-14/cls-wb08` | `eqo-w14-cls-08` | WB08 | `BLOCKED` | `runs/wb08/W3_CLS.md` | `3e4d1ff7f` |
| batch | `eqo/wave-14/cls-wb09` | `eqo-w14-cls-09` | WB09 | `FAIL` | `runs/wb09/W3_CLS.md` | `4d9941a49` |
| batch | `eqo/wave-14/cls-wb10` | `eqo-w14-cls-10` | WB10 | `FAIL` | `runs/wb10/W3_CLS.md` | `4b6e8e2d3` |
| batch | `eqo/wave-14/cls-wb11` | `eqo-w14-cls-11` | WB11 | `FAIL` | `runs/wb11/W3_CLS.md` | `2049cd286` |
| batch | `eqo/wave-14/cls-wb12` | `eqo-w14-cls-12` | WB12 | `BLOCKED` | `runs/wb12/W3_CLS.md` | `190ca2a21` |
| **G-DRV** | `eqo/wave-14/drv` | `eqo-w14-drv` | WB01–12 | — | `COR_DRV.md` DRV deferrals | `5dae8e435` |
| **G-W9** | `eqo/wave-14/w9` | `eqo-w14-w9` | C01–C25 | — | matrix sweep | `2a76385f3` |
| **G-INT** | `eqo/wave-14/integ` | `eqo-w14-integ` | — | — | This manifest, README, ticket, PR → `development` | _(integrator tip)_ |

**Out of scope:** `EQO_CLOSED`, session E verdict changes, pytest, product edits, W9-02 terminal.

## W3 CLS summary (@ TRAIN_PIN, wave 14)

| WB | AUD E | Terminal `W3_CLS.md` |
|----|-------|----------------------|
| WB01 | FAIL | `DISPOSITIONED_NOT_CLOSED` |
| WB02 | FAIL | `DISPOSITIONED_NOT_CLOSED` |
| WB03 | FAIL | `DISPOSITIONED_NOT_CLOSED` |
| WB04 | BLOCKED | `DISPOSITIONED_NOT_CLOSED` |
| WB05 | BLOCKED | `DISPOSITIONED_NOT_CLOSED` |
| WB06 | FAIL | `DISPOSITIONED_NOT_CLOSED` |
| WB07 | FAIL | `DISPOSITIONED_NOT_CLOSED` |
| WB08 | BLOCKED | `DISPOSITIONED_NOT_CLOSED` (first terminal CLS lane) |
| WB09 | FAIL | `DISPOSITIONED_NOT_CLOSED` |
| WB10 | FAIL | `DISPOSITIONED_NOT_CLOSED` |
| WB11 | FAIL | `DISPOSITIONED_NOT_CLOSED` |
| WB12 | BLOCKED | `DISPOSITIONED_NOT_CLOSED` |

**Published aggregate:** **0 PASS** @ pin (12 lanes `DISPOSITIONED_NOT_CLOSED`).

## Merge ladder

```text
eqo/wave-14/cls-wb* (×12) ──► eqo/wave-14/drv ──► eqo/wave-14/w9 ──► eqo/wave-14/integ ──► development
```

**Docs gates:** `ticket-governance` + `governance-policy-pin` per T-IGX-046-F.

All qualification commits: trailer `Ref: governance/tickets/T-IGX-046-F.md`.
