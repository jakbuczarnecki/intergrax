# EQO wave 9 — W3 COR/DRV disposition manifest

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/runs/wbNN/COR_DRV.md` (12 lanes), this manifest, README index row.

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `2b5c3aec6` (post PR #705 wave-11 integ).

## Parallel lanes

| Lane | Branch | Worktree | WB | AUD session E | Deliverable | Commit |
|------|--------|----------|-----|---------------|-------------|--------|
| batch A | `eqo/wave-9/cor-wb01` | `eqo-w9-cor-wb01` | WB01 | `FAIL` | `runs/wb01/COR_DRV.md` | `48b1968ea` |
| batch A | `eqo/wave-9/cor-wb02` | `eqo-w9-cor-wb02` | WB02 | `FAIL` | `runs/wb02/COR_DRV.md` | `94e241bc2` |
| batch A | `eqo/wave-9/cor-wb03` | `eqo-w9-cor-wb03` | WB03 | `FAIL` | `runs/wb03/COR_DRV.md` | `0d823d3a2` |
| batch A | `eqo/wave-9/cor-wb04` | `eqo-w9-cor-wb04` | WB04 | `BLOCKED` | `runs/wb04/COR_DRV.md` | `5b436d21e` |
| batch B | `eqo/wave-9/cor-wb05` | `eqo-w9-cor-wb05` | WB05 | `BLOCKED` | `runs/wb05/COR_DRV.md` | `8ab8ec8ac` |
| batch B | `eqo/wave-9/cor-wb06` | `eqo-w9-cor-wb06` | WB06 | `FAIL` | `runs/wb06/COR_DRV.md` | `47a00e588` |
| batch B | `eqo/wave-9/cor-wb07` | `eqo-w9-cor-wb07` | WB07 | `FAIL` | `runs/wb07/COR_DRV.md` | `6f22fe631` |
| batch B | `eqo/wave-9/cor-wb08` | `eqo-w9-cor-wb08` | WB08 | `BLOCKED` | `runs/wb08/COR_DRV.md` | `b1f5bc234` |
| batch B | `eqo/wave-9/cor-wb09` | `eqo-w9-cor-wb09` | WB09 | `FAIL` | `runs/wb09/COR_DRV.md` | `d057a3715` |
| batch B | `eqo/wave-9/cor-wb10` | `eqo-w9-cor-wb10` | WB10 | `FAIL` | `runs/wb10/COR_DRV.md` | `4f45158b8` |
| batch B | `eqo/wave-9/cor-wb11` | `eqo-w9-cor-wb11` | WB11 | `FAIL` | `runs/wb11/COR_DRV.md` | `fe3ce9fd9` |
| batch B | `eqo/wave-9/cor-wb12` | `eqo-w9-cor-wb12` | WB12 | `BLOCKED` | `runs/wb12/COR_DRV.md` | `d7b89341b` |
| **G-INT** | `eqo/wave-9/integ` | `eqo-w9-integ` | — | — | This manifest, README index, lane merges, PR → `development` | _(integrator tip)_ |

**Out of scope:** product/FRZ edits, pytest EXE replay, AUD verdict changes, terminal `EQO_CLOSED`, G6/IIAC independence claims.

## COR summary (@ TRAIN_PIN)

| WB | AUD E | COR outcome |
|----|-------|-------------|
| WB01 | FAIL | `NOT_APPLICABLE_WITH_EVIDENCE` — docs disposition paths only |
| WB02 | FAIL | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB03 | FAIL | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB04 | BLOCKED | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB05 | BLOCKED | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB06 | FAIL | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB07 | FAIL | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB08 | BLOCKED | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB09 | FAIL | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB10 | FAIL | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB11 | FAIL | `NOT_APPLICABLE_WITH_EVIDENCE` |
| WB12 | BLOCKED | `NOT_APPLICABLE_WITH_EVIDENCE` |

Product CL-01 / pytest DRV replays deferred to wave 10+ per lane `COR_DRV.md` DRV sections.

## Merge ladder

```text
eqo/wave-9/cor-wb01 … cor-wb12 (12 lanes) ──► eqo/wave-9/integ ──► development (PR; docs gates)
```

**Docs gates (integrator PR):** `ticket-governance` + `governance-policy-pin` per ticket merge policy.

## Follow-on triggers

- Wave 10 EXE reruns for selected W3 slices @ TRAIN_PIN.
- #680 / train pin advance → per-lane `DRV` tasks (`RV-01`); no new EQO campaign.

All qualification commits: trailer `Ref: governance/tickets/T-IGX-046-F.md`.