# EQO wave 3 — AUD packet integration manifest

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/` + ticket checkpoint.

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `8631b1e8459e096b74d1c80247d3e432b44fff30` (PR #690).

## Lane

| Branch | Worktree | Deliverable |
|--------|----------|-------------|
| `eqo/wave-3/integ` | `eqo-wv-w3-integ` | AUD_PACKET (WB01–12), W4-03 batch draft, C-04 replay log, matrix AUD § |

**Out of scope:** E-02 apply, session E verdicts (PASS/FAIL), product fixes.

## AUD packet summary

| WB | Packet status | EXE exit |
|----|---------------|---------:|
| WB01 | OPEN | 1 |
| WB02 | PRE_AUD | 0 |
| WB03 | PRE_AUD | 0 |
| WB04 | OPEN | 1 |
| WB05 | PRE_AUD | 0 |
| WB06 | OPEN | 4 |
| WB07 | OPEN | 1 |
| WB08 | OPEN | — |
| WB09 | OPEN | 1 |
| WB10 | OPEN | 1 |
| WB11 | OPEN | 1 |
| WB12 | PRE_AUD | 0 |

## EQO-C-04

| Item | Result |
|------|--------|
| Lock | acquired + released |
| Run | `BLOCKED` (pytest exit 1) |

## Merge ladder

```text
eqo/wave-3/integ ──► development (PR)
```

## Wave-4 triggers

- Independent session E per `AUD_PACKET.md` (separate chat; implementer must not self-PASS).
- C-04 re-run after train/product or narrower slice.
- W4-03 operator OD packet.