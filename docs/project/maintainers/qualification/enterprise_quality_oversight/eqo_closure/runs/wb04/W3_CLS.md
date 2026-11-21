# EQO-W3-WB04 — W3 CLS disposition (wave 5)

**Task ID:** `EQO-W3-WB04-CLS`
**Disposition:** PARTIAL — not EQO_CLOSED.
**AUD session E (aggregate):** `BLOCKED` (AUD-2, `runs/wb04/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb04/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb04/AUD_SESSION_E.md` | Independent session E verdict |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close note (CLS_PRE)

Optional-deps EXE re-run deferred (no pytest in this integrator). Terminal EQO-W3-WB04-CLS deferred.

## Aggregate rationale (session E)

`BLOCKED`: no hypothesis is contradicted by source, but none can be decided. The only test aimed at the MCP exposure path died at collection, and the other two findings are not covered by the slice. EXE exit 1 is an environment or collection problem, not a product verdict.

## Required follow-up routes

1. Re-run the slice with the `mcp` extra present. Record the cause of the `exit=2` and the exit code of each P5 target.
2. A2-047: run `test_ctrl_x_r2_security_baseline.py` at the pin, or get an operator decision on the `FRZ-CTL-01` deferral wording.
3. A5-015: produce an ENT-AUDIT FULL run record after T-IGX-041-M, or record a scoped acceptance.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
