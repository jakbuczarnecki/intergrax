# EQO-W3-WB04 — W3 CLS terminal disposition (wave 14)

**Task ID:** `EQO-W3-WB04-CLS`
**Disposition:** `DISPOSITIONED_NOT_CLOSED` — not PASS; campaign aggregate remains **0 PASS** @ pin.
**AUD session E (aggregate):** `BLOCKED` (AUD-2, `runs/wb04/AUD_SESSION_E.md`)
**Wave-13 / latest EXE:** exit `0` (13 passed in 79.41s) — `runs/wb04/pytest_wb04_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-6/exe-docs`); latest EXE @ pin (pre wave-14 CLS).
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb04/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb04/AUD_SESSION_E.md` | Independent session E verdict (**unchanged**) |
| `runs/wb04/pytest_wb04_exe.log` | Latest EXE @ pin |
| `runs/wb04/RUN_MANIFEST.json` | EXE custody record |
| `runs/wb04/W3_CLS.md` | Wave-5 partial; superseded by this wave-14 terminal band |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close composition (wave 14)

Terminal EQO-W3-WB04-CLS: compose session E + latest EXE @ pin. **Session E verdict not upgraded** by implementer.

## Aggregate rationale (session E + EXE)

Session E aggregate **`BLOCKED`** stands @ pin. Latest EXE log does **not** upgrade session E and does **not** assert PASS for this WB.

## Required follow-up routes

1. Re-run the slice with the `mcp` extra present. Record the cause of the `exit=2` and the exit code of each P5 target.
2. A2-047: run `test_ctrl_x_r2_security_baseline.py` at the pin, or get an operator decision on the `FRZ-CTL-01` deferral wording.
3. A5-015: produce an ENT-AUDIT FULL run record after T-IGX-041-M, or record a scoped acceptance.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Wave-14 compose cites pytest logs @ pin; implementer does not upgrade session E.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
- Wave-14 terminal CLS band: **no PASS**, **no** `EQO_CLOSED`.
