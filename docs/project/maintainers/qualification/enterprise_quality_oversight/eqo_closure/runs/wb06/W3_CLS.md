# EQO-W3-WB06 — W3 CLS terminal disposition (wave 14)

**Task ID:** `EQO-W3-WB06-CLS`
**Disposition:** `DISPOSITIONED_NOT_CLOSED` — not PASS; campaign aggregate remains **0 PASS** @ pin.
**AUD session E (aggregate):** `FAIL` (AUD-2, `runs/wb06/AUD_SESSION_E.md`)
**Wave-13 / latest EXE:** exit `0` (32 passed in 188.25s (0:03:08)) — `runs/wb06/pytest_wb06_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-10/exe`); latest EXE @ pin (pre wave-14 CLS).
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb06/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb06/AUD_SESSION_E.md` | Independent session E verdict (**unchanged**) |
| `runs/wb06/pytest_wb06_exe.log` | Latest EXE @ pin |
| `runs/wb06/RUN_MANIFEST.json` | EXE custody record |
| `runs/wb06/W3_CLS.md` | Wave-5 partial; superseded by this wave-14 terminal band |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close composition (wave 14)

Terminal EQO-W3-WB06-CLS: compose session E + latest EXE @ pin. **Session E verdict not upgraded** by implementer.

## Aggregate rationale (session E + EXE)

Session E aggregate **`FAIL`** stands @ pin. Latest EXE log does **not** upgrade session E and does **not** assert PASS for this WB.

## Required follow-up routes

1. Re-run the slice with the `llm-langchain-ollama` extra installed, or split the unit probe from the CONFIG-X files so one missing extra cannot hide them. Attach the log.
2. A4-042: assign an owner for `intergrax/globals` and decide whether the Poland-specific defaults stay, move to configuration or become neutral.
3. A4-025: replay the CONFIG-X gates and record which of the 22 runtime files are inside the closed-world scan, or record the scope limitation with an owner.
4. A5-002: attach the replay of the guard probe when the extra is available (confirmation only).

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Wave-14 compose cites pytest logs @ pin; implementer does not upgrade session E.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
- Wave-14 terminal CLS band: **no PASS**, **no** `EQO_CLOSED`.
