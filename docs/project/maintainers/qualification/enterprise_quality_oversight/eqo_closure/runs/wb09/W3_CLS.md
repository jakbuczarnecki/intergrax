# EQO-W3-WB09 — W3 CLS terminal disposition (wave 14)

**Task ID:** `EQO-W3-WB09-CLS`
**Disposition:** `DISPOSITIONED_NOT_CLOSED` — not PASS; campaign aggregate remains **0 PASS** @ pin.
**AUD session E (aggregate):** `FAIL` (AUD-2, `runs/wb09/AUD_SESSION_E.md`)
**Wave-13 / latest EXE:** exit `0` (66 passed in 66.86s) — `runs/wb09/pytest_wb09_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-6/exe-docs`); latest EXE @ pin (pre wave-14 CLS).
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb09/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb09/AUD_SESSION_E.md` | Independent session E verdict (**unchanged**) |
| `runs/wb09/pytest_wb09_exe.log` | Latest EXE @ pin |
| `runs/wb09/RUN_MANIFEST.json` | EXE custody record |
| `runs/wb09/W3_CLS.md` | Wave-5 partial; superseded by this wave-14 terminal band |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close composition (wave 14)

Terminal EQO-W3-WB09-CLS: compose session E + latest EXE @ pin. **Session E verdict not upgraded** by implementer.

## Aggregate rationale (session E + EXE)

Session E aggregate **`FAIL`** stands @ pin. Latest EXE log does **not** upgrade session E and does **not** assert PASS for this WB.

## Required follow-up routes

1. A2-027: owner updates the SKILL-X scope note (roadmap L487) and runs a targeted replay of `host_lifecycle.py` disable/remove/upgrade and the pinning wiring, or records them as outside scope.
2. A4-040: operator decides whether reference Tier-3 apps must start through the hosting lifecycle engine or are excepted with scope and reopen condition.
3. Re-run the P5 wrapper with the `mcp` extra (shared with WB04) so the slice has no unrelated red.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Wave-14 compose cites pytest logs @ pin; implementer does not upgrade session E.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
- Wave-14 terminal CLS band: **no PASS**, **no** `EQO_CLOSED`.
