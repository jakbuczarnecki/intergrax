# EQO-W3-WB05 — W3 CLS terminal disposition (wave 14)

**Task ID:** `EQO-W3-WB05-CLS`
**Disposition:** `DISPOSITIONED_NOT_CLOSED` — not PASS; campaign aggregate remains **0 PASS** @ pin.
**AUD session E (aggregate):** `BLOCKED` (AUD-1, `runs/wb05/AUD_SESSION_E.md`)
**Wave-13 / latest EXE:** exit `0` (14 passed in 32.86s) — `runs/wb05/pytest_wb05_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-13/exe-wb05`); wave-13 EXE refresh.
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb05/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb05/AUD_SESSION_E.md` | Independent session E verdict (**unchanged**) |
| `runs/wb05/pytest_wb05_exe.log` | Latest EXE @ pin |
| `runs/wb05/RUN_MANIFEST.json` | EXE custody record |
| `runs/wb05/W3_CLS.md` | Wave-5 partial; superseded by this wave-14 terminal band |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close composition (wave 14)

Terminal EQO-W3-WB05-CLS: compose session E + latest EXE @ pin. **Session E verdict not upgraded** by implementer.

## Aggregate rationale (session E + EXE)

Session E aggregate **`BLOCKED`** stands @ pin. Latest EXE log does **not** upgrade session E and does **not** assert PASS for this WB.

## Required follow-up routes

1. Create `DES_PREREG.json` and a DIS record for WB05 that cites this verdict. The integrator or EXE owner does this in a later wave, not this session.
2. A4-007: add an adversarial runtime proof for the loader seams, or an owner decision that sanctions the registries as the only loadable roots.
3. A3-011: classify the 4 `hasattr` probes (semantic or non-semantic) and record the result.
4. R1: add `agents` to the gate scan roots, or record why it is excluded.
5. R2: record the closed list as owner-controlled in the gate documentation.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Wave-14 compose cites pytest logs @ pin; implementer does not upgrade session E.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
- Wave-14 terminal CLS band: **no PASS**, **no** `EQO_CLOSED`.
