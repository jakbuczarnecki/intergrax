# EQO-W3-WB02 — W3 CLS terminal disposition (wave 14)

**Task ID:** `EQO-W3-WB02-CLS`
**Disposition:** `DISPOSITIONED_NOT_CLOSED` — not PASS; campaign aggregate remains **0 PASS** @ pin.
**AUD session E (aggregate):** `FAIL` (AUD-1, `runs/wb02/AUD_SESSION_E.md`)
**Wave-13 / latest EXE:** exit `1` (2 failed, 72 passed, 1 skipped in 17.77s) — `runs/wb02/pytest_wb02_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-13/exe-wb02`); wave-13 EXE refresh.
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb02/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb02/AUD_SESSION_E.md` | Independent session E verdict (**unchanged**) |
| `runs/wb02/pytest_wb02_exe.log` | Latest EXE @ pin |
| `runs/wb02/RUN_MANIFEST.json` | EXE custody record |
| `runs/wb02/W3_CLS.md` | Wave-5 partial; superseded by this wave-14 terminal band |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close composition (wave 14)

Terminal EQO-W3-WB02-CLS: compose session E + latest EXE @ pin. **Session E verdict not upgraded** by implementer.

## Aggregate rationale (session E + EXE)

Session E aggregate **`FAIL`** stands. Wave-13 EXE exit `1` records slice execution only; it does not close A4-006, A4-027, or EBH-7 gaps.

## Required follow-up routes

1. A4-006: owner updates the gate path and re-runs the propagation audit, then replay on a new pin.
2. A4-027: operator decision OD-EQO-08 (serving-router write set) or a recorded FRZ-TEN-10 scope exception for reference Tier-3 apps.
3. A7-009, A7-028, A7-031: operator/owner EBH-7 disposition and label standardisation, or replay of the tenant E2E routed proofs.
4. A8-006: owner doc patch or acceptance record.
5. Vectorstore isolation: execute under the `dev-ci-rag` extra (rag-guard profile) and attach the log.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Wave-14 compose cites pytest logs @ pin; implementer does not upgrade session E.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
- Wave-14 terminal CLS band: **no PASS**, **no** `EQO_CLOSED`.
