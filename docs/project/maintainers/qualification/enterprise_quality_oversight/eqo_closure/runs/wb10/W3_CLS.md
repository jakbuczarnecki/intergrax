# EQO-W3-WB10 — W3 CLS terminal disposition (wave 14)

**Task ID:** `EQO-W3-WB10-CLS`
**Disposition:** `DISPOSITIONED_NOT_CLOSED` — not PASS; campaign aggregate remains **0 PASS** @ pin.
**AUD session E (aggregate):** `FAIL` (AUD-2, `runs/wb10/AUD_SESSION_E.md`)
**Wave-13 / latest EXE:** exit `1` (6 failed, 66 passed, 1 skipped in 98.73s (0:01:38)) — `runs/wb10/pytest_wb10_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-10/exe`); latest EXE @ pin (pre wave-14 CLS).
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb10/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb10/AUD_SESSION_E.md` | Independent session E verdict (**unchanged**) |
| `runs/wb10/pytest_wb10_exe.log` | Latest EXE @ pin |
| `runs/wb10/RUN_MANIFEST.json` | EXE custody record |
| `runs/wb10/W3_CLS.md` | Wave-5 partial; superseded by this wave-14 terminal band |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close composition (wave 14)

Terminal EQO-W3-WB10-CLS: compose session E + latest EXE @ pin. **Session E verdict not upgraded** by implementer.

## Aggregate rationale (session E + EXE)

Session E aggregate **`FAIL`** stands @ pin. Latest EXE log does **not** upgrade session E and does **not** assert PASS for this WB.

## Required follow-up routes

1. Classify `intergrax/platform/enterprise_audit/impact_map.py` for TRACE-X P6, and replace prefix-wide mappings with file-level classification (see WB07 A2-049).
2. Refresh `REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS` or run the ROADMAP-REPLAY-X requalification for the product delta, and clear the 3 closure counters and the B6 R2 reconciliation violation.
3. Re-run the TRACE-X P6 bundle with `TRACE_X_P6_PASS1=1` so the adversarial bundle is not skipped.
4. Replay DUP-X and cite the EBH-7 GHA run record, or record the limitation (A2-030, second half).
5. Both gates may need a new pin: the verdict binds to `28c27cb`, and `development` has moved.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Wave-14 compose cites pytest logs @ pin; implementer does not upgrade session E.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
- Wave-14 terminal CLS band: **no PASS**, **no** `EQO_CLOSED`.
