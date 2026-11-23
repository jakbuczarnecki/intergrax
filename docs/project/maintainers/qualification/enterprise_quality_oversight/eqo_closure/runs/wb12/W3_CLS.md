# EQO-W3-WB12 — W3 CLS terminal disposition (wave 14)

**Task ID:** `EQO-W3-WB12-CLS`
**Disposition:** `DISPOSITIONED_NOT_CLOSED` — not PASS; campaign aggregate remains **0 PASS** @ pin.
**AUD session E (aggregate):** `BLOCKED` (AUD-1, `runs/wb12/AUD_SESSION_E.md`)
**Wave-13 / latest EXE:** exit `0` (33 passed, 1 skipped in 48.97s) — `runs/wb12/pytest_wb12_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-13/exe-wb12`); wave-13 EXE refresh.
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb12/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb12/AUD_SESSION_E.md` | Independent session E verdict (**unchanged**) |
| `runs/wb12/pytest_wb12_exe.log` | Latest EXE @ pin |
| `runs/wb12/RUN_MANIFEST.json` | EXE custody record |
| `runs/wb12/W3_CLS.md` | Wave-5 partial; superseded by this wave-14 terminal band |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close composition (wave 14)

Terminal EQO-W3-WB12-CLS: compose session E + latest EXE @ pin. **Session E verdict not upgraded** by implementer.

## Aggregate rationale (session E + EXE)

Session E aggregate **`BLOCKED`** stands @ pin. Latest EXE log does **not** upgrade session E and does **not** assert PASS for this WB.

## Required follow-up routes

1. Re-run the slice with `ENT_AUDIT_X_SEMANTIC=1` at the pin and attach the log (A4-047, A5-010).
2. A4-039: owner decision on the two `ProofReceipt` models and stores, recorded as a scoped limitation or a rename ticket.
3. A3-019: replay the EXE/GOV/CTL/REC/HRN rows at a current SHA, or record the SCOPED replay as the accepted basis.
4. A8-020: owner/operator decision to track the Protocol-v2 remediation or except it.
5. A2-031: run or cite the NPSC-4.2 allowlist gate result at the pin.
6. A5-010: produce the module delta since the acceptance pin and check it against the DUP-X census.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Wave-14 compose cites pytest logs @ pin; implementer does not upgrade session E.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
- Wave-14 terminal CLS band: **no PASS**, **no** `EQO_CLOSED`.
