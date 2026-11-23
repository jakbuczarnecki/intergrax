# EQO-W3-WB07 — W3 CLS terminal disposition (wave 14)

**Task ID:** `EQO-W3-WB07-CLS`
**Disposition:** `DISPOSITIONED_NOT_CLOSED` — not PASS; campaign aggregate remains **0 PASS** @ pin.
**AUD session E (aggregate):** `FAIL` (AUD-2, `runs/wb07/AUD_SESSION_E.md`)
**Wave-13 / latest EXE:** exit `1` (1 failed, 56 passed, 1 skipped in 107.04s (0:01:47)) — `runs/wb07/pytest_wb07_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-10/exe`); latest EXE @ pin (pre wave-14 CLS).
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb07/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb07/AUD_SESSION_E.md` | Independent session E verdict (**unchanged**) |
| `runs/wb07/pytest_wb07_exe.log` | Latest EXE @ pin |
| `runs/wb07/RUN_MANIFEST.json` | EXE custody record |
| `runs/wb07/W3_CLS.md` | Wave-5 partial; superseded by this wave-14 terminal band |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close composition (wave 14)

Terminal EQO-W3-WB07-CLS: compose session E + latest EXE @ pin. **Session E verdict not upgraded** by implementer.

## Aggregate rationale (session E + EXE)

Session E aggregate **`FAIL`** stands @ pin. Latest EXE log does **not** upgrade session E and does **not** assert PASS for this WB.

## Required follow-up routes

1. A2-049: re-run the TRACE-X P6 bundle with `TRACE_X_P6_PASS1=1` and replace the directory-wide prefix rows with per-file classification, or record the accepted limitation.
2. A4-034: owner decision on the five retry-budget mechanisms.
3. A4-045: owner mapping of the artifact-reference vocabularies.
4. A2-043: replay the EBH-2E/2G/2H surfaces against their certified SHAs, or record the requalification.
5. HARNESS-01: classify the new higher-layer Nexus importers (separate defect, not decided here).

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Wave-14 compose cites pytest logs @ pin; implementer does not upgrade session E.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
- Wave-14 terminal CLS band: **no PASS**, **no** `EQO_CLOSED`.
