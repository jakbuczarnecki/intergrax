# EQO-W3-WB01 — W3 CLS terminal disposition (wave 14)

**Task ID:** `EQO-W3-WB01-CLS`
**Disposition:** `DISPOSITIONED_NOT_CLOSED` — not PASS; campaign aggregate remains **0 PASS** @ pin.
**AUD session E (aggregate):** `FAIL` (AUD-2, `runs/wb01/AUD_SESSION_E.md`)
**Wave-13 / latest EXE:** exit `1` (1 failed, 23 passed, 1 skipped in 314.65s (0:05:14)) — `runs/wb01/pytest_wb01_exe.log`, `RUN_MANIFEST.json` (`rerun_lane`: `eqo/wave-10/exe`); latest EXE @ pin (pre wave-14 CLS).
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb01/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb01/AUD_SESSION_E.md` | Independent session E verdict (**unchanged**) |
| `runs/wb01/pytest_wb01_exe.log` | Latest EXE @ pin |
| `runs/wb01/RUN_MANIFEST.json` | EXE custody record |
| `runs/wb01/W3_CLS.md` | Wave-5 partial; superseded by this wave-14 terminal band |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close composition (wave 14)

Terminal EQO-W3-WB01-CLS: compose session E + latest EXE @ pin. **Session E verdict not upgraded** by implementer.

## Aggregate rationale (session E + EXE)

Session E aggregate **`FAIL`** stands @ pin. Latest EXE log does **not** upgrade session E and does **not** assert PASS for this WB.

## Required follow-up routes

1. A4-044 and A4-015: land T-IGX-043-A / #680 on the train, then replay host-task routing and the three Tier-2 `RuntimeContext` builders.
2. A1-021: owner relabels `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:1939` as a documentation publication, or cites the audit record.
3. A5-003: owner supplies the GOV-X1 gate-weakening review or a scoped acceptance record.
4. A7-030: operator assigns an owner, target SHA and date for each attached audit obligation.
5. IIAC activation (X-03): handled in the RB-2 track, not by this WB.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Wave-14 compose cites pytest logs @ pin; implementer does not upgrade session E.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
- Wave-14 terminal CLS band: **no PASS**, **no** `EQO_CLOSED`.
