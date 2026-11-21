# EQO-W3-WB06 — W3 CLS disposition (wave 5)

**Task ID:** `EQO-W3-WB06-CLS`
**Disposition:** PARTIAL — not EQO_CLOSED.
**AUD session E (aggregate):** `FAIL` (AUD-2, `runs/wb06/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb06/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb06/AUD_SESSION_E.md` | Independent session E verdict |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close note (CLS_PRE)

EXE env BLOCKED for optional stack (mcp, llm-langchain-ollama) — documented only; no EXE fix this wave. Terminal EQO-W3-WB06-CLS deferred.

## Aggregate rationale (session E)

`FAIL` because A4-042 is confirmed by source at the pin and the slice gives it no coverage. A4-025 is not decidable without a working slice. A5-002 is supported by source.

## Required follow-up routes

1. Re-run the slice with the `llm-langchain-ollama` extra installed, or split the unit probe from the CONFIG-X files so one missing extra cannot hide them. Attach the log.
2. A4-042: assign an owner for `intergrax/globals` and decide whether the Poland-specific defaults stay, move to configuration or become neutral.
3. A4-025: replay the CONFIG-X gates and record which of the 22 runtime files are inside the closed-world scan, or record the scope limitation with an owner.
4. A5-002: attach the replay of the guard probe when the extra is available (confirmation only).

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
