# EQO-W3-WB09 — W3 CLS disposition (wave 5)

**Task ID:** `EQO-W3-WB09-CLS`
**Disposition:** PARTIAL — not EQO_CLOSED.
**AUD session E (aggregate):** `FAIL` (AUD-2, `runs/wb09/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb09/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb09/AUD_SESSION_E.md` | Independent session E verdict |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close note (CLS_PRE)

Partial optional stack; AUD FAIL on confirmed defects. Terminal EQO-W3-WB09-CLS deferred.

## Aggregate rationale (session E)

`FAIL` because both findings are confirmed by source at the pin. The P5 failure (EXE exit 1) is an environment or collection problem shared with WB04 and does not bear on these two findings.

## Required follow-up routes

1. A2-027: owner updates the SKILL-X scope note (roadmap L487) and runs a targeted replay of `host_lifecycle.py` disable/remove/upgrade and the pinning wiring, or records them as outside scope.
2. A4-040: operator decides whether reference Tier-3 apps must start through the hosting lifecycle engine or are excepted with scope and reopen condition.
3. Re-run the P5 wrapper with the `mcp` extra (shared with WB04) so the slice has no unrelated red.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
