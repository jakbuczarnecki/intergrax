# EQO-W3-WB02 — W3 CLS disposition (wave 5)

**Task ID:** `EQO-W3-WB02-CLS`
**Disposition:** PARTIAL — not EQO_CLOSED.
**AUD session E (aggregate):** `FAIL` (AUD-1, `runs/wb02/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb02/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb02/AUD_SESSION_E.md` | Independent session E verdict |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close note (CLS_PRE)

PRE_AUD tenant isolation slice @ TRAIN_PIN. Terminal EQO-W3-WB02-CLS deferred.

## Aggregate rationale (session E)

`FAIL` because two findings (A4-006, A4-027) are contradicted by source at the pin and the slice gives them no coverage. The remaining four are not decidable by this slice. EXE `GREEN` (exit 0) reflects only that the selected tests pass. It does not close any hypothesis in `DES_PREREG.json`.

## Required follow-up routes

1. A4-006: owner updates the gate path and re-runs the propagation audit, then replay on a new pin.
2. A4-027: operator decision OD-EQO-08 (serving-router write set) or a recorded FRZ-TEN-10 scope exception for reference Tier-3 apps.
3. A7-009, A7-028, A7-031: operator/owner EBH-7 disposition and label standardisation, or replay of the tenant E2E routed proofs.
4. A8-006: owner doc patch or acceptance record.
5. Vectorstore isolation: execute under the `dev-ci-rag` extra (rag-guard profile) and attach the log.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
