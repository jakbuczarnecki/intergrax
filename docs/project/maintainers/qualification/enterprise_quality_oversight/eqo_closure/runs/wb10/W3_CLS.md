# EQO-W3-WB10 — W3 CLS disposition (wave 5)

**Task ID:** `EQO-W3-WB10-CLS`
**Disposition:** PARTIAL — not EQO_CLOSED.
**AUD session E (aggregate):** `FAIL` (AUD-2, `runs/wb10/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb10/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb10/AUD_SESSION_E.md` | Independent session E verdict |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close note (CLS_PRE)

Qualification pytest slice @ pin. Terminal EQO-W3-WB10-CLS deferred.

## Aggregate rationale (session E)

`FAIL`: three findings are contradicted by gates that fail at the pin, not only by source reading. No finding in this WB is decidable as PASS.

## Required follow-up routes

1. Classify `intergrax/platform/enterprise_audit/impact_map.py` for TRACE-X P6, and replace prefix-wide mappings with file-level classification (see WB07 A2-049).
2. Refresh `REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS` or run the ROADMAP-REPLAY-X requalification for the product delta, and clear the 3 closure counters and the B6 R2 reconciliation violation.
3. Re-run the TRACE-X P6 bundle with `TRACE_X_P6_PASS1=1` so the adversarial bundle is not skipped.
4. Replay DUP-X and cite the EBH-7 GHA run record, or record the limitation (A2-030, second half).
5. Both gates may need a new pin: the verdict binds to `28c27cb`, and `development` has moved.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
