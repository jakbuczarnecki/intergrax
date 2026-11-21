# EQO-W3-WB03 — W3 CLS disposition (wave 5)

**Task ID:** `EQO-W3-WB03-CLS`
**Disposition:** PARTIAL — not EQO_CLOSED.
**AUD session E (aggregate):** `FAIL` (AUD-1, `runs/wb03/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb03/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb03/AUD_SESSION_E.md` | Independent session E verdict |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close note (CLS_PRE)

PRE_AUD adversarial matrix row. Terminal EQO-W3-WB03-CLS deferred.

## Aggregate rationale (session E)

`FAIL` because three findings (A4-026, A4-031, A4-037) describe silent no-op or skipped-guard behaviour that is still present in source at the pin. The passing slice exercises only the explicitly configured paths, so the interim `ACCEPTED_SCOPED_LIMITATION` is not supported by any evidence in the packet. A3-027 needs a strict-host replay.

## Required follow-up routes

1. A4-026: owner decision. Either fail closed on unset or invalid verifier mode in strict hosts, or record a scoped lab-only limitation with an owner and reopen condition. Add a default-path negative test.
2. A4-031: owner decision. Either apply the CE-01 fail-closed guard on the UAEP path, or record the limitation. Add a UAEP-path negative test.
3. A4-037: owner decision. Wire a production human decision store or fail closed, with a production-construction test. Track via T-IGX-043-E and the #680 probe.
4. A3-027: replay D4 and P2-N02 nodes at the pin with strict fixtures.
5. After the pin moves, run `EQO-W3-WB03-DRV`. Do not open a new campaign.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
