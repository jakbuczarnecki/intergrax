# EQO-W3-WB05 — W3 CLS disposition (wave 5)

**Task ID:** `EQO-W3-WB05-CLS`
**Disposition:** PARTIAL — not EQO_CLOSED.
**AUD session E (aggregate):** `BLOCKED` (AUD-1, `runs/wb05/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb05/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb05/AUD_SESSION_E.md` | Independent session E verdict |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close note (CLS_PRE)

PRE_AUD qualification slice; session E could not decide all rows. Terminal EQO-W3-WB05-CLS deferred.

## Aggregate rationale (session E)

`BLOCKED`: three findings (A2-009, A4-020, A7-021) have supporting static-gate evidence at the pin. A4-007 and A3-011 cannot be decided from this slice. The WB also has no preregistration (DES_PREREG, command hash) and no DIS artefact, so the checklist item "verify pytest log hash / command" cannot be completed and the findings were never formally dispositioned. No evidence contradicts a stated disposition, so this is not `FAIL`.

## Required follow-up routes

1. Create `DES_PREREG.json` and a DIS record for WB05 that cites this verdict. The integrator or EXE owner does this in a later wave, not this session.
2. A4-007: add an adversarial runtime proof for the loader seams, or an owner decision that sanctions the registries as the only loadable roots.
3. A3-011: classify the 4 `hasattr` probes (semantic or non-semantic) and record the result.
4. R1: add `agents` to the gate scan roots, or record why it is excluded.
5. R2: record the closed list as owner-controlled in the gate documentation.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
