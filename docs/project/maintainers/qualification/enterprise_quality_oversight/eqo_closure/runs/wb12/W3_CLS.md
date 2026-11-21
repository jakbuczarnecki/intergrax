# EQO-W3-WB12 — W3 CLS disposition (wave 5)

**Task ID:** `EQO-W3-WB12-CLS`
**Disposition:** PARTIAL — not EQO_CLOSED.
**AUD session E (aggregate):** `BLOCKED` (AUD-1, `runs/wb12/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb12/CLS_PRE.md` | Partial pre-close memo (post-AUD integrator) |
| `runs/wb12/AUD_SESSION_E.md` | Independent session E verdict |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Pre-close note (CLS_PRE)

PRE_AUD slice; owner/operator prerequisite. Terminal EQO-W3-WB12-CLS deferred.

## Aggregate rationale (session E)

`BLOCKED`: none of the six hypotheses is falsified or confirmed by the slice. The only genuinely relevant test (ENT-A-DUP-01 semantic gate) was skipped. One finding (A4-039) shows confirmed duplication that needs an owner decision. No stated disposition is contradicted, so this is not `FAIL`, but no finding qualifies for `PASS`.

## Required follow-up routes

1. Re-run the slice with `ENT_AUDIT_X_SEMANTIC=1` at the pin and attach the log (A4-047, A5-010).
2. A4-039: owner decision on the two `ProofReceipt` models and stores, recorded as a scoped limitation or a rename ticket.
3. A3-019: replay the EXE/GOV/CTL/REC/HRN rows at a current SHA, or record the SCOPED replay as the accepted basis.
4. A8-020: owner/operator decision to track the Protocol-v2 remediation or except it.
5. A2-031: run or cite the NPSC-4.2 allowlist gate result at the pin.
6. A5-010: produce the module delta since the acceptance pin and check it against the DUP-X census.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
