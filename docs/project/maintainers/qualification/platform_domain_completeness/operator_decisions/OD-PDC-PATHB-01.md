# OD-PDC-PATHB-01 — Path B exclusions for PDC-X (Dimension B live qualification)

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-PATHB-01 |
| **Stage** | PDC-X |
| **Status** | **APPROVED — Option A** |
| **Requirements** | PDC-D3-R021, PDC-D6-R026, live portions of PDC-D3-R017 and PDC-D6-R010 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

Confirm that live provider/environment qualification stays with PROD-Q Dimension B and is not a PDC-X closure condition.

## Recommended option (A)

**Confirm** per roadmap §1.2 / §2.0.3 and checklist lines 57–68: live provider parity (Ollama, sandbox providers E2B/Modal/Daytona), production auth-bypass live proof and FRZ-PRD-01/02 stay OPEN under PROD-Q; PDC-X certifies architecture (Dimension A) only. No expansion of Path B.

## Alternate option (B)

Pull a live slice into PDC-X (would require vendor environments, outside the C10/C15 freeze restriction).

## Operator record

Approved: operator @ 2026-11-12T07:11:25+01:00

**Selected variant:** Option A (recommended).

Provenance: the Operator answered the P3-gate question "Accept the recommended option (A) for all P3-gate ODs?" in chat with the words "Accept all A". This record was transcribed on the Operator's instruction under T-IGX-036-E.
