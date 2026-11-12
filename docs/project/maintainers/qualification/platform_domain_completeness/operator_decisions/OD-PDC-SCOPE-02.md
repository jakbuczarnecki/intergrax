# OD-PDC-SCOPE-02 — APP-HOST-6 / APP-HOST-7 / APP-HOST-9E — authorize post-freeze exclusion

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SCOPE-02 |
| **Stage** | PDC-X |
| **Status** | **APPROVED — Option A** |
| **Requirements** | PDC-D4-R011, PDC-D4-R012, PDC-D4-R013 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

InteractionProfile (no class exists), OS service-manager adapters and host plugins are Planned. Are they mandatory for the declared supported hosting lifecycle?

## Recommended option (A)

**Authorize exclusion** of all three. Evidence: the declared lifecycle (start, ready, degrade, drain, supervise, restart) is complete without them (ADR-PDC-04); MCP exposure is owned by MCP-SERVER-X; an InteractionProfile now would compete with ApplicationEnvironmentProfile authority (GAP_AUDIT §16 PRESERVE). Recorded as FRZ-DEBT-06 evidence-backed exclusions.

## Alternate option (B)

Implement APP-HOST-6 (and/or 7, 9E) before freeze, under a new ADR.

## Operator record

Approved: operator @ 2026-11-12T07:11:25+01:00

**Selected variant:** Option A (recommended).

Provenance: the Operator answered the P3-gate question "Accept the recommended option (A) for all P3-gate ODs?" in chat with the words "Accept all A". This record was transcribed on the Operator's instruction under T-IGX-036-E.
