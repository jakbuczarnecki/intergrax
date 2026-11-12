# OD-PDC-SCOPE-07 — Plugin entry-point provenance enforcement — implement or authorize scope

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SCOPE-07 |
| **Stage** | PDC-X |
| **Status** | **APPROVED — Option A** |
| **Requirements** | PDC-D6-R016 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

Agent packages are attested and trust-checked at install, but plugin entry-point admission checks structure only (no signature/provenance). Must plugin provenance be enforced before freeze (FRZ-SEC-06)?

## Recommended option (A)

**Implement** provenance admission for platform plugins by reusing the agent package trust/attestation mechanism (`AgentPackageTrustPolicy`, Ed25519 verifier) at entry-point load; unsigned plugins rejected in strict profiles.

## Alternate option (B)

Authorize scope: FRZ-SEC-06 is satisfied for agent packages only; plugins remain trusted-by-installation, recorded as a freeze limitation.

## Operator record

Approved: operator @ 2026-11-12T07:11:25+01:00

**Selected variant:** Option A (recommended).

Provenance: the Operator answered the P3-gate question "Accept the recommended option (A) for all P3-gate ODs?" in chat with the words "Accept all A". This record was transcribed on the Operator's instruction under T-IGX-036-E.
