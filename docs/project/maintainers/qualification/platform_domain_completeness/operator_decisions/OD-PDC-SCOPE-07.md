# OD-PDC-SCOPE-07 — Plugin entry-point provenance enforcement — implement or authorize scope

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SCOPE-07 |
| **Stage** | PDC-X |
| **Status** | **PENDING** |
| **Requirements** | PDC-D6-R016 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

Agent packages are attested and trust-checked at install, but plugin entry-point admission checks structure only (no signature/provenance). Must plugin provenance be enforced before freeze (FRZ-SEC-06)?

## Recommended option (A)

**Implement** provenance admission for platform plugins by reusing the agent package trust/attestation mechanism (`AgentPackageTrustPolicy`, Ed25519 verifier) at entry-point load; unsigned plugins rejected in strict profiles.

## Alternate option (B)

Authorize scope: FRZ-SEC-06 is satisfied for agent packages only; plugins remain trusted-by-installation, recorded as a freeze limitation.

## Operator record

_Pending._
