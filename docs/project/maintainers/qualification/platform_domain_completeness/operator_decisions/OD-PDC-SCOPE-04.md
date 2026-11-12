# OD-PDC-SCOPE-04 — Governed secure artifact download — implement or authorize exclusion

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SCOPE-04 |
| **Stage** | PDC-X |
| **Status** | **APPROVED — Option A** |
| **Requirements** | PDC-D8-R010 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

Only provider-level `presigned_url` exists. Must a governed download path (authorization, tenant check, audit, expiry) exist before freeze?

## Recommended option (A)

**Implement** as part of the governed read port in ADR-PDC-08 (download = governed read + short-lived provider URL). Recommended because a presigned URL issued without the governed port bypasses artifact access authorization (FRZ-SEC-08).

## Alternate option (B)

Authorize exclusion and forbid presigned URL issuance from platform paths until implemented.

## Operator record

Approved: operator @ 2026-11-12T07:11:25+01:00

**Selected variant:** Option A (recommended).

Provenance: the Operator answered the P3-gate question "Accept the recommended option (A) for all P3-gate ODs?" in chat with the words "Accept all A". This record was transcribed on the Operator's instruction under T-IGX-036-E.
