# OD-PDC-SCOPE-03 — LLM provider plugin layer and OCI/sidecar runtime factory — authorize exclusion

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SCOPE-03 |
| **Stage** | PDC-X |
| **Status** | **APPROVED — Option A** |
| **Requirements** | PDC-D3-R020, PDC-D1-R019 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

LLM-PROVIDER-PLUGIN-1 (Planned) and non-VENV_BUNDLE runtime factories (deferred) — required before freeze?

## Recommended option (A)

**Authorize exclusion.** Provider replaceability is already qualified through `LLMAdapterRegistry` registrations (EBH-5) and agent packaging works through VENV_BUNDLE; both are additive extensions that do not change frozen contracts.

## Alternate option (B)

Implement either before freeze under its own ADR.

## Operator record

Approved: operator @ 2026-11-12T07:11:25+01:00

**Selected variant:** Option A (recommended).

Provenance: the Operator answered the P3-gate question "Accept the recommended option (A) for all P3-gate ODs?" in chat with the words "Accept all A". This record was transcribed on the Operator's instruction under T-IGX-036-E.
