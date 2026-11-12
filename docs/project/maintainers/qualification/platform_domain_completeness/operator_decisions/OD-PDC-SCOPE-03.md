# OD-PDC-SCOPE-03 — LLM provider plugin layer and OCI/sidecar runtime factory — authorize exclusion

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SCOPE-03 |
| **Stage** | PDC-X |
| **Status** | **PENDING** |
| **Requirements** | PDC-D3-R020, PDC-D1-R019 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

LLM-PROVIDER-PLUGIN-1 (Planned) and non-VENV_BUNDLE runtime factories (deferred) — required before freeze?

## Recommended option (A)

**Authorize exclusion.** Provider replaceability is already qualified through `LLMAdapterRegistry` registrations (EBH-5) and agent packaging works through VENV_BUNDLE; both are additive extensions that do not change frozen contracts.

## Alternate option (B)

Implement either before freeze under its own ADR.

## Operator record

_Pending._
