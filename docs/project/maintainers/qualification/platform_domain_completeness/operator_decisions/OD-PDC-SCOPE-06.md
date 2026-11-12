# OD-PDC-SCOPE-06 — RAG source freshness / index consistency policy — implement or authorize exclusion

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-SCOPE-06 |
| **Stage** | PDC-X |
| **Status** | **APPROVED — Option A** |
| **Requirements** | PDC-D2-R005 |
| **Gate** | PDC-P3 (architecture lock) |

## Question

Only an embedding-version warning and an optional reindex hook exist; no source freshness or staleness check. Mandatory before freeze?

## Recommended option (A)

**Implement a minimal freshness contract**: source revision/timestamp on indexed documents plus a typed staleness verdict at retrieval, with the reindex hook admitted as background work. No new retrieval engine.

## Alternate option (B)

Authorize exclusion as a documented RAG limitation (`RAG.md` already states PRODUCTION_QUALIFIED_WITH_LIMITATIONS).

## Operator record

Approved: operator @ 2026-11-12T07:11:25+01:00

**Selected variant:** Option A (recommended).

Provenance: the Operator answered the P3-gate question "Accept the recommended option (A) for all P3-gate ODs?" in chat with the words "Accept all A". This record was transcribed on the Operator's instruction under T-IGX-036-E.
