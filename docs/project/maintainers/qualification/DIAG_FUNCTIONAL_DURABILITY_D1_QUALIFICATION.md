# DIAG-FUNCTIONAL-DURABILITY-D1 Qualification

**Verdict:** PASS (see machine artifact)

**Date:** 2026-09-02

**Branch:** `development`

## Scope

D1 proves durable `PlatformFunctionalEvidence` persistence via backend-neutral `FunctionalEvidencePersistence`, with `DocumentStoreFunctionalEvidencePersistence` on `ConditionalDocumentStore`, process-restart recovery, and cross-domain round-trip fidelity.

## Backend

| Property | Value |
| -------- | ----- |
| DocumentStore implementation | `InMemoryDocumentStore` (ConditionalDocumentStore conformance) |
| Capabilities | `put_if_absent`, `get`, `put`, prefix `query` with cursor pagination |
| Durability mechanism | Shared document partition survives adapter instance replacement |
| Vendor imports in core | NONE |

Production LKW composition uses `wire_functional_evidence_runtime(document_store=assert_conditional_document_store(...))` in `applications/local_workspace_application/host/factory.py`.

## Contract

- `append(evidence) -> PlatformFunctionalEvidence` — append-only, idempotent on `evidence_id`
- `query_evidence(request) -> FunctionalEvidenceQueryPage` — tenant/task/run scoped keyset pagination

## Record schema

- `intergrax.functional_evidence.persistence.v1`
- Index: `intergrax.functional_evidence.index.v1`

## Gates (D1-A … D1-J)

| Gate | Result |
| ---- | ------ |
| D1-A durable append/read | PASS |
| D1-B idempotent duplicate | PASS |
| D1-C conflicting duplicate | PASS |
| D1-D restart recovery | PASS |
| D1-E cross-domain round-trip | PASS |
| D1-F tenant/run isolation | PASS |
| D1-G partial index repair | PASS |
| D1-H corruption fail-closed | PASS |
| D1-I concurrent duplicate append | PASS |
| D1-J backend pluginability | PASS |

## Fidelity

- evidence_round_trip_fidelity: 100%
- identity_fidelity: 100%
- assessment_recovery_fidelity: 100%

## Architecture invariants

- `FunctionalDiagnosticAnalyzer`: UNCHANGED
- domain_specific_persistence branches: 0
- hidden in-memory fallback in production wiring: NONE

## What D1 proves

- Functional evidence survives process restart against durable DocumentStore
- Idempotent append + conflict detection + index repair
- Cross-domain evidence kinds round-trip through one codec
- Analyzer assessment identical before/after adapter replacement

## What D1 does NOT prove

- Production scale = OPEN
- H1 = OPEN
- Disaster recovery = OPEN

## Recommendation

```
FUNCTIONAL EVIDENCE DURABILITY = QUALIFIED
FUNCTIONAL DIAGNOSTIC RECOVERY = QUALIFIED FOR PROCESS RESTART AGAINST DURABLE DOCUMENT STORE
```

**Machine artifact:** `.tmp/session/diag-functional-durability-d1/qualification-report.json`

**Runner:** `uv run python -m tests.system.functional_diagnostics_durability.runner`
