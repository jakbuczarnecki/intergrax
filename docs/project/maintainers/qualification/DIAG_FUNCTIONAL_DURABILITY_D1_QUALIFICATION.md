# DIAG-FUNCTIONAL-DURABILITY-D1 / D1-R1 Qualification

**Verdict:** PASS (see machine artifact)

**Date:** 2026-09-03

**Branch:** `development`

## History

1. **D1 initial implementation** — persistence architecture and correctness **PASS** (contract gates D1-CONTRACT-A … J on `InMemoryDocumentStore`).
2. **Independent audit** — process-restart durability claim **REJECTED** because canonical D1 runner used `InMemoryDocumentStore` and D1-CONTRACT-D only recreated the persistence adapter in-process (same store object survived).
3. **D1-R1** — real MongoDB-backed `DocumentStore` with separate writer/reader OS processes — **PASS** (2026-09-03).

> First canonical run failed D1-R1-D (tenant isolation harness defect); fixed and re-qualified PASS in same session.

## D1 architecture (unchanged)

```text
FunctionalEvidencePersistence
        ↑
        ├── InMemoryFunctionalEvidencePersistence
        └── DocumentStoreFunctionalEvidencePersistence
```

## D1-CONTRACT gates (in-memory — NOT process durability)

| Gate | Result | Note |
| ---- | ------ | ---- |
| D1-CONTRACT-A durable append/read | PASS | in-memory store |
| D1-CONTRACT-B idempotent duplicate | PASS | |
| D1-CONTRACT-C conflicting duplicate | PASS | |
| D1-CONTRACT-D adapter replacement | PASS | **NOT process restart** |
| D1-CONTRACT-E cross-domain round-trip | PASS | |
| D1-CONTRACT-F tenant/run isolation | PASS | |
| D1-CONTRACT-G partial index repair | PASS | |
| D1-CONTRACT-H corruption fail-closed | PASS | |
| D1-CONTRACT-I concurrent duplicate append | PASS | |
| D1-CONTRACT-J backend pluginability | PASS | |

Run with: `uv run python -m tests.system.functional_diagnostics_durability.runner --contract-only`

## D1-R1 gates (real Mongo process durability)

| Gate | Required | Description |
| ---- | -------- | ----------- |
| D1-R1-A | YES | Writer process → Mongo → reader process |
| D1-R1-B | YES | Cross-process idempotent retry |
| D1-R1-C | preferred | Cross-process conflict detection |
| D1-R1-D | YES | Tenant isolation on Mongo |
| D1-R1-E | YES | Paginated recovery (`query_page_limit=2`) |
| D1-R1-F | YES | Assessment recovery fidelity |
| D1-R1-G | preferred | Backend plugin abstraction (unit) |

Default runner: `uv run python -m tests.system.functional_diagnostics_durability.runner`

If Mongo unavailable: **BLOCKED** (no in-memory fallback).

## Production provider path

```text
applications/local_workspace_application/workspaces/document_store_factory.py
  → resolve_lkw_runtime_document_store()
  → create_mongodb_document_store()   [when INTERGRAX_MONGODB_URI set]

applications/local_workspace_application/host/factory.py
  → wire_functional_evidence_runtime(
        document_store=assert_conditional_document_store(lkw_document_store),
        cursor_secret=resolve_problem_list_cursor_secret(),
     )
```

D1-R1 qualification uses `create_mongodb_document_store()` directly with qualification-scoped collection `intergrax_diag_d1_r1_<run>`.

## Authenticity

| Field | Required |
| ----- | -------- |
| `writer_reader_same_process` | `false` |
| `backend_in_memory` | `false` |
| `backend_mocked` | `false` |
| `raw_pymongo_bypass` | `false` |
| `production_provider_factory_used` | `true` |

## What D1-R1 proves (after PASS)

```text
FUNCTIONAL EVIDENCE DURABILITY
= QUALIFIED AGAINST REAL MONGODB-BACKED DOCUMENTSTORE

FUNCTIONAL DIAGNOSTIC PROCESS-RESTART RECOVERY
= QUALIFIED
```

## What remains OPEN

- Production scale = OPEN
- Disaster recovery = OPEN
- H1 = OPEN
- Other durable vendors = NOT YET QUALIFIED

## Machine artifacts

- `.tmp/session/diag-functional-durability-d1/qualification-report.json` — D1-R1 result
- `.tmp/session/diag-functional-durability-d1/qualification-report-pre-r1.json` — preserved D1 pre-R1
- `.tmp/session/diag-functional-durability-d1/qualification-report-contract.json` — contract-only
