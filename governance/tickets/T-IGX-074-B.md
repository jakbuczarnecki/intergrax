# T-IGX-074-B — EC-W3 vector isolation fixture tenant binding

## Scope

Test-only correction for LanceDB and Typesense factories in cross-tenant isolation contract tests. Pass explicit `tenant_id` into provider bundle factories while retaining in-memory vector store substitutes.

## Plan digest

Plan digest: n/a (EC-W3-CORRECTIVE-IMPLEMENTATION-04 Package B)

## Approved

Approved: operator @ 2026-11-27T10:30:00+01:00

Authorization: EC-W3-CORRECTIVE-IMPLEMENTATION-04 Package B fixture correction.

## Write set

```text
governance/tickets/T-IGX-074-B.md
tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py
```

## Test contract

```text
uv run --no-sync pytest tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py -vv -rA --tb=long
```

## Out of scope

Provider implementation, Weaviate factory, VectorIntegrationConfig, live databases.

## Lifecycle state

in_progress