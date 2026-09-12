# Diagnostic Problem persistence contract architecture (HARDENING-8)

## Previous problem

Diagnostic Problem persistence defined its domain port inside runtime:

```text
intergrax/runtime/diagnostics/problem_persistence.py
  ABC ``ProblemPersistence``
  conflict / integrity errors
  ``ProblemListPage``
```

Runtime also imported grouping subject types directly, so the **persistence contract lived in the execution plane** instead of the shared contracts layer. That weakened parity with execution evidence persistence (NPSC-5F) and self-healing repository ports, where **contracts own the port** and runtime supplies adapters only.

## Target boundary

```text
Diagnostic domain / lifecycle (runtime)
        |
        v
intergrax/contracts/diagnostics/
  problem_persistence.py   — ProblemPersistence port (Protocol)
  problem_record.py        — PersistedProblem structural contract
  problem_identity.py      — ProblemId, ProblemStatus, aggregate health
  subject_ref.py           — subject index port for lookups
  reconciliation_key.py    — reconciliation index port
  diagnostic_read_model.py — list/read projections
  diagnostic_repository.py — repository alias
        |
        v
intergrax/runtime/diagnostics/
  in_memory_problem_persistence.py
  document_store_problem_persistence.py
  problem_persistence.py   — compatibility re-export only (no port definition)
        |
        v
Configured provider (SQLite document store, in-memory lab, future vendors)
```

**Rule:** `intergrax/contracts/**` must not import `intergrax/runtime/**` for diagnostic persistence (enforced by `test_diagnostic_persistence_contract_boundary.py` and HARDENING-3).

## Data flow

1. **Composition root** (harness / application wiring) selects a `ProblemPersistence` implementation (`wire_problem_persistence`, test helpers, or explicit DI).
2. **ProblemLifecycleEngine** and **DiagnosticReadService** depend on the port type from contracts (via runtime re-export or direct contract import).
3. **Adapters** translate `PersistedProblem` rows to provider documents; codecs stay in runtime (`problem_record_codec.py`).
4. **Read models** (`ProblemListPage`) are contract types; list ordering semantics remain unchanged.

## Adapters and extension

| Adapter | Role |
|---------|------|
| `InMemoryProblemPersistence` | Tests, conformance, local lab — not production default |
| `DocumentStoreProblemPersistence` | Production path over `ConditionalDocumentStore` |
| Future external provider | New runtime module implementing `ProblemPersistence` without changing contracts |

To add a backend:

1. Implement all `ProblemPersistence` methods with CAS / idempotency semantics matching `persistence_conformance.assert_problem_persistence_conformance`.
2. Wire the adapter at the composition root — **do not** construct stores inside lifecycle or orchestrator code.
3. Keep vendor SDK imports out of `intergrax/contracts/diagnostics/`.

## Compatibility

- Public method names and semantics (`get`, `create`, `update`, `query_problems`, …) are unchanged.
- `intergrax.runtime.diagnostics.problem_persistence` re-exports contract symbols for existing import paths.
- `ProblemId`, `ProblemStatus`, and `ProblemOccurrenceAggregateHealth` canonical definitions moved to `intergrax/contracts/diagnostics/problem_identity.py`; runtime lifecycle imports them without behavior change.
- Execution authority, orchestrator flow, and diagnostic lifecycle logic were not modified in this hardening.

## Verification

- `tests/unit/runtime/diagnostics/test_diagnostic_persistence_contract_boundary.py` — contracts ↔ runtime boundary
- Existing diagnostic persistence / lifecycle test suites — regression
- `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py` — global contracts layer gate
