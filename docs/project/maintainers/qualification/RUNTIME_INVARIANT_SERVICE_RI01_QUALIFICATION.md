# Runtime Invariant Service — RI-01 Qualification

**Task:** RI-01 — Runtime Invariant Service foundation  
**Branch:** `development`  
**Status:** CLOSED (pending independent GitHub audit of commit SHA)

## Ownership model

| Layer | Owns |
| --- | --- |
| Domain | Invariant meaning, rule semantics, probe/evidence interpretation |
| Platform RI foundation | Contracts, invocation protocol, deterministic runner, aggregation, failure normalization, report shape |
| Consumer | Policy, diagnostics projection, CI gates, remediation |

RI is **not** governance authority, execution authority, policy engine, diagnostics owner, or runtime mutator.

## Public contracts

- `intergrax/contracts/runtime_invariants.py` — typed IDs, selection, results, report, rule/pack protocols, composition errors.

## Shared foundation (no domain imports)

- `intergrax/runtime/invariants/service.py` — `RuntimeInvariantService`
- `intergrax/runtime/invariants/default_runner.py` — sequential deterministic evaluation
- `intergrax/runtime/invariants/composition.py` — duplicate pack/rule fail-fast
- `intergrax/runtime/invariants/foundation_composition.py` — optional default three-pack wiring

## Domain rule packs (initial)

| Pack | Location | Stable rule IDs |
| --- | --- | --- |
| Execution | `intergrax/runtime/execution/invariants/` | `EE-INV-001` … `EE-INV-003` |
| Delegated provider | `intergrax/runtime/execution/delegated_execution/invariants/` | `DELEGATION-INV-001` … `DELEGATION-INV-004` |
| Governance | `intergrax/runtime/governance/invariants/` | `GOV-INV-001` |

Probes are contract-first; rules are read-only.

## Evaluation semantics

- Statuses: `PASS`, `VIOLATION`, `NOT_APPLICABLE`, `EVALUATION_ERROR`
- Overall: `CONFORMANT`, `NON_CONFORMANT`, `INDETERMINATE` (not enforcement)
- Rule order: `(domain, rule_id, rule_version)` after composition
- Exceptions → `EVALUATION_ERROR` with sanitized summary (`invariant evaluation failed`)
- `evaluation_id` ≠ `ExecutionId` (execution may appear as correlation only)

## Pluginability

External `RuntimeInvariantRulePack` / `RuntimeInvariantRule` implementations work without runner changes. No global registry.

## Tests

- Matrix RI-T1 … RI-T15: `tests/unit/runtime/invariants/test_runtime_invariant_foundation_matrix.py`
- Architecture gates: `tests/unit/runtime/invariants/test_runtime_invariant_architecture_gates.py`
- Domain packs: `tests/unit/runtime/invariants/test_domain_invariant_packs.py`
- Qualification gate: `tests/unit/runtime/invariants/test_runtime_invariant_qualification.py`

## Qualification SHA

Record at commit close:

```text
TASK_SHA: (set by commit — see git rev-parse HEAD after push)
DIRECT_PARENT: (HEAD^)
```

## Explicit non-goals (RI-01)

- No persistence / event store for reports
- No mandatory application startup gate
- No ToolRuntime (TR-01) invariant freeze
- No P2.1 runtime semantic changes
