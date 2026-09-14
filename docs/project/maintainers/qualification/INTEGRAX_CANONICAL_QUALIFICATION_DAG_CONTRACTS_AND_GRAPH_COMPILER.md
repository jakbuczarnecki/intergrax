# INTEGRAx-CANONICAL-QUALIFICATION-DAG-CONTRACTS-AND-GRAPH-COMPILER

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-CANONICAL-QUALIFICATION-DAG-CONTRACTS-AND-GRAPH-COMPILER` |
| **Date** | 2026-09-14 |
| **Branch** | `development` |
| **Parent design** | [`INTEGRAX_QUALIFICATION_EXECUTION_GRAPH_AUDIT_AND_OPTIMIZATION_DESIGN.md`](INTEGRAX_QUALIFICATION_EXECUTION_GRAPH_AUDIT_AND_OPTIMIZATION_DESIGN.md) (P1) |
| **Scope** | Typed DAG contracts + deterministic graph compiler (**no** pytest execution, **no** orchestrator migration) |

## Scope

In scope:

```text
definition → validation → compile QualificationExecutionPlan
```

Out of scope: coordinator migration, in-run receipt dedup, scheduler changes, production `intergrax/` runtime.

## Reused Architecture

Unchanged and reused:

- `QualificationCoordinator`, `QualificationRunManifest`, `QualificationSuite`
- `ExecutionQualificationSuiteResult`, `ExecutionQualificationRunResult`
- `QualificationPerformanceSnapshot` (unchanged in this task)

## Contracts

Public types live in `testing_support.execution_qualification.graph_contracts` and `compiler`:

| Contract | Role |
| -------- | ---- |
| `QualificationNodeKind` | `LEAF_SUITE`, `AGGREGATE_GATE`, `STATIC_GATE` |
| `QualificationGateDefinition` | `gate_id`, `requires`, `mandatory`, `declaration_index`, optional `description`, `gate_kind` |
| `QualificationProfile` | `profile_id`, `root_gate_ids` |
| `QualificationGraphDefinition` | Optional bundle: `run_manifest`, `gates`, `profiles` |
| `QualificationExecutionNode` | Compiled node with references to suite or gate |
| `QualificationExecutionPlan` | `profile_id`, `ordered_nodes`, `leaf_suite_ids`, `root_gate_ids`, `dependency_edges` |

Entry point: `compile_qualification_execution_plan(manifest, gates, profile)` and `QualificationGraphCompiler`.

## Node Identity

- `suite_id` — canonical leaf identity (`QualificationSuite` remains source of truth for pytest args and isolation metadata).
- `gate_id` — canonical aggregate/static identity.
- `suite_id == gate_id` → **fail closed** (`QualificationManifestConflictError`).
- Path lists are not primary identity.

## Graph Definition

Compiler input is explicit composition only:

```text
QualificationRunManifest (suites)
+ tuple[QualificationGateDefinition]
+ QualificationProfile
```

No filesystem scan, reflection, or global registry.

Gate `requires` entries may reference `suite_id` or `gate_id`.

## Compilation Algorithm

1. Validate profile (`root_gate_ids` non-empty; each root exists).
2. Build gate map (duplicate `gate_id` → fail).
3. Build suite map from manifest (collision with gate → fail).
4. Traverse from profile roots; mark reachable gates and suites.
5. Fail closed on missing `requires` targets and dependency cycles (deterministic cycle path).
6. Topological layering (Kahn) on the reachable subgraph.
7. Emit `QualificationExecutionNode` per reachable id (suite or gate reference, not copied payloads).
8. Emit `leaf_suite_ids` as deduplicated tuple in plan order.

Unreachable suites/gates in the manifest/catalog are **excluded** from the plan (not an error).

## Deduplication Semantics

Each reachable `suite_id` appears **once** in `ordered_nodes` and `leaf_suite_ids`. Shared leaves required by multiple gates are not duplicated in the compiled plan.

Each reachable `gate_id` appears once in `ordered_nodes`.

## Topological Ordering

Canonical `ordered_nodes` sort key:

```text
topological layer (dependency wave)
→ declaration_index (manifest index for suites, gate field for gates)
→ node_id (lexicographic tie-break)
```

Dependencies always precede dependents. Sibling order is stable across differing input tuple order when declaration metadata is unchanged.

## Validation Rules

| Rule | Failure |
| ---- | ------- |
| Empty `root_gate_ids` | `QualificationProfileError` |
| Unknown root / missing `requires` target | `QualificationProfileError` / `MissingQualificationDependencyError` |
| Cycle / self-edge | `QualificationDependencyCycleError` |
| Duplicate `gate_id` | `QualificationManifestConflictError` |
| `suite_id` / `gate_id` collision | `QualificationManifestConflictError` |
| Same `suite_id`, different `QualificationSuite` | `ConflictingQualificationSuiteDefinitionError` (`merge_qualification_suites`) |
| Duplicate `suite_id` in manifest | existing `QualificationRunManifest` `ValueError` (unchanged) |

## Typed Failures

```text
QualificationGraphError
├─ QualificationManifestConflictError
│  └─ ConflictingQualificationSuiteDefinitionError
├─ MissingQualificationDependencyError
├─ QualificationDependencyCycleError (cycle_path)
└─ QualificationProfileError
```

## Complexity

Single pass reachability + Kahn topological sort: **O(V + E)** over reachable nodes and gate `requires` edges.

## Tests

`tests/unit/testing_support/execution_qualification/test_graph_compiler.py` — happy paths, dedup, unreachable exclusion, determinism, cycles, missing deps, collisions, conflicts, topological order, no coordinator invocation, import guards.

## Regression

```bash
uv run pytest tests/unit/testing_support/execution_qualification/ -q
```

## Production Changes

**NONE**

## Findings

P1 delivers compile-only execution plans. **P2 implementation:** [`INTEGRAX_QUALIFICATION_IN_RUN_DEDUP_AND_RECEIPT_REUSE.md`](INTEGRAX_QUALIFICATION_IN_RUN_DEDUP_AND_RECEIPT_REUSE.md) (plan runner, receipt reuse, plan hardening).

## Final Verdict

```text
CANONICAL QUALIFICATION DAG CONTRACTS + GRAPH COMPILER = PASS
```

(pending CI/local static gates on the implementation commit)
