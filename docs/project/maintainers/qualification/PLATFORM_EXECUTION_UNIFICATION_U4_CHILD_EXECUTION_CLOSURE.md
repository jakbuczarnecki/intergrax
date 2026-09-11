# Platform Execution Unification — U4 Child Execution Closure

## BY-01 before

```text
DelegatedSubtaskServiceFactory.create (child_execution omitted)
  → as_child_execution_port(ChildExecutionRunner())
  → DelegatedSubtaskService
```

Factory owned child runner construction — bypassed explicit composition-root injection (EP-15 / P1).

## BY-01 after

```text
build_production_agent_capability_runtime
  → build_production_delegated_subtask_child_execution_port
  → as_child_execution_port(ChildExecutionRunner with RunBudget ledger)
  → ProductionAgentCapabilityRuntime.delegated_subtask_child_execution

DelegatedSubtaskServiceFactory.create(
  child_execution=capability_runtime.delegated_subtask_child_execution.port(),
)
  → DelegatedSubtaskService
  → ChildExecutionPort
  → ChildExecutionRunner (low-level admission only)
```

## Canonical production call path

| Layer | Owner |
| --- | --- |
| Composition | `production_delegated_subtask_child_execution_wiring.py` |
| Port contract | `ChildExecutionPort` (`agent_distribution/delegated_subtasks.py`) |
| Runtime adapter | `as_child_execution_port` / `delegated_subtask_child_port.py` |
| Admission | `ChildExecutionRunner` (same class family as `ChildExecutionWorkPort`) |
| Orchestration service | `DelegatedSubtaskService` |
| Physical delegation policy | `ProductionAgentCapabilityRuntime.physical_delegation_governance` |

Optional Nexus alignment: `build_production_delegated_subtask_child_execution_port(nexus_loop=...)` mirrors compensation wiring budget seam.

## Identity / authority / governance / lineage

- **Identity:** unchanged — child ids minted inside `ChildExecutionRunner` under active parent (`test_production_composition_child_execution_lineage`).
- **Authority:** unchanged — `DefaultStrictAuthorityPolicy` on runner.
- **Governance:** unchanged — `physical_delegation_governance` on `DelegatedSubtaskService`.
- **Lineage:** unchanged — child parent execution id preserved in AC-4 production E2E.

## Tests

- `tests/unit/runtime/architecture/test_platform_execution_unification_u4_child_execution_closure.py`
- `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py`
- P0 inventory gate (BY-01 closed, import surface moved to wiring module)

## Remaining before U5

- **EP-17:** UNCHANGED / AMBIGUOUS — REQUIRES OWNER DECISION
- **EP-14:** UNCHANGED / CANONICAL WITH GAP
- **U5:** full zero-bypass re-qualification across all entrypoints
