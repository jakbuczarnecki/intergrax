# Platform Execution Unification — U4 Child Execution Closure

## BY-01 before

```text
DelegatedSubtaskServiceFactory.create (child_execution omitted)
  → as_child_execution_port(ChildExecutionRunner())
  → DelegatedSubtaskService
```

Factory owned child runner construction — bypassed explicit composition-root injection (EP-15 / P1).

## BY-01 after (canonical work port)

```text
build_production_agent_capability_runtime
  → build_production_delegated_subtask_child_execution_port
  → delegated_subtask_child_execution_work_port (ledger)
  → ProductionAgentCapabilityRuntime.delegated_subtask_child_execution

DelegatedSubtaskServiceFactory.create(
  child_execution=capability_runtime.delegated_subtask_child_execution.port(),
)
  → DelegatedSubtaskService
  → ChildExecutionPort (child_execution_port_from_work_port)
  → DelegatedSubtaskChildExecutionWorkPort (ExecutionWorkPort)
  → ChildExecutionRunner (runtime internal)
  → _DelegatedSpecialistEnvelopeDelegate → per-invocation specialist
```

## Canonical production call path

| Layer | Owner |
| --- | --- |
| Composition | `production_delegated_subtask_child_execution_wiring.py` |
| Domain port | `ChildExecutionPort` (`agent_distribution/delegated_subtasks.py`) |
| Port adapter | `child_execution_port_from_work_port` / `delegated_subtask_child_port.py` |
| Work port | `DelegatedSubtaskChildExecutionWorkPort` (`execution_work_port.py`) |
| Child admission | `ChildExecutionRunner` (internal to work port) |
| Specialist routing | `_DelegatedSpecialistEnvelopeDelegate` (per-call `DelegatedSubtaskDelegate`) |
| Orchestration service | `DelegatedSubtaskService` |
| Physical delegation policy | `ProductionAgentCapabilityRuntime.physical_delegation_governance` |

**Nexus:** `build_production_delegated_subtask_child_execution_port(nexus_loop=...)` aligns **run budget / ledger** with the active Nexus run — **NEXUS BUDGET ALIGNMENT**, not Nexus child scheduling. Single delegated subtask execution (NPSC-5A) does not require fan-out scheduler.

## Identity / authority / governance / lineage

- **Identity:** child ids minted inside `ChildExecutionRunner` under active parent (`test_production_composition_child_execution_lineage`, `test_u4_production_child_port_invokes_canonical_work_port`).
- **Authority:** `DefaultStrictAuthorityPolicy` on runner; scopes from `DelegatedChildExecutionOptions` forwarded by work port.
- **Governance:** unchanged — `physical_delegation_governance` on `DelegatedSubtaskService`.
- **Lineage:** unchanged — child parent execution id preserved in AC-4 production E2E.

## Tests

- `tests/unit/runtime/architecture/test_platform_execution_unification_u4_child_execution_closure.py`
- `tests/unit/applications/test_ac4_phase9_production_composition_e2e.py`
- P0 inventory gate (BY-01 closed; production wiring does not import `ChildExecutionRunner`)

## EP-15

**CANONICAL** — composition → `ChildExecutionPort` → `DelegatedSubtaskChildExecutionWorkPort` → `ChildExecutionRunner` → envelope specialist delegate.

## Remaining before U5

- **EP-17:** UNCHANGED / AMBIGUOUS — REQUIRES OWNER DECISION
- **EP-14:** UNCHANGED / CANONICAL WITH GAP
- **U5:** full zero-bypass re-qualification across all entrypoints
