# DG-001 — Execution lineage admission persistence R1

> **Task:** `DG-001-MULTI-AGENT-EXECUTION-LINEAGE-ADMISSION-PERSISTENCE-R1`  
> **Correction:** `DG-001-MULTI-AGENT-EXECUTION-LINEAGE-ADMISSION-PERSISTENCE-R1-CORRECTION`
> **Architecture:** `docs/project/maintainers/architecture/DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_ARCHITECTURE_R1.md`  
> **Baseline START_HEAD:** `0d39e21258b138a886bea6656a9f462786552654`  
> **Architecture base:** `5e2fef428329608e2437c8d5a1da87ac0b6848c7`

## Implemented contracts

- `ExecutionLineageAttemptScope`
- `ExecutionLineageAdmissionRecord`
- `ExecutionLineageSegmentRecord`
- `ExecutionLineageAttemptState`
- `ExecutionLineageAdmissionPage`
- `ExecutionLineagePersistence` (ABC)
- `ExecutionLineagePersistenceProvider` (composition gate)
- `AttemptLineageDegradationState` (runtime ContextVar)

## Provider strategy

- **R1 durable:** `DocumentStoreExecutionLineagePersistence` over `PartitionAtomicDocumentStore`
- **Tests / single-process:** `InMemoryExecutionLineagePersistence`
- **KV adapter:** NOT_IMPLEMENTED (by design)
- **Composition:** `resolve_execution_lineage_persistence(explicit_persistence=..., document_store=..., provider=...)`

## Root flow

`ExecutionRuntime.execute` → `open_attempt` → `open_segment` → bind active lineage → `ExecutionBoundary` → lineage root hook → `admit_root` (durable) → delegate.

## Child flow

`ChildExecutionRunner` peeks active lineage context and prepends one platform lineage hook before caller hooks.

## Segment continuity

- Segment identity = canonical root `ExecutionId`
- Resume: `predecessor_root_execution_id` from checkpoint historical root
- Unclean predecessor → `SEGMENT_UNCLEAN` + `degraded=True` atomically when opening successor segment
- Clean pause: `close_segment_for_resume` after successful checkpoint persistence

## Failure policy

- Root / segment open / root admission unavailable → fail closed (no delegate)
- Child admission unavailable + durable `mark_degraded` → child may continue
- Child admission unavailable + `mark_degraded` failure → fail closed
- Structural conflicts → `ExecutionLineageIntegrityError` (fail closed)
- Nested child after degraded parent without durable parent admission → fail closed (`parent admission missing`)

## Concurrency

Bounded CAS retry with monotonic `admission_position` via attempt metadata; conformance suite includes 32 concurrent siblings.

## Tests

```bash
uv run pytest tests/unit/contracts/test_execution_lineage_contracts.py tests/unit/runtime/execution/lineage/ -q
uv run pytest tests/unit/runtime/execution/test_child_execution.py tests/unit/runtime/execution/test_execution_runtime.py tests/unit/runtime/execution/test_execution_boundary.py -q
```

## Out of scope

- ExecutionReconstructor / DiagnosticReadService
- Decision System / causal evidence changes
- ExecutionTreeRecorder convergence
- KV adapter

## Original verdict (R1 — retained for audit)

**PASS** — write-side durable admission lineage implemented with production composition hooks and focused regression coverage.

**Audit note:** the original R1 implementation used non-atomic multi-row sequences (`put_if_absent` then `replace_if_match`) for admission, segment open, and seal. That verdict was **incorrect** for crash/concurrency safety claims.

## Correction — Atomicity correction

All multi-row lineage mutations now commit through `_PartitionAtomicRowStore.execute_partition_atomic_batch`:

- **Admission:** `admission:<execution_id>` primary + attempt metadata `replace_if_match` in `on_created_ops`
- **Segment open:** segment primary + attempt metadata; resume-with-open-predecessor adds predecessor `SEGMENT_OPEN → SEGMENT_UNCLEAN` in the same batch
- **Seal:** `meta:seal` primary + active segment close + attempt seal metadata in one batch

`InMemoryExecutionLineagePersistence` uses the same partition-atomic snapshot semantics as `PartitionAtomicDocumentStore`.

Forced interleaving tests: `test_execution_lineage_atomic_fault_injection.py` (A1–A5).

## Correction — HostTask composition correction

- `build_host_task_execution(nexus_loop)` passes `nexus_loop.execution_lineage_persistence`
- `HostTaskExecution.execute()` preserves `task_id`, `tenant_id`, and `segment_predecessor_root_execution_id` in `RootExecutionOptions`
- Resume checkpoint identity follows canonical orchestration resume planner semantics

## Correction — Terminal reconciliation

`_commit_durable_terminal_authority()` now idempotently seals lineage after `ExecutionTerminalConflictError` reconciliation when canonical terminal truth is loaded.

## Correction — Degradation lifecycle

Root activation binds `AttemptLineageDegradationState` from durable attempt metadata; root deactivation resets the ContextVar so degradation does not leak across tasks in the same async worker.

## Correction — Nested fail-open decision

**PASS (fail-closed enterprise semantics):** when child admission is unavailable and only `mark_degraded` succeeds, nested children without durable parent admission raise `ExecutionLineageIntegrityError` (`parent admission missing`). No synthetic/orphan lineage is created.

## Correction — New tests

- `test_execution_lineage_atomic_fault_injection.py`
- `test_host_task_lineage_wiring.py`
- `test_nexus_factory_lineage_wiring.py`
- `test_degradation_context_isolation.py`
- `test_nested_child_after_degraded_parent.py`
- `test_terminal_conflict_seal.py`

## Final correction

- **HostTask single root identity:** `resolve_root_task_identity(..., execution_id=...)` mints exactly one canonical root; `HostTaskExecution.execute()` passes `identity.execution_id` to resume plan, revision admission, and `RootExecutionOptions` (no second mint in `ExecutionRuntime`).
- **Explicit provider fail-closed:** `resolve_execution_lineage_persistence(provider=DOCUMENT_STORE, document_store=None)` raises `ExecutionLineageConfigurationError`; `provider=None` remains lineage disabled.
- **Real nested degraded-parent proof:** `test_nested_child_after_degraded_parent.py` uses one persistence object and production `ChildExecutionRunner` nested delegation; nested child without durable parent admission fails closed (`parent admission missing`).
- **Post-open_segment degradation binding:** root activation binds `AttemptLineageDegradationState` from durable attempt metadata after `open_segment` (unclean predecessor resume).

## Final verdict

**PASS** — logical lineage mutations (admission, segment open, unclean successor, seal) are single durable atomic transitions; canonical HostTask path uses production lineage composition without manual injection; resume and provider contracts are fail-closed where required.
