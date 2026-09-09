# DG-001 — Execution lineage admission persistence R1

> **Task:** `DG-001-MULTI-AGENT-EXECUTION-LINEAGE-ADMISSION-PERSISTENCE-R1`  
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
- `AttemptLineageDegradationState` (runtime ContextVar)

## Provider strategy

- **R1 durable:** `DocumentStoreExecutionLineagePersistence` over `PartitionAtomicDocumentStore`
- **Tests / single-process:** `InMemoryExecutionLineagePersistence`
- **KV adapter:** NOT_IMPLEMENTED (by design)
- **Composition:** `resolve_execution_lineage_persistence(explicit_persistence=...)`

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

## Verdict

**PASS** — write-side durable admission lineage implemented with production composition hooks and focused regression coverage.
