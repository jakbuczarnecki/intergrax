# DIAGNOSTIC-ENGINE-EXECUTION-FAILURE-EVIDENCE-R1 (R2)

## Scope

Production slice: real child delegate failure → durable `RuntimeEventType.EXECUTION_FAILED` on the canonical bus → central `ExecutionFailureAnalyzer` → conservative Problem grouping → operator read.

## Frozen R1 architecture dependency

`ae3e9162d22e0d22fbc2dcdb1c84c9c0ad7dbbb8` — single diagnostic authority, single RuntimeEvent spine, OPTION B evidence composition.

## As-built before R2

Child failures surfaced as `ChildExecutionFailedError` / coordination codes without execution-scoped durable forensic events.

## Production implementation

- Port: `ExecutionFailureEvidenceRecorder` (`intergrax/contracts/execution_failure_evidence.py`)
- Adapter: `RuntimeEventExecutionFailureEvidenceRecorder`
- Active context: `ActiveExecutionEvidenceContext` (no `execution_id`; taken from active boundary)
- Emission: `ExecutionFailureRecordingDelegate` inside `ChildExecutionRunner` / `ExecutionBoundary`
- Wiring: `ExecutionRuntime`, `execute_root_task`, `build_host_task_execution` → existing `RuntimeEventBus`

## Evidence identity

Typed `ExecutionFailurePayloadV1` (`execution_failure.v1`), `ExecutionFailureKind.DELEGATE_EXCEPTION`, bounded `safe_summary`, no raw exception fields.

## Persist-before-propagate proof

Delegate `except Exception` path records via bus before `raise`; ordering covered in unit/integration harness tests.

## Persistence outage semantics

`MandatoryEvidencePersistenceError` → `UNAVAILABLE` result; original delegate exception still propagates.

## Central analyzer

`ExecutionFailureAnalyzer` — `DiagnosticFindingKind.EXECUTION_FAILED`, `DiagnosticPrecision.EXECUTION_LEVEL`, `FailureBoundary` with `DiagnosticCertainty.PROVEN`.

## Problem lifecycle integration

Deterministic grouping strategy version `2` includes `execution_id` + `execution_failure_kind` in signature (conservative per-failure merge).

## Operator read

`DiagnosticReadService` lists Problems/Occurrences after terminal trigger; DG-001 P3 central failure path.

## Nested execution proof

DG-001 harness + coordination qualification; multiple `EXECUTION_FAILED` events possible per nested boundary (no causal inference).

## Legacy semantics

Lifecycle-only findings unchanged; `source_anomaly_kind` remains required for lifecycle findings.

## Security/redaction

No exception message/traceback in payload; bounded string fields.

## Regression matrix

`tests/unit/runtime/execution/test_execution_failure_evidence_r2.py`, DG-001 P3, event catalog tests, diagnostic assessment suite.

## Known limitations

No R3 causal root-cause graph; duplicate `EXECUTION_FAILED` may appear when both Nexus graph child and delegated specialist child fail in one run.

## Final verdict

IMPLEMENTED / CORRECTION_REQUIRED → correction landed; qualification PASS pending full regression batch completion in operator environment.

**Implementation commit:** `40cc8c11e` (R2 base).

**Correction commit:** (record after `fix(diagnostics): harden execution failure evidence R2`).

Prior state `40cc8c11e`: IMPLEMENTED / CORRECTION_REQUIRED (private Nexus terminal API, false PERSISTED without store, persistence signature compatibility, insufficient exact-child P3, typing gaps, matrix not executed).
