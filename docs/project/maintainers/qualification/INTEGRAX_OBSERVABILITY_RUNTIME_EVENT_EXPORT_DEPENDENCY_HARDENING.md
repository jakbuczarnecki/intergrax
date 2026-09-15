# INTEGRAx-OBSERVABILITY-RUNTIME-EVENT-EXPORT-DEPENDENCY-HARDENING

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-OBSERVABILITY-RUNTIME-EVENT-EXPORT-DEPENDENCY-HARDENING` |
| Session | Qualification optimization + final canonical certification closure |
| Classification | **Class B** — internal ownership / dependency cleanup |
| Date | 2026-09-15 |

## Session Scope

Remove the remaining structural edge `runtime_event_export_mapping → export_boundary` while preserving public observability export contracts, mapper semantics, and cold-import behavior.

## Previous Functional Fix

Commit `9c710fc98` introduced `runtime_event_export_mapping` and routed `runtime_event_delivery` through the mapper instead of `export_boundary`, restoring cold import of `InMemoryObservabilityExporter`. The mapper still imported `RuntimeEventExportSource` from `export_boundary`, leaving a logical cycle in the dependency graph.

## Remaining Structural Dependency

```text
export_boundary → runtime_event_export_mapping → export_boundary
```

## Dependency Graph Before

```text
export_boundary
    ↓
runtime_event_export_mapping
    ↓
export_boundary
```

```text
runtime_event_delivery
    ↓
runtime_event_export_mapping
```

## Ownership Analysis

`RuntimeEventExportSource` is a frozen Pydantic export-domain model (OBS-EXPORT-2), not boundary orchestration, delivery, or persistence. It belonged in neutral model ownership separate from `export_boundary` implementation wiring.

## Selected Model Ownership

`intergrax/runtime/observability/runtime_event_export_models.py` — single canonical definition of `RuntimeEventExportSource`.

## Change Classification

**Class B** — public API paths unchanged (re-export from `export_boundary`), schema fields unchanged, mapping allowlist unchanged, delivery lifecycle unchanged.

## Public API Compatibility

`from intergrax.runtime.observability.export_boundary import RuntimeEventExportSource` and `runtime_event_export_source_from_event` remain valid via model re-export and end-of-module mapper re-export.

## Pluginability Assessment

No new coupling to concrete exporters, sinks, or persistence. Mapper remains pure projection.

## Dependency Graph After

```text
runtime_event_export_models
        ↑
        │
runtime_event_export_mapping
        ↑
        │
runtime_event_delivery
```

```text
export_boundary
    ↓
runtime_event_export_models
    ↓
runtime_event_export_mapping   (re-export at end of export_boundary)
```

No edge from `runtime_event_export_mapping` or `runtime_event_delivery` back to `export_boundary`.

## Structural Import Guards

AST guards in `tests/unit/runtime/observability/test_runtime_event_export_import_cycle.py`:

- `runtime_event_export_mapping` must not import `export_boundary`
- `runtime_event_delivery` must not import `export_boundary`
- `runtime_event_export_models` must not import `export_boundary` or `runtime_event_delivery`
- subprocess import-order independence (mapper ↔ boundary)

## Mapping Parity

Unchanged: event_id, run_id, task_id, attempt_id, execution_id, event_type, agent_id, tenant_id, correlation_id, timestamp (`occurred_at`), execution phase, parent event, traceparent/tracestate, safe payload allowlist `_SAFE_RUNTIME_EVENT_PAYLOAD_KEYS`, bool-not-int filtering.

## Schema Parity

`RuntimeEventExportSource` moved verbatim to `runtime_event_export_models.py` (same fields, validators, schema version literal).

## Cold Import Matrix

| Command | Result |
| --- | --- |
| `from intergrax.runtime.observability import InMemoryObservabilityExporter` | PASS |
| `from intergrax.runtime.observability.export_boundary import RuntimeEventExportSource` | PASS |
| `from intergrax.runtime.observability.runtime_event_export_mapping import runtime_event_export_source_from_event` | PASS |
| Import order mapper → boundary | PASS |
| Import order boundary → mapper | PASS |

Mapper module load avoids eager `RuntimeEvent` import (TYPE_CHECKING-only type reference) so delivery → mapper does not recurse through partially initialized mapping during `events` package initialization.

## Changed Files

- `intergrax/runtime/observability/runtime_event_export_models.py` (new)
- `intergrax/runtime/observability/runtime_event_export_mapping.py`
- `intergrax/runtime/observability/export_boundary.py`
- `tests/unit/runtime/observability/test_runtime_event_export_import_cycle.py`
- `docs/project/maintainers/qualification/INTEGRAX_OBSERVABILITY_RUNTIME_EVENT_EXPORT_DEPENDENCY_HARDENING.md`

## Runtime Observability Regression

`uv run pytest tests/unit/runtime/observability/ -q` → **439 passed**.

## Runtime Events Regression

Included in combined qualification batch → **pass** (see Qualification Regression).

## Qualification Regression

`uv run pytest tests/unit/testing_support/execution_qualification/ -q` (with embedded harness modules) → **334 passed**.

## Full Canonical Verification

One `npsc5f-final` run (`repetitions=1`, `max_parallel=2`) at git `68adbe244` (pre-hardening commit on branch; hardening uncommitted during first run):

- `run_status_pass`: **false**
- `certification_decision`: **blocked**
- Slowest leaf: `npsc5e-r3.mandatory.r3-implementation-gate` (~390 s) — `R2 Final` mandatory frozen suite failures (checkpoint/durable resume qualification)
- `npsc5f-final.drift-sentinel`: Evidence Plane BREAKING drift (broad observability / event_delivery surface vs baseline)

## Protected Drift Assessment

Post-hardening commit will add `runtime_event_export_models.py` and touch `export_boundary` / mapping — **EXPECTED PROTECTED DRIFT — BASELINE REVIEW REQUIRED** for drift sentinel; not auto-updated in this task.

## Performance Observation

Measured sample (pre-hardening HEAD `68adbe244`):

| Metric | Value |
| --- | --- |
| Wall | ~493.7 s |
| Slowest leaf | `npsc5e-r3.mandatory.r3-implementation-gate` ~390.1 s |
| `runtime-observability` | ~48.3 s |
| `runtime-events` | ~20.8 s |
| `npsc5e-r2.mandatory.r2-h2-q1` | ~13.6 s |
| `npsc5f-final.recovery` | ~26.3 s |
| Total leaf work | ~702.2 s |
| Effective concurrency | ~1.42 |

## Production Changes

Neutral model module; mapper imports model only; `export_boundary` imports model at top and re-exports mapper at module end (after existing intentional late-import block).

## Findings

- Structural acyclicity for mapper ↔ boundary achieved via model extraction.
- Cold mapper import required deferring eager `RuntimeEvent` module import in mapping (annotation-only) to break `events` package → `event_delivery` → mapper re-entrancy.
- Full `npsc5f-final` green blocked by unrelated R2 Final gate failures and protected drift on parallel observability WIP — not by targeted hardening tests.

## Decision

Proceed with Class B hardening commit; defer full canonical PASS until R2 gate / drift baseline governance closes.

## Final Verdict

**OBSERVABILITY RUNTIME EVENT EXPORT DEPENDENCY HARDENING = PASS** (targeted scope)

**FULL CANONICAL VERIFICATION = BLOCKED BY NEW LEAF** (`npsc5e-r3.mandatory.r3-implementation-gate` / R2 Final) and **EXPECTED PROTECTED DRIFT** on drift sentinel.
