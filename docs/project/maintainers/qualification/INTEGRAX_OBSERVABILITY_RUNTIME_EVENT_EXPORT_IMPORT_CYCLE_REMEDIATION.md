# INTEGRAx-OBSERVABILITY-RUNTIME-EVENT-EXPORT-IMPORT-CYCLE-REMEDIATION

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-OBSERVABILITY-RUNTIME-EVENT-EXPORT-IMPORT-CYCLE-REMEDIATION` |
| Session | Qualification optimization + final canonical certification closure |
| Classification | **Class B** — compatible internal implementation change |
| Date | 2026-09-15 |

## Session Scope

Structural removal of the runtime observability import cycle without changing public contracts, export schemas, event delivery semantics, or execution ownership.

## Confirmed Import Cycle

```text
export_boundary
    → RuntimeEvent (runtime.events)
    → events/__init__ → event_bus
    → event_delivery/__init__
    → runtime_event_delivery
    → export_boundary (runtime_event_export_source_from_event)
```

## Dependency Graph Before

```text
export_boundary
    ↓
RuntimeEvent
    ↓
events/event_bus/event_delivery
    ↓
runtime_event_delivery
    ↓
export_boundary
```

## Ownership Analysis

`runtime_event_export_source_from_event` is **export-domain mapping** (pure `RuntimeEvent` → `RuntimeEventExportSource`). It is consumed by event delivery for payload construction but is not a delivery or persistence concern.

## Considered Designs

| Variant | Summary | Verdict |
| --- | --- | --- |
| A. Lazy import in `runtime_event_delivery` | Defers `export_boundary` import | Symptom suppression; hides dependency |
| B. Dedicated `runtime_event_export_mapping` module | One-way: delivery → mapper → contracts/models | **Selected** |
| C. Mapper inside delivery | Couples export projection to delivery ownership | Rejected |

## Selected Design

Internal module `intergrax/runtime/observability/runtime_event_export_mapping.py` holds the canonical pure mapper. `export_boundary` re-exports `runtime_event_export_source_from_event` for unchanged public surface. `runtime_event_delivery` imports the mapper only.

## Change Classification

**Class B** — public API, schemas, lifecycle, governance, identity, and persistence ownership unchanged.

## Layer Boundary Assessment

No tier violations. Mapper depends on `runtime.events.runtime_event` (direct) and `RuntimeEventExportSource` model from `export_boundary` (defined before late imports).

## Public Contract Assessment

Unchanged: observability facade, `RuntimeEvent`, `RuntimeEventExportSource`, envelope schema, `FORBIDDEN_EXPORT_CONTENT_FIELDS`, safe-payload allowlist semantics.

## Pluginability Assessment

Exporter/provider wiring unchanged; no new concrete exporter coupling.

## Dependency Graph After

```text
RuntimeEvent
    ↓
runtime_event_export_mapping
    ↓
RuntimeEventExportSource

runtime_event_delivery
    ↓
runtime_event_export_mapping

export_boundary (late)
    ↓
runtime_event_export_mapping (re-export)
```

## Changed Files

| Path | Change |
| --- | --- |
| `intergrax/runtime/observability/runtime_event_export_mapping.py` | **New** — pure mapper |
| `intergrax/runtime/observability/export_boundary.py` | Delegate mapper; remove inline implementation |
| `intergrax/runtime/observability/event_delivery/runtime_event_delivery.py` | Import mapper instead of `export_boundary` |
| `tests/unit/runtime/observability/test_runtime_event_export_import_cycle.py` | Subprocess cold-import + static guard |

## Cold Import Verification

```bash
uv run python -c "from intergrax.runtime.observability import InMemoryObservabilityExporter"
```

**PASS**

## Mapping Parity

`test_runtime_event_projection_preserves_semantics` in `test_export_boundary_contracts.py` — **PASS** (identical safe payload and identity fields).

## Export Schema Parity

No changes to `RuntimeEventExportSource`, `ObservabilityExportEnvelope`, or schema version constants.

## Runtime Observability Regression

```bash
uv run pytest tests/unit/runtime/observability/test_export_boundary_contracts.py -q
uv run pytest tests/unit/runtime/observability/ -q
```

Export-boundary contracts: **15 PASS**. Full leaf: **430 PASS** (import-cycle suite included). Parallel-session WIP may fail unrelated bounded-delivery tests outside this change set.

## Runtime Events Regression

```bash
uv run pytest tests/unit/runtime/events/ -q
```

**PASS**

## Qualification Regression

```bash
uv run pytest tests/unit/testing_support/execution_qualification/ -q
```

**PASS**

## Full Canonical Verification

One `npsc5f-final` run (`repetitions=1`, `max_parallel=2`) executed after remediation. Import-cycle and `runtime-observability` leaf green; profile may remain **blocked** on unrelated drift / parallel WIP (evidence, bounded delivery).

## Performance Observation

From measured `npsc5f-final` sample (post-remediation worktree):

| Metric | Value |
| --- | --- |
| Wall | ~160.1 s |
| Slowest leaf | `runtime-observability` ~51.3 s |
| `runtime-events` wall (suite) | ~24.6 s |
| `npsc5e-r2.mandatory.r2-h2-q1` | ~17.2 s |
| `npsc5f-final.recovery` | ~14.3 s |
| Total leaf work | ~319.8 s |
| Effective concurrency | ~2.0 |

## Production Changes

Minimal: one internal mapping module + two import-path updates.

## Findings

Root cause was module-level coupling of event delivery to `export_boundary` while `export_boundary` eagerly imported `RuntimeEvent` through the events package `__init__`, completing a cycle on cold import.

## Decision

Proceed with **Design B**; no architecture boundary change required.

## Final Verdict

**OBSERVABILITY IMPORT-CYCLE REMEDIATION = PASS**

**FULL CANONICAL CLOSURE = BLOCKED BY NEW LEAF** (protected drift / unrelated parallel WIP — out of scope for this task).
