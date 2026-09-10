# NPSC-5F — Execution Evidence, Replay & Observability Architecture

> **Status:** P0 reconciliation (inventory + contracts; no new evidence framework)  
> **Frozen execution baseline:** NPSC-5E Final `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7`  
> **Canonical domain doc:** [`docs/project/architecture/OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md)  
> **P0 qualification:** [`NPSC_5F_P0_EXECUTION_EVIDENCE_ARCHITECTURE_RECONCILIATION.md`](../qualification/NPSC_5F_P0_EXECUTION_EVIDENCE_ARCHITECTURE_RECONCILIATION.md)

## Scope

Reconcile the existing Harness Observability Spine (HOS), `RuntimeEvent` persistence, Unified Run Journal, export boundary, causal evidence, and diagnostic reconstruction with frozen NPSC-5E execution owners. Define enterprise evidence-plane direction without implementing a second event bus or moving execution authority into observability.

## Non-goals (P0)

- New `ExecutionEvidenceBus`, `ReplayEngine`, or production EventStore
- Changing NPSC-5E lifecycle, retry, checkpoint, partial recovery, lineage, terminal, governance, or authority semantics
- Remediating all audit findings (documented as blockers/gaps)

## Core principle (frozen candidate)

| Plane | Role |
| ----- | ---- |
| **Execution** | Side-effect-capable runtime; mints/coordinates lifecycle |
| **Execution evidence** | Immutable durable facts about what happened (`RuntimeEvent` + related persisted envelopes) |
| **Observability** | Projections/signals derived from execution/system behavior (metrics, traces, export) |
| **Diagnostics** | Analysis over evidence/observability (`execution_reconstruction`, Problems) |
| **Replay taxonomy** | (A) reconstruction — no execution; (B) simulation/re-evaluation — sandboxed; (C) active re-execution — **new** Execution through current governance |

**Invariant:** evidence may prove and reconstruct history; it must never decide whether execution may run, mint lineage, resume checkpoints, retry, recover slots, or apply policy.

## Existing architecture inventory (material)

| Component | Path | Role | Authoritative? | Durable? | Ordering scope | Tenant-aware? | Typical consumer |
| --------- | ---- | ---- | -------------- | -------- | -------------- | ------------- | ---------------- |
| `RuntimeEvent` | `intergrax/runtime/events/runtime_event.py` | Canonical execution envelope | Yes (when persisted) | When accepted by store | N/A (identity) | Optional field | Bus, persistence, journal |
| `EventId` | `intergrax/contracts/execution_identity.py` | Event identity | Yes | With event row | Global per accepted id | Via persistence scope | Idempotent append |
| `RuntimeEventBus` | `intergrax/runtime/events/event_bus.py` | In-process transport + optional persist hook | No (transport) | No | N/A | Via `resolve_event_tenant_id` | Hooks, metrics, `publish`/`record` |
| `RuntimeEventPersistence` | `intergrax/runtime/events/persistence_contract.py` | Durable evidence port | **Yes** (accepted events) | Yes | Per `(tenant, run)` position | Yes | Nexus, debug, journal, DIAG |
| `SQLiteRuntimeEventStore` | `intergrax/runtime/events/stores/sqlite_runtime_event_store.py` | Default durable adapter | Delegates to contract | Yes | `ExecutionEventPosition` per run | Yes | Hosts, proofs |
| `InMemoryRuntimeEventStore` | `intergrax/runtime/events/stores/memory_runtime_event_store.py` | Test / ephemeral | Same contract | Process-local | Per run | Yes | Unit tests |
| `DocumentBackedRuntimeEventStore` | `intergrax/runtime/events/stores/document_backed_runtime_event_store.py` | Document-store adapter | Same contract | Yes | Per run + global event_id partition | Yes | Integration |
| `ValidatingRuntimeEventPersistence` | `intergrax/runtime/events/stores/validating_runtime_event_store.py` | Schema guard wrapper | Delegates | Delegates | Delegates | Delegates | Production wiring |
| `reconcile_idempotent_event_acceptance` | `persistence_contract.py` | EventId + content equivalence | Yes | N/A | N/A | Tenant match | All stores |
| `ExecutionEventPosition` | `intergrax/runtime/events/execution_position.py` | Run-local sequence | Yes within run | Yes | **Per tenant + run** | Yes | Journal, as-of |
| `build_unified_run_journal` | `intergrax/runtime/events/unified_run_journal.py` | Derived read model | No (projection) | No | Delegates to store | Yes | Export, inspect |
| `load_positioned_run_journal_through` | same | Prefix completeness authority | No (read helper) | N/A | Run prefix | Yes | As-of, reconstruction |
| `RunExecutionAsOfProjection` | `intergrax/runtime/events/asof_projection.py` | Historical view at **E** | No | N/A | Run | Yes | DIAG, audit |
| `ObservabilityExportEnvelope` | `intergrax/runtime/observability/export_boundary.py` | Redacted export contract | Export view | N/A | N/A | Yes | Integrations |
| `export_bridge` / `export_routing` | `intergrax/runtime/observability/` | Safe export path | No | N/A | N/A | Yes | Vendor backends |
| `journal_export` | `intergrax/runtime/observability/journal_export.py` | Journal OTLP/log snapshot | **Bypass risk** | N/A | Uses journal limit | Yes | Default runtime plugin |
| `PlatformCausalEvidence` | `intergrax/runtime/observability/causal_evidence.py` | Cross-boundary causal facts | Separate plane | Via `CausalEvidencePersistence` | Causal query order | Yes | DIAG |
| `CausalEvidencePersistence` | `intergrax/runtime/observability/causal_evidence_persistence.py` | Causal durable store | Causal only | Yes | Typed query key | Yes | Reconstruction |
| `RunTraceWriter` / Plane B | `intergrax/runtime/nexus/tracing/` | Diagnostic trace detail | No | Optional | Trace seq | Yes | Operator detail |
| `ExecutionReconstructionService` | `intergrax/runtime/diagnostics/execution_reconstruction.py` | Read-only lineage+events view | No | N/A | N/A | Yes | DIAG read APIs |
| `ExecutionLineagePersistence` | `intergrax/runtime/execution/lineage/` | Ancestry/provenance | **Yes (lineage)** | Yes | Admission position | Yes | Runtime, DIAG (not evidence store) |

## Canonical owners (unchanged from 5E)

| Concern | Owner |
| ------- | ----- |
| Execution lifecycle | `ExecutionRuntime` |
| Attempt transitions | `AttemptLifecycleService` |
| Retry | `ExecutionAttemptRetryService` (R1) |
| Durable resume | `LongRunningCoordinator` / `RuntimeCheckpoint` (R2) |
| Checkpoint revision CAS | `TaskCheckpointPersistence` |
| Partial recovery | `FanOutPartialRecoveryService` (R3) |
| Scheduling / topology | Nexus |
| Child execution | `ChildExecutionPort` |
| Lineage | `ExecutionLineagePersistence` |
| Terminal truth | `ExecutionTerminalService` |
| Governance / authority | Governance plane + effective authority |
| Durable execution **facts** | `RuntimeEventPersistence` (observability spine) |
| Event **transport** | `RuntimeEventBus` |
| Export redaction (when used) | `ObservabilityExportEnvelope` / `export_boundary` |
| Run journal read model | `build_unified_run_journal` (derived) |
| Prefix-complete as-of reads | `load_positioned_run_journal_through` |

## Event taxonomy (P0)

| Class | Examples |
| ----- | -------- |
| **A — Authoritative durable execution evidence** | Persisted `RuntimeEvent` lifecycle transitions (step/task/attempt boundaries on canonical path) |
| **B — Audit/security evidence** | Authority/governance **references** on events; causal evidence records; export audit metadata |
| **C — Operational telemetry** | Bus history before persist; subscriber dispatch; exporter health |
| **D — Metrics** | Aggregates downstream of events |
| **E — Diagnostic-only** | `TraceEvent` / Plane B rows, `PlatformProblemSignal` |
| **F — Application/domain** | Product-specific payloads inside allowed envelope fields |

## Durability model

| State | Meaning |
| ----- | ------- |
| **emitted** | Handler/bus received event |
| **persisted** | `RuntimeEventPersistence.append` returned position |
| **exported** | Passed export boundary (or journal path) |
| **projected** | Journal, as-of, metrics, DIAG view |
| **consumed** | Subscriber/export sink acknowledged |

Transport (`RuntimeEventBus`) ≠ durable commit. Today, persistence failure in `_store_event` is **logged and swallowed** — execution path may continue (**FAIL-OPEN** for mandatory evidence).

### Mandatory durable (target classification)

Lifecycle admission facts, attempt boundaries, terminal transitions, governance decision **references**, checkpoint/retry/recovery **references**, child lineage **references**, side-effect authorization **references** — persisted as typed `RuntimeEvent` (or causal envelope) when emit path uses `should_persist_event`.

### Best-effort

Metrics, in-memory bus history, optional trace rows, exporter delivery, default journal plugin when persistence disabled.

## Ordering scopes

- **`ExecutionEventPosition`:** monotonic per `(tenant_id, run_id)`; allocated atomically in SQLite (`BEGIN IMMEDIATE` + sequence table).
- **Not task-global:** `list_for_task` orders by `execution_position` across runs — positions restart per run (**documented gap** for cross-run chronology).
- **As-of prefix:** `load_positioned_run_journal_through` fail-closed on truncation (`PositionedJournalPrefixTruncatedError`).

## Idempotency

- **Current:** `reconcile_idempotent_event_acceptance` — same `EventId` requires identical canonical `RuntimeEvent` equality; tenant scope mismatch fails closed.
- **Tests:** `tests/unit/runtime/events/test_event_id_persistence_semantics.py` (DG-002 R1).

## Tenant isolation

- Reads scoped by persistence `tenant_id`; wrong tenant → empty/`None`.
- **Gap:** `resolve_event_tenant_id` prefers explicit route tenant without requiring `event.tenant_id` equality when both set — index tenant may differ from deserialized event tenant.

## Schema evolution

- `RuntimeEvent.schema_version` + `schema_guard` / `ValidatingRuntimeEventPersistence`.
- Unknown payload registry versions fail closed in registry tests; persisted unknown versions require explicit reconstruction handling (**partial**).

## Redaction / export

- Canonical path: `ObservabilityExportEnvelope`, `runtime_event_export_source_from_event`, forbidden field sets in `export_boundary.py`.
- **Gap:** `journal_export.serialize_runtime_event` uses full `model_dump` — bypasses redaction boundary (audit finding 03).

## Reconstruction & replay

| Mode | Mechanism | Executes? |
| ---- | --------- | --------- |
| **Reconstruction** | `ExecutionReconstructionService`, `RunExecutionAsOfProjection`, journal loaders | No |
| **Simulation** | Policy/decision re-evaluation over frozen inputs (domain-specific) | No external effects unless sandboxed |
| **Active re-execution** | New `Execution` via boundary + governance | Yes — not “evidence replay” |

Reconstructed views must remain read-only — never call execution methods on hydrated runtime objects.

## Interactions with 5E planes

- **Lineage:** evidence may reference lineage IDs; `ExecutionLineagePersistence` remains sole ancestry authority.
- **Checkpoint:** evidence records checkpoint refs; R2 stores resume state.
- **Terminal:** evidence records transitions; `ExecutionTerminalService` owns terminal truth.
- **Retry/R3:** evidence explains transitions; owners unchanged.

## Security model (P0)

- Fail-closed for **mandatory** execution evidence on the bus (NPSC-5F/R1); best-effort/debug signals remain explicit.
- No raw secret fields in export allowlists; journal export path is a **P0 blocker** until aligned.
- Query/export access control: governed at product host layer (not reimplemented in P0).

## Concurrency / backpressure

- SQLite: transactional position allocation; concurrent append tests in `test_execution_position_asof.py`, `test_event_id_ownership_crash_recovery.py`.
- Bus: no unbounded durable queue; mandatory persistence failure fails the record/publish boundary (R1); scale/backpressure hardening is Session C.

## Known gaps / enterprise blockers

See qualification doc for severity. Summary:

| ID | Topic | State |
| -- | ----- | ----- |
| OBS-01 | Bus fail-open on persist error | **FIXED by R1** (mandatory tier fail-closed) |
| OBS-02 | EventId content conflict | **FIXED** (reconcile + tests) |
| OBS-03 | Journal export raw `model_dump` | STILL PRESENT |
| OBS-04 | `build_unified_run_journal` silent truncation | STILL PRESENT |
| OBS-05 | Route vs event tenant mismatch | **FIXED by R1** (equality enforced, zero write) |
| OBS-06 | Task ordering via run-local position | STILL PRESENT |

## Implementation roadmap (proposed)

1. **5F/R1** — **Done:** durable evidence contract hardening (`evidence_durability.py`, bus fail-closed mandatory tier, tenant equality at `resolve_event_tenant_id` / store scope).
2. **5F/R2** — Journal completeness, scoped ordering documentation/API (`is_complete` / pagination), gap detection.
3. **5F/R3** — Governed export: align `journal_export` with `ObservabilityExportEnvelope`; remove raw payload bypass.
4. **5F/R4** — Reconstruction quality model, as-of/bitemporal public query alignment (build on TRACE slices).
5. **5F Final** — Platform evidence plane qualification & freeze.

## Flow map (repository)

```text
ExecutionRuntime / Nexus / delegates
  → RuntimeEvent emission (HOS, NexusRuntimeEventPublisher, lifecycle hooks)
  → RuntimeEventBus.publish / record
  → should_persist_event → evidence_persistence_requirement → durable append → subscribers
  → RuntimeEventPersistence.append (scoped tenant, run position)
  → list_positioned_for_run / load_positioned_run_journal_through
  → build_unified_run_journal (derived)
  → ObservabilityExportEnvelope (canonical export) OR journal_export (parallel path — gap)
  → ExecutionReconstructionService / DIAG read APIs
```
