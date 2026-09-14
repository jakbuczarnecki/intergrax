# ADR-OBS-005: Runtime event delivery failure contract (EventSinkPort)

| Field | Value |
|-------|-------|
| **Status** | Proposed — design freeze for P1B-R3 (audit required before implementation) |
| **Date** | 2026-09-14 |
| **Deciders** | Platform observability / VPI platform evolution |
| **Related** | `intergrax/contracts/event_delivery.py` · ADR-OBS-001 · VPI-PLATFORM-EVOLUTION-P1B-D1 · P1B-R3 |

## Context

W5-A introduced `EventSinkPort.publish(...) -> EventDeliveryResult` with dispositions `ACCEPTED`, `DROPPED`, `REJECTED`, `DEFERRED`. Durable evidence (Plane A) uses `EvidencePersistencePort` and `MandatoryEvidencePersistenceError`. Observability delivery (Plane B) and subscriber handlers (Plane C) are separate.

P1B wired stage signals through `RuntimeEventBus` -> optional `EventSinkPort`. Post P1B-R2 audit (`f7749d0a509bc1c86dedd6e34975a22aa1c01607`) identified a contract gap: infrastructure failures before a controlled `EventDeliveryResult` are not standardized; vendor exceptions can escape plugin boundaries. VPI exposed the gap; the fix is scenario-neutral. VPI continues to observe only `ApplicationExecutionStageSignalEmissionError` at its boundary.

### Current path

```text
RuntimeEvent -> RuntimeEventBus (persist) -> EventSinkPort.publish
  -> often BoundedEventSink -> RuntimeEventExportSink -> EventExportSinkPort
-> subscribers
```

### Implementation inventory

| Implementation | Contract | Raw error risk | R3 migration |
|----------------|----------|----------------|--------------|
| AcceptingObservabilityEventSink | EventSinkPort | Low | Document behavioral contract |
| InMemoryEventSink | EventSinkPort | Low | Same |
| BoundedEventSink | EventSinkPort | CriticalEventDeliveryError on full buffer; drain worker may leak downstream exceptions | Normalize downstream failures |
| RuntimeEventExportSink | EventSinkPort bridge | Broad except -> result | Narrow to ExportError family |
| NoopEventExportSink | EventExportSinkPort | None | None |
| RecordingEventExportSink | EventExportSinkPort | None | None |
| OtlpEventExportSink | EventExportSinkPort | OtlpTransportError normalization | Align docs |

## Problem

1. No public typed sink failure on `EventSinkPort`.
2. `RuntimeEventBus._deliver_through_event_sink` does not handle boundary failures.
3. External plugins lack stable errors after vendor translation.
4. Export layer has `ExportError`; must stay coherent without merging persistence failures.

## Design options

### Option A — Single EventDeliveryBoundaryError + EventDeliveryFailureKind

Minimal; policy uses priority + single except type; kinds for metrics only.

### Option B — Small hierarchy (Transport / Unavailable)

Rejected: no distinct platform policy per type today.

### Option C — Result-only failures

Rejected: conflicts with CriticalEventDeliveryError and blurs controlled policy vs broken mechanism.

## Decision

Adopt **Option A**: add `EventDeliveryFailureKind` (TRANSPORT, UNAVAILABLE, SINK_INTERNAL) and `EventDeliveryBoundaryError` in `intergrax/contracts/event_delivery.py`.

### Result vs exception

| Case | Outcome |
|------|---------|
| Policy-complete drop/reject/defer | EventDeliveryResult |
| Cannot complete publish | EventDeliveryBoundaryError (plugin) |
| CRITICAL + DROPPED/REJECTED result | CriticalEventDeliveryError (RuntimeEventBus) |
| CRITICAL + boundary error | CriticalEventDeliveryError chained (RuntimeEventBus) |
| BEST_EFFORT + boundary error | tolerate + metrics (RuntimeEventBus) |

EventSinkPort plugins MUST NOT raise CriticalEventDeliveryError. BoundedEventSink may raise it for buffer rules.

### CriticalEventDeliveryError

Platform fail-closed policy — not vendor transport. Not raised by external EventSinkPort plugins.

### Export layer

EventExportSinkPort failures -> ExportError / OtlpTransportError -> RuntimeEventExportSink maps to dispositions. Do not merge with EventDeliveryBoundaryError.

Normalization: vendor SDK -> plugin -> public platform error; never vendor SDK -> RuntimeEventBus.

### Pluginability

RuntimeEventBus depends only on EventSinkPort (injection). Composition: `runtime_event_delivery_wiring.py`, `EventExportSinkFactoryPort`. Core plugin registry not required (OPTION A: constructor injection; optional discovery FUTURE).

### Lifecycle

publish + close sufficient — NO CHANGE REQUIRED.

### Ownership matrix

| Concern | Owner |
|---------|--------|
| Delivery contract | Platform contracts |
| Delivery result | Platform contracts |
| Failure taxonomy | EventDeliveryBoundaryError |
| Priority | Platform |
| Critical fail-closed | RuntimeEventBus, BoundedEventSink |
| Vendor translation | Plugin |
| Buffering | BoundedEventSink |
| Retry | FUTURE — separate from EventSinkPort |
| Persistence | Evidence subsystem |
| Business reaction | ApplicationExecutionStageSignalEmissionError |

### Strategy extension points

| Area | Status |
|------|--------|
| Priority | EXISTING |
| Overflow | EXISTING |
| Failure policy per priority | EXISTING (bus) |
| Retry | FUTURE |
| Sink selection | EXISTING wiring |
| EventDeliveryFailurePolicy | FUTURE if tenants diverge |

### Governance

Reuse observability export profile + composition wiring. Third-party sink allowlist via manifest: SEPARATE DESIGN REQUIRED if needed.

### Execution Engine

No new ids — PASS.

### Diagnostics

InternalDeliveryMetrics + logging; no recursive RuntimeEvent through failing sink — YES.

### Security

Sanitized public messages; chain __cause__ internally.

### VPI

Only ApplicationExecutionStageSignalEmissionError at scenario boundary.

### Versioning

Additive exceptions — SOURCE-COMPATIBLE, BEHAVIORALLY HARDENED.

### Migration

Phase 1 contract · 2 sinks · 3 bus · 4 emitter mapping · 5 VPI proof.

### R3 files (not this task)

event_delivery.py, event_bus.py, bounded_event_sink.py, runtime_event_export_sink.py, tests, optional application_execution_stage_signal.py phase 4.

### Contract tests (R3)

SUCCESS, DROPPED, REJECTED, boundary error, no vendor leak, critical policy, best effort, plugin swap, import gates, VPI boundary.

### External plugin contract

Implement publish/close; return results for policy outcomes; raise only EventDeliveryBoundaryError; never vendor or CriticalEventDeliveryError; no secrets in messages; document thread-safety if shared.

## Non-goals

Code in D1; retry; governance; VPI adoption; Kafka/OTLP impl.

## Consequences

Vendor-neutral bus; R3 migration burden on sinks and bus.

## Rollback

Remove types and bus handling.

## Compliance

Tier boundaries; persistence != delivery; scenario-neutral.

## Enterprise design review

1–10: YES. Verdict PASS pending GitHub audit.
