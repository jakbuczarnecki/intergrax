# ADR-OBS-001: Harness Observability Spine (unified bus)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-08 |
| **Deciders** | Harness platform |
| **Related** | [`OBSERVABILITY_ARCHITECTURE.md`](../OBSERVABILITY_ARCHITECTURE.md) · [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md) §33 · [Phase OBS-BUS](../INTERGRAX_IMPLEMENTATION_PLAN.md#phase-obs-bus--unified-observability-spine) |

## Context

Intergrax requires end-to-end observability from user intake to final response across Harness (Tier-0/1), applications (Tier-3), and agents (Tier-2). Operators must reconstruct any interaction without reading agent source code (canon §42.24 inspectability guarantee).

Prior iterations delivered:

- `TraceEvent` + `DiagnosticPayload` (typed trace plane)
- `RuntimeEvent` + `RuntimeEventBus` (canonical event plane)
- `trace_bridge` + `build_unified_run_journal` (read-model merge)
- Tier-3 `wire_application_observability` (profile wiring)

Gaps identified in the 2026-06-08 audit:

- Dual-path emission (trace vs bus) without a single developer-facing API
- `RuntimeEvent.payload` remains `Dict[str, Any]`
- Incomplete catalog emission (`AGENT_SELECTED`, `STEP_FAILED`, causal `parent_event_id`)
- No formal extension contract for agent/application custom steps beyond `DiagnosticPayload` namespace rules

Alternatives considered:

1. **External APM only (Datadog/OpenTelemetry as source of truth)** — rejected: violates event-first canon, couples operators to vendor, loses typed harness semantics.
2. **Trace-only (drop RuntimeEvent bus)** — rejected: ops filtering, policy hooks, and middleware require canonical event types.
3. **Per-tier observability stacks** — rejected: duplicates wiring, breaks unified journal, violates Harness-as-product principle.
4. **Harness Observability Spine (HOS)** — **accepted**: one bus, typed extension, pluggable persistence, unified read model.

## Decision

Adopt the **Harness Observability Spine (HOS)** as the single observability mechanism for all tiers:

1. **Event-first:** `RuntimeEvent` is the primary audit signal; `TraceEvent` carries rich `DiagnosticPayload` detail; metrics are derived.
2. **Single emit API (target):** `ObservabilityEmitter` facade wraps `RuntimeState.trace_event`, `RuntimeEventBus.record`, and `RunTraceWriter.append` — developers do not choose stores.
3. **Typed extension:** All tiers extend `DiagnosticPayload` with stable `schema_id`; agent schemas use `agents.<slug>.diag.*`, applications use `applications.<slug>.diag.*`.
4. **TraceScope (target):** Context manager sets `parent_event_id` for causal trees.
5. **Typed canonical payloads (target):** `RuntimeEventPayload` registry replaces raw dicts at the bus layer.
6. **Unified read model:** `build_unified_run_journal` remains the operator timeline; external sinks subscribe or dual-write from the journal.
7. **Wiring unchanged at Tier-3:** `wire_application_observability` — no per-product trace stores.

## Consequences

### Positive

- One mental model for harness, application, and agent authors
- Full-flow reconstruction from a single journal per run
- Modular scale-out (SQLite → Cassandra) without changing emitters
- CI gates can enforce catalog emission and schema registration
- Aligns with large-system patterns: event sourcing spine + structured diagnostic spans + metrics sinks

### Negative

- Migration effort for `RuntimeEvent.payload` typing (OBS-BUS-1)
- Short-term dual emit paths remain until `ObservabilityEmitter` ships (OBS-BUS-2)
- Agent authors must learn `DiagnosticPayload` contract (documented in OBSERVABILITY_ARCHITECTURE.md §5.3)

## Compliance

- Tier boundaries preserved — Tier-2 does not own stores; Tier-3 wires, does not reimplement
- PII redaction remains at `DiagnosticPayload.redact()` boundary
- Linked: `OBSERVABILITY_ARCHITECTURE.md`, canon §33 pointer, `INTERGRAX_IMPLEMENTATION_PLAN.md` Phase OBS-BUS

## Implementation notes

- Architecture doc: `docs/OBSERVABILITY_ARCHITECTURE.md`
- Implementation tracker: Phase OBS-BUS (OBS-BUS-0 through OBS-BUS-7)
- Verification: `scripts/check_observability_gates.py` (OBS-BUS-7 Done)
