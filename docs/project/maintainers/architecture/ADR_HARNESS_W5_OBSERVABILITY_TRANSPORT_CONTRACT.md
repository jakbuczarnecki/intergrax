# ADR — HARNESS-W5-P0: Observability Transport Contract (OBS-EXPORT-TRANSPORT-CONTRACT-D1)

| Field | Value |
|-------|-------|
| **Status** | **Proposed** — architecture decision for HARNESS-W5-R1 (no production change in P0) |
| **Date** | 2026-09-26 |
| **Task** | HARNESS-W5-P0 — Observability Transport Contract Decision & Closed-World Recertification Baseline |
| **Parent** | HARNESS-W5 — Events / Observability delivery and export recertification |
| **Related** | ADR-OBS-005 (P1B-R3) · OBS-EXPORT-TRANSPORT-CONTRACT-D1 · ADR_ENTERPRISE_OTLP_TRANSPORT_ADAPTER · ADR_ENTERPRISE_DISTRIBUTED_OBSERVABILITY_TRANSPORT · ADR_ENTERPRISE_OBSERVABILITY_EXPORTER_COMPOSITION |

## Context

Independent W4 closure sync established **HARNESS-W4 = CLOSED**, **HARNESS-W5 = CURRENT**, **HARNESS-W6 = PLANNED**, **SCENARIO-GATE = BLOCKED** (roadmap + freeze checklist, 2026-09-26 read).

Current-HEAD recertification of W5 must prove one event/evidence spine, composition ownership, and **no observability-created execution truth**. Historical W5-A…H/H1 qualification artifacts are **input evidence only**, not current-HEAD certification.

**ADR-OBS-005** (Accepted — Implemented for P1B-R3 delivery boundary) explicitly leaves **OTLP wire typing partial** under debt ID **OBS-EXPORT-TRANSPORT-CONTRACT-D1**. That debt blocks parent **HARNESS-W5** closure until this decision is accepted and **HARNESS-W5-R1** implements it.

**Layer constraint (hard):** `intergrax/contracts/**` must not import `intergrax/runtime/**`. Transport typing must be resolved in contracts + runtime adapters without reversing layer direction.

### Current-HEAD topology (verified)

```text
RuntimeEvent producer
    → RuntimeEventBus (Plane A durable commit, then Plane B delivery)
    → runtime_event_to_deliverable → DeliverableEvent(ObservabilityExportPayload)
    → EventSinkPort.publish
    → BoundedEventSink (buffer / backpressure)
    → RuntimeEventExportSink (EventSinkPort bridge)
    → EventExportSinkPort.export(ObservabilityExportPayload)
    → OtlpEventExportSink
    → envelope_from_observability_export_payload (runtime mapper)
    → ObservabilityExportEnvelope
    → OtlpTransportPort.export(event: object)   ← D1 blocker
    → OtlpTransport / CollectorTransport
    → external collector
```

Parallel **non-spine** export routes (journal, fanout, JSONL, integration OTLP, problem signals) consume **`ObservabilityExportEnvelope`** directly at runtime — intentionally outside `EventSinkPort`.

### Known D1 debt (current code)

| Location | Issue |
|----------|--------|
| `intergrax/contracts/observability_export.py` | `OtlpTransportPort.export(self, event: object)` |
| `intergrax/runtime/observability/exporters/otlp/otlp_transport.py` | `isinstance(event, ObservabilityExportEnvelope)` else `envelope_from_runtime_event(event)  # type: ignore[arg-type]` |
| `intergrax/runtime/observability/exporters/distributed/collector_transport.py` | `export(self, event: object)` delegates to inner transport |

**Enterprise violation:** generic `object` semantic seam, dynamic type probing, `type: ignore` masking contract mismatch on an **extensible** transport port.

---

## Closed-world caller inventory (§12)

| Symbol | Path | Layer | Semantic responsibility | Callers | Implementations | Lifecycle owner | Current typing | Authority | Decision impact |
|--------|------|-------|-------------------------|---------|-----------------|-----------------|----------------|-----------|-----------------|
| `EventSinkPort` | `contracts/event_delivery.py` | T0 | Observability delivery publish/close | `RuntimeEventBus`, `BoundedEventSink` drain | `BoundedEventSink`, `RuntimeEventExportSink`, `InMemoryEventSink`, `AcceptingObservabilityEventSink`, tests | App wiring → bus owns bus+sink shutdown | `DeliverableEvent` | Contract | None |
| `EventExportSinkPort` | `contracts/event_delivery.py` | T0 | Export plugin boundary | `RuntimeEventExportSink` | `NoopEventExportSink`, `OtlpEventExportSink`, `RecordingEventExportSink`, test fakes | Factory + bridge `close()` | `ObservabilityExportPayload` | Contract | None |
| `EventExportSinkFactoryPort` | `contracts/observability_export.py` | T0 | Profile → sink | `runtime_event_delivery_wiring` | `ObservabilityExportSinkFactory` | Composition root per env | Typed | Contract | None |
| `OtlpTransportPort` | `contracts/observability_export.py` | T0 | OTLP/collector transport seam | `OtlpEventExportSink`, `ObservabilityExportSinkFactory`, wiring | `OtlpTransport`, `CollectorTransport`, test fakes | Wiring holds transport ref; closed via export sink chain | **`object`** | Contract (weak) | **R1 primary** |
| `ObservabilityExportPayload` | `contracts/event_delivery.py` | T0 | Canonical contract export fields for event-delivery spine | Bus mapper, `DeliverableEvent`, `EventExportSinkPort`, tests | Dataclass + factories | Producer mapper only writes | Strong | **Canonical spine truth** | **R1 wire input** |
| `DeliverableEvent` | `contracts/event_delivery.py` | T0 | Sink envelope + sequence | All `EventSinkPort` | Factory `make_deliverable_event` | Bus / mapper | Strong | Delegates identity to payload | None |
| `ObservabilityExportEnvelope` | `runtime/observability/export_boundary.py` | T1 | Normalized export record (multi-route) | Fanout, JSONL, journal, policy, legacy OTLP tests, `envelope_from_runtime_event` | Pydantic model | Export policy / mappers | Strong (runtime) | **Runtime export truth (non-spine routes)** | Not promoted to transport port |
| `RuntimeEventExportSink` | `runtime/.../runtime_event_export_sink.py` | T1 | `EventSinkPort` → `EventExportSinkPort` bridge | `BoundedEventSink` downstream | Single class | Bounded sink / bus close | Strong | Bridge only | Simplify after R1 (drop envelope hop) |
| `OtlpEventExportSink` | `runtime/.../otlp_event_export_sink.py` | T1 | `EventExportSinkPort` → transport | Factory | Single class | `close()` flushes transport | Payload in, envelope to transport today | Adapter | **R1: pass payload to transport** |
| `OtlpTransport` | `runtime/.../otlp/otlp_transport.py` | T1 | SDK adapter | Wiring, `CollectorTransport`, tests | Class | `close()` idempotent | **`object` + probe** | Adapter | **R1: typed payload** |
| `CollectorTransport` | `runtime/.../distributed/collector_transport.py` | T1 | Distributed OTLP wrapper | Wiring | Class | Delegates inner | **`object`** | Adapter | **R1: typed payload** |
| `ObservabilityExportSinkFactory` | `runtime/.../export_factory.py` | T1 | Default factory | Wiring default | Class | Per wiring instance | Strong | Composition | None |
| `RuntimeEventBus` | `runtime/events/event_bus.py` | T1 | Persist + deliver + subscribers | Platform | Class | `close()` drains sink | Strong | **Sole `CriticalEventDeliveryError` owner** | None |
| `BoundedEventSink` | `runtime/.../bounded_event_sink.py` | T1 | Bounded buffer + QoS | Bus `event_sink` | Class | `close()` drain bound | Strong | Buffer authority | None |
| `ApplicationRuntimeEventDeliveryWiring` | `applications/_shared/runtime_event_delivery_wiring.py` | T2 | Composition root | Harness / apps | Dataclass + resolvers | `close_application_runtime_event_delivery` | Strong | **Exporter/transport selection owner** | None |

No unexplained consumers remain in closed-world grep for `OtlpTransportPort` and spine types.

---

## Payload vs envelope — field responsibility (§8)

| Field / concern | `ObservabilityExportPayload` (contracts) | `ObservabilityExportEnvelope` (runtime) | Classification |
|-----------------|------------------------------------------|----------------------------------------|----------------|
| Schema id | `observability_export_payload.v1` | `observability_export_envelope.v1` | Distinct schema versions — spine vs general export |
| Identity (`event_id`, `kind`) | Canonical stored | `event_id` + `record_kind` / `event_type` | Spine: **canonical semantic truth** in payload; envelope: **general export truth** |
| Correlation / W3C | `correlation_id`, `w3c_traceparent`, `w3c_tracestate` | Same + more ids | Spine: canonical; envelope: transport projection for OTLP attrs |
| Execution ids | `run_id`, `task_id`, `attempt_id`, `execution_id`, `agent_id`, `tenant_id` | Same + `workspace_id`, `capability`, `tool_id` | Payload: spine subset; extra envelope fields via **safe_attributes** or other routes |
| Timing | — | `recorded_at` | **Wire encoding / projection** (mapper sets UTC now for spine) |
| Status / latency / counts | Encoded in `safe_attributes` bag | `status`, `latency_ms`, `counts` | **Transport projection** from bag in `envelope_from_observability_export_payload` |
| Artifacts / paths / app attrs | Forbidden at contract layer | `artifact_ref`, `safe_relative_path`, `application_attributes`, … | **Unrelated export route** (journal, problem, fanout) |
| Problem / tool / RAG kinds | — | `ExportRecordKind` variants | **Unrelated export route** |
| Spine → OTLP today | Full sufficient for RUNTIME_EVENT subset | Produced by mapper | **Duplicate for transport path only** — not second authority if mapper is single runtime owner |

**Conclusion:** Two models reflect **two semantic responsibilities**:

1. **`ObservabilityExportPayload`** — public, vendor-neutral **event-delivery / export-spine** contract (ADR-OBS-005 Decision 1).
2. **`ObservabilityExportEnvelope`** — runtime **general observability export** surface (policy, fanout, journal, integrations).

They must not be merged by promoting envelope into contracts solely to fix OTLP typing. The spine transport path should not require a permanent envelope hop.

---

## Architecture options

### Option A — Reuse `ObservabilityExportPayload` on `OtlpTransportPort`

```text
EventExportSinkPort → ObservabilityExportPayload
OtlpEventExportSink → ObservabilityExportPayload (passthrough)
OtlpTransportPort.export(payload: ObservabilityExportPayload)
OtlpTransport / CollectorTransport → runtime mapper → OTLP SDK
```

| Criterion | Assessment |
|-----------|------------|
| Layer direction | Contracts stay independent; mapping stays in runtime adapters |
| Semantic ownership | Payload remains single spine truth |
| Duplicate truth risk | Low — envelope hop removed from spine |
| Strong typing | Fixes D1 on extensible port |
| Pluginability | External transport implements typed port |
| Replaceability | Swap OTLP/Collector without consumer rewrite |
| Transport neutrality | Port stays vendor-neutral |
| Schema/evolution | Evolve `observability_export_payload.v1` via contract ADR |
| Existing consumers | Envelope routes unchanged |
| Migration | Medium — tests calling `transport.export(RuntimeEvent\|Envelope)` retarget helpers |
| Test impact | OTLP adapter tests + fakes update signature |

### Option B — New `ObservabilityTransportRecord` in contracts

Rejected unless payload provably insufficient. **Evidence:** OTLP adapter today maps payload → envelope with no envelope-only required fields that cannot be derived from payload + deterministic mapper rules (`observability_export_payload_mapping.py`). No extra transport-only authority demonstrated.

### Option C — Promote `ObservabilityExportEnvelope` to contracts

Rejected: broad Pydantic surface, runtime policy coupling, large blast radius, contradicts ADR-OBS-005 re-use note (“do not widen runtime envelope into EventSinkPort”).

### Option D — Union on transport port

Rejected: violates §28 (no permanent `Union[RuntimeEvent, Envelope, Payload]` or `isinstance` probing).

---

## Chosen architecture

**Option A** — `OtlpTransportPort.export(payload: ObservabilityExportPayload) -> None`.

| Role | Owner |
|------|--------|
| Canonical semantic owner (spine) | `ObservabilityExportPayload` + `runtime_event_to_deliverable` mapper |
| Canonical wire type at transport port | `ObservabilityExportPayload` |
| Canonical mapper owner (payload → OTLP log record) | `intergrax/runtime/observability/exporters/otlp/` (may internally call `envelope_from_observability_export_payload` or equivalent **private** projection — not a second public truth) |
| Composition owner | `ApplicationRuntimeEventDeliveryWiring` / `runtime_event_delivery_wiring.py` |
| Lifecycle owner | `close_application_runtime_event_delivery` → bus → bounded sink → export sink → transport `close()` |
| Plugin boundary | `EventSinkPort`, `EventExportSinkPort`, `EventExportSinkFactoryPort`, `OtlpTransportPort` — all typed |
| General export routes | Continue `ObservabilityExportEnvelope` at runtime; **out of spine transport port** |

**Rejected:** B (unnecessary duplicate), C (over-broad contracts import), D (dynamic union).

---

## W5 closed-world architecture audit (current HEAD)

| Area | Status | Notes |
|------|--------|-------|
| **W5-A** backpressure | STILL TRUE | `BoundedEventSink` + `EventDeliveryPolicy`; fixed capacity; CRITICAL completion; P1B-R3 tests green |
| **W5-B/B2** bus + wiring | STILL TRUE | `compose_runtime_event_bus`; composition in `runtime_event_delivery_wiring.py` |
| **W5-C** export bridge | STILL TRUE | `RuntimeEventExportSink`; typed payload; export failure isolated |
| **W5-D** factory | STILL TRUE | `ObservabilityExportSinkFactory`; no global registry |
| **W5-E** OTLP adapter | **BLOCKED** | D1 typing on `OtlpTransport` |
| **W5-F** distributed transport | **BLOCKED** | Same `object` seam via delegation |
| **W5-G** profile activation | STILL TRUE | `resolve_observability_export_profile` |
| **W5-H** final qualification | **NEEDS REPLAY** | Historical PASS not current-HEAD; transport seam claim **SUPERSEDED** by ADR-OBS-005 D1 |
| **W5-H1** OTLP optional | STILL TRUE | `require_otlp_observability_dependency_profile`; `test_w5_h1_otlp_dependency_contract.py` |
| **ADR-OBS-005 / P1B-R3** | STILL TRUE | Delivery boundary implemented; D1 explicitly open |

### One event/evidence spine (§15)

| Plane | Owner | Proof (current HEAD) |
|-------|-------|----------------------|
| **A — durable evidence** | `RuntimeEventBus` persistence ports, CAS stores | Bus commits before delivery; contracts doc on `event_delivery.py` |
| **B — observability delivery/export** | `EventSinkPort` stack | Does not replace persistence |
| **C — subscribers** | `RuntimeEventBus` handlers | Do not mint execution IDs |

Invariants hold: export failure does not mutate execution outcome; observability does not grant execution permission; delivery reaction cannot widen governance (see §17).

### Critical event authority (§16)

`CriticalEventDeliveryError` raised only in `event_bus.py`. `BoundedEventSink` / `RuntimeEventExportSink` return `EventDeliveryResult` / `EventDeliveryBoundaryError` per P1B-R3 tests.

### Delivery reaction (§17)

`EnterpriseDefaultEventSinkDeliveryReaction` implements `EventSinkDeliveryReactionPort`; bus injects strategy; critical fail-closed floor via `effective_event_delivery_obligation`.

### Lifecycle (§19)

Single stack per `ApplicationRuntimeEventDeliveryWiring`; drain ordering via bus `close()`; no global transport singleton.

### Observability ≠ execution truth (§23)

No in-scope blocker found beyond D1 typing (observability paths remain non-authoritative).

---

## Weak-boundary inventory (W5 production scope)

| Location | Construct | Classification |
|----------|-----------|----------------|
| `observability_export.py` | `OtlpTransportPort.export(object)` | **SEMANTIC BOUNDARY BLOCKER** (D1) |
| `otlp_transport.py` | `export(object)`, `isinstance`, `type: ignore` | **SEMANTIC BOUNDARY BLOCKER** |
| `collector_transport.py` | `export(object)` | **SEMANTIC BOUNDARY BLOCKER** |
| `runtime_event_delivery_wiring.py` | `settings: object \| None` | **SAFE INTERNAL** — optional settings protocol probe at composition |
| `observability_export_payload_mapping.py` | `isinstance` on safe attribute values | **SAFE INTERNAL** — bag parsing |
| `queue_backed_event_delivery_buffer.py` | `cast`, queue `object` | **SAFE INTERNAL** — buffer mechanics |
| `exporters/__init__.py`, `otlp/__init__.py` | `__getattr__ -> object` | **SAFE INTERNAL** — lazy optional OTLP imports |

---

## Historical drift (§25)

| Claim | Classification |
|-------|----------------|
| W5-H: `OtlpTransportPort` stable sync seam | **SUPERSEDED BY ADR-OBS-005** — structurally true, typing incomplete (D1) |
| W5-H1 OTLP optional capability | **STILL TRUE CURRENT HEAD** |
| W5-A bounded delivery gap | **SUPERSEDED BY ADR-OBS-005** — closed in P1B-R3 |
| P1B-R3 `EventExportSinkPort.export(ObservabilityExportPayload)` | **STILL TRUE** |
| ADR-OBS-005 blocker C (`export(event: object)`) | **STILL TRUE** for **transport** port (export sink port fixed) |

---

## P0 baseline evidence

| Gate | Command / scope | Result (HEAD `de57f5dd…`, 2026-09-26) |
|------|-----------------|--------------------------------------|
| P1 | `test_enterprise_scale_resilience_w5_h_final_qualification.py` | PASS (batch) |
| P2 | W5-A/B/B2/C/D/G suites | PASS (batch) |
| P3 | `test_event_bus_import_boundary.py` | PASS |
| P4 | `test_otlp_transport_adapter.py` | PASS |
| P5 | `test_distributed_observability_transport.py` | **1 FAIL** — `RuntimeEventSchemaError` phase mismatch in harness publish (test event fixture) |
| P6 | `test_w5_h1_otlp_dependency_contract.py` | PASS |
| P7 | P1B-R3 `test_event_delivery_contract_p1b_r3.py`, `test_bounded_delivery_p1b_r3_r2/r3` | PASS |
| P8 | Combined batch (96 passed, 1 failed) | Log: `.tmp/session/harness-w5-p0/pytest-w5-baseline.log` |

**Note:** Green tests do not override D1 architecture blocker.

### Pyright baseline

```text
uv run pyright intergrax/contracts/event_delivery.py intergrax/contracts/observability_export.py \
  intergrax/runtime/observability/event_delivery intergrax/runtime/observability/exporters/otlp \
  intergrax/runtime/observability/exporters/distributed \
  intergrax/applications/_shared/runtime_event_delivery_wiring.py
→ 0 errors, 1 warning: otlp/__init__.py __all__ / validate_otlp_export_configuration (reportUnsupportedDunderAll)
```

`git diff --check` → clean.

### Precondition note

Operator pin `007a3dc…` is ancestor of HEAD; **no commits** on listed W5 paths between pin and `de57f5dd…`. HEAD advanced without W5 semantic delta on audited paths.

---

## Qualification test inventory (consolidated R1 replay)

| Test file | Concern | Historical purpose | Still current? | R1 use | Gap |
|-----------|---------|-------------------|----------------|--------|-----|
| `test_enterprise_scale_resilience_w5_a_observability_backpressure.py` | W5-A | Backpressure | Yes | Replay | — |
| `test_enterprise_scale_resilience_w5_b_event_bus_integration.py` | W5-B | Bus integration | Yes | Replay | — |
| `test_enterprise_scale_resilience_w5_b2_composition_wiring.py` | W5-B2 | Wiring | Yes | Replay | — |
| `test_enterprise_scale_resilience_w5_c_event_export.py` | W5-C | Export bridge | Yes | Replay | — |
| `test_enterprise_scale_resilience_w5_d_exporter_composition.py` | W5-D | Factory | Yes | Replay | — |
| `test_otlp_transport_adapter.py` | W5-E | OTLP adapter | Yes | **Update for typed transport** | D1 |
| `test_distributed_observability_transport.py` | W5-F | Collector | Yes | Replay + fix harness fixture | 1 failing test |
| `test_enterprise_scale_resilience_w5_g_profile_activation.py` | W5-G | Profiles | Yes | Replay | — |
| `test_enterprise_scale_resilience_w5_h_final_qualification.py` | W5-H | Parent | Partial | Full replay post-R1 | Transport claim stale |
| `test_w5_h1_otlp_dependency_contract.py` | W5-H1 | Optional OTLP | Yes | Replay | — |
| `test_event_delivery_contract_p1b_r3.py` | ADR-OBS-005 | Delivery contract | Yes | Replay | — |
| `test_bounded_delivery_p1b_r3_r2.py`, `r3.py` | P1B-R3 | Shutdown/health | Yes | Replay | — |
| `test_event_bus_import_boundary.py` | W5-H | Import order | Yes | Replay | — |

**Consolidated R1 replay:** single pytest invocation of all rows above + new mechanical gate forbidding `OtlpTransportPort.export(event: object)` in `intergrax/contracts/observability_export.py` (AST or grep gate).

---

## FRZ mapping (evidence only — all OPEN)

| FRZ | P0 contribution |
|-----|-----------------|
| FRZ-OBS-01…07 | Architecture plan + spine inventory; no PASS |
| FRZ-CTR-01,03,05 | Contract inventory; D1 gap documented |
| FRZ-TYP-01,02,03,04,06 | D1 blocker = primary typing debt |
| FRZ-PLG-01, FRZ-RPL-04 | Option A preserves replaceable ports |
| FRZ-EXE-01,02,07 | Observability non-authoritative — evidenced in bus/bridge design |
| FRZ-GOV-01,02,09 | Delivery reaction + bus authority — evidenced |
| FRZ-REG-02,06,09 | R1 gates listed below |

---

## Regression gates (R1)

Mechanically prevent:

- `OtlpTransportPort.export(event: object)`
- `# type: ignore` on transport input typing
- `isinstance` / runtime probing on transport `export` input
- Concrete exporter selection outside `runtime_event_delivery_wiring.py`
- Second event/export spine

---

## HARNESS-W5-R1 — implementation scope (next wave)

**Title:** HARNESS-W5-R1 — Typed Observability Transport & Current-HEAD W5 Recertification

### Production

- `intergrax/contracts/observability_export.py` — typed `OtlpTransportPort.export`
- `intergrax/runtime/observability/exporters/otlp/otlp_transport.py` — remove probe branch; map from payload
- `intergrax/runtime/observability/exporters/distributed/collector_transport.py` — typed delegate
- `intergrax/runtime/observability/event_delivery/otlp_event_export_sink.py` — pass payload directly
- Optional: collapse duplicate mapper call path (keep `envelope_from_observability_export_payload` as **private** OTLP projection helper only)

### Tests

- Update fakes (`FakeOtlpTransport`, adapter tests)
- Fix `test_distributed_observability_transport.py` harness event phase
- Add AST/grep regression gate for transport `object` seam
- Full consolidated replay table above

### Docs

- Mark ADR **Accepted** after independent audit
- Parent W5 qualification record (current-HEAD)

### Out of scope

- Promoting `ObservabilityExportEnvelope` to contracts
- New global registry / scheduler / queue below canonical sink
- contracts → runtime imports

---

## STOP conditions

No additional semantic owner required beyond this ADR. If R1 discovers payload insufficient for OTLP **without** envelope-only fields from non-spine routes → **STOP — ARCHITECTURE DECISION REQUIRED** (reopen Option B with evidence).

---

## P0 outcome

| Item | Value |
|------|--------|
| **HARNESS-W5-P0** | READY FOR AUDIT |
| **HARNESS-W5** | BLOCKED pending P0 acceptance + **HARNESS-W5-R1** |
| **IN-SCOPE BLOCKER count** | **1** (OBS-EXPORT-TRANSPORT-CONTRACT-D1 — manifests as 3 code sites + protocol) |
| **TRACKED FREEZE DEBT** | W5-H historical PASS; pyright `__all__` warning |
| **ENVIRONMENT/TEST** | 1 distributed harness test failure (schema phase on fixture) |

---

## R1 implementation (HARNESS-W5-R1)

| Item | Value |
|------|--------|
| **Decision** | Option A unchanged — not redesigned |
| **OBS-EXPORT-TRANSPORT-CONTRACT-D1** | Removed in R1 (`OtlpTransportPort.export(ObservabilityExportPayload)`) |
| **HARNESS-W5-R1** | **IMPLEMENTED — READY FOR AUDIT** (independent SHA acceptance required) |
| **Regression gate** | `tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py` |
| **Parent record** | `docs/project/maintainers/qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_PARENT_RECERTIFICATION.md` |
