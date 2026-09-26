# Enterprise Execution Scale & Resilience — W5 Parent Recertification (HARNESS-W5-R1)

**Status:** READY FOR AUDIT  
**Task:** HARNESS-W5-R1 — Typed Observability Transport & Current-HEAD W5 Recertification  
**Audited HEAD:** _(set at commit — see git rev-parse after push)_  
**P0 baseline:** `307a871c059f150a6b744e39135a737399467e88`  
**Accepted architecture:** ADR Option A — `ADR_HARNESS_W5_OBSERVABILITY_TRANSPORT_CONTRACT.md`, semantic authority `ADR-OBS-005.md`

## Canonical program state

| Item | State |
|------|--------|
| HARNESS-W4 | CLOSED |
| HARNESS-W5 | CURRENT → **READY FOR AUDIT** (recommendation) |
| HARNESS-W5-R1 | **READY FOR AUDIT** |
| HARNESS-W6 | PLANNED |
| SCENARIO-GATE | BLOCKED |

## Transport topology

### Before (D1 blocker)

```text
ObservabilityExportPayload
→ envelope_from_observability_export_payload (OtlpEventExportSink)
→ OtlpTransportPort.export(object)
→ isinstance / envelope_from_runtime_event + type: ignore
→ OTLP SDK
```

### After (R1)

```text
RuntimeEvent
→ runtime_event_to_deliverable
→ ObservabilityExportPayload
→ DeliverableEvent
→ EventSinkPort → BoundedEventSink
→ RuntimeEventExportSink → EventExportSinkPort (OtlpEventExportSink)
→ OtlpTransportPort.export(ObservabilityExportPayload)
→ OtlpTransport / CollectorTransport (private payload → LogRecord)
→ external collector
```

**Canonical semantic owner:** `ObservabilityExportPayload`  
**Composition owner:** `intergrax/applications/_shared/runtime_event_delivery_wiring.py`

## Contract / owner inventory

| Contract | Owner layer | Primary consumers |
|----------|-------------|-------------------|
| `ObservabilityExportPayload` | contracts | export bridge, transport port |
| `DeliverableEvent` | contracts | `BoundedEventSink`, `RuntimeEventExportSink` |
| `EventSinkPort` | contracts | `RuntimeEventBus`, `BoundedEventSink` |
| `EventExportSinkPort` | contracts | `RuntimeEventExportSink`, `OtlpEventExportSink` |
| `EventExportSinkFactoryPort` | contracts | `ObservabilityExportSinkFactory`, app wiring |
| `OtlpTransportPort` | contracts | `OtlpEventExportSink`, `OtlpTransport`, `CollectorTransport` |

## Closed-world transport callers (HEAD)

| Kind | Symbol |
|------|--------|
| Production adapter | `OtlpTransport.export` |
| Production adapter | `CollectorTransport.export` |
| Export bridge | `OtlpEventExportSink.export` → `transport.export(payload)` |
| Composition | `runtime_event_delivery_wiring._create_export_transport` |
| Test fakes | W5-E/F/G/H `FailingTransport` / `FailingCollector` / `SlowTransport` |
| Gate | `RecordingTransport` in architecture gate |

## Strong typing / weak-seam scan (W5 transport scope)

| Check | Result |
|-------|--------|
| `OtlpTransportPort.export` semantic `object` | **0** |
| Transport `isinstance` / `getattr` / `hasattr` on input | **0** |
| Transport input `type: ignore` | **0** |
| `contracts/observability_export` → runtime import | **0** |
| D1 residual OBS-EXPORT-TRANSPORT-CONTRACT-D1 | **0** |

`runtime_event_to_otlp_log_record` — **diagnostic/test helper**, not canonical transport contract.

## Fixture reconciliation (TASK_PROGRESS + COMPLETION → STEP_EXECUTION)

| File | Correction |
|------|------------|
| `test_enterprise_scale_resilience_w5_c_event_export.py` | `COMPLETION` → `STEP_EXECUTION` |
| `test_enterprise_scale_resilience_w5_d_exporter_composition.py` | same |
| `test_enterprise_scale_resilience_w5_g_profile_activation.py` | same |
| `test_enterprise_scale_resilience_w5_h_final_qualification.py` | same |
| `test_otlp_transport_adapter.py` | same |
| `test_distributed_observability_transport.py` | same |

W5-B uses `TASK_COMPLETED` + `COMPLETION` — **legitimate**, unchanged.

## W5 replay matrix (current-HEAD)

| Suite | Result |
|-------|--------|
| W5-A backpressure | PASS |
| W5-B bus integration | PASS |
| W5-B2 composition wiring | PASS |
| W5-C event export | PASS |
| W5-D exporter composition | PASS |
| W5-E OTLP adapter | PASS |
| W5-F distributed transport | PASS |
| W5-G profile activation | PASS |
| W5-H final qualification | PASS |
| W5-H1 OTLP dependency | PASS |
| ADR-OBS-005 / P1B-R3 (r2, r3, contract) | PASS |
| Event bus import boundary | PASS |
| R1 typed transport gate | PASS |
| Combined batch A (106 tests) | 0 failed |
| Combined batch B (106 tests) | 0 failed |

Evidence logs: `.tmp/session/harness-w5-r1/`

## Observability ≠ execution truth

| Plane | Role | W5 evidence |
|-------|------|-------------|
| A | Durable execution / evidence truth | Unchanged — export does not mint `ExecutionId` or governance |
| B | Observability delivery/export | Typed payload only; failures → metrics / isolated errors |
| C | Subscribers | Handlers run independent of export success |

`CriticalEventDeliveryError` — raised only from `RuntimeEventBus` (P1B-R3 suites green).

## Lifecycle

`close_application_runtime_event_delivery` → bus → bounded sink → export bridge → `EventExportSinkPort` → `OtlpTransportPort.close()`; flush before shutdown; double combined batch exposes no orphan workers.

## Pluginability / replaceability

Custom `OtlpTransportPort` implementing `export(ObservabilityExportPayload)` substituted via `OtlpEventExportSink` without `OtlpTransport` / `RuntimeEvent` on transport seam — gate `test_replaceable_recording_transport_receives_canonical_payload`.

## FRZ evidence (scoped — all OPEN globally)

Primary FRZ-OBS-01..07: W5 spine + correlation field preservation on transport path — contributed, not closed.  
Supporting FRZ-CTR/TYP/PLG/RPL/EXE/GOV/REG: typed transport removes object seam (FRZ-TYP-03, FRZ-CTR-03, FRZ-REG-02/06/09 mechanical gate). Global QUAL-X closure pending.

## Unresolved findings

| ID | Classification |
|----|----------------|
| _(none)_ | IN-SCOPE BLOCKER = **0** |

## Recommended status

```text
HARNESS-W5-R1 = READY FOR AUDIT
HARNESS-W5 = READY FOR AUDIT (recommendation)
NEXT: independent exact-SHA acceptance → roadmap/checklist sync → HARNESS-W6
```
