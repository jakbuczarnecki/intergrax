# Enterprise Scale & Resilience — W5-B2 Production Composition Wiring Inventory

**Task:** W5-B2 — BoundedEventSink explicit composition-root wiring  
**Baseline:** W5-B bus `event_sink` injection qualified on `development`.

## Composition ownership (ETAP 1)

| Element | Owner | Aktualnie |
|---------|--------|-----------|
| **RuntimeEventBus** | `wire_application_environment` → `ApplicationBuildContext.runtime_event_bus` | Created via `compose_runtime_event_bus` when `ObservabilityProfile.bounded_event_delivery_enabled`; else legacy `RuntimeEventBus()` |
| **EventSink** | `ApplicationRuntimeEventDeliveryWiring` on `ApplicationEnvironmentWiring.event_delivery` | `BoundedEventSink` → `AcceptingObservabilityEventSink` terminal (process-local export handoff TBD) |
| **Lifecycle owner** | Tier-3 composition root (`ApplicationEnvironmentWiring` + `HarnessHostRuntime`) | Same owner as bus; sink not created inside `RuntimeEventBus` |
| **Close path** | `HarnessHostRuntime.close()` → `close_application_runtime_event_delivery` → `RuntimeEventBus.close()` | Closes bounded worker then downstream terminal |
| **Production composition** | `intergrax/applications/_shared/environment_wiring.py`, `runtime_event_delivery_wiring.py`, `harness_host_runtime.py` | Enabled for `GovernanceBundle.production_slo()` and `harness_production_defaults` |

## RuntimeEventBus instantiation (production-relevant)

| Site | Role |
|------|------|
| `environment_wiring.wire_application_environment` | **Canonical** Tier-3 bus + optional bounded delivery |
| `applications/lab_application/host/wiring.py` | Optional override `runtime_event_bus or RuntimeEventBus()` |
| `intergrax/applications/_shared/environment_wiring.py` L448 (pre-B2) | Default mint when caller omits bus |
| `intergrax/runtime/nexus/nexus_loop.py` | Fallback bus only when host omits `event_bus` (non-canonical) |
| `testing_support/decision_e2e/composition.py` | E2E harness placeholders |

## Related types (inventory)

| Type | Primary constructor | Lifecycle |
|------|---------------------|-----------|
| `RuntimeEventRecorder` | Nexus / execution failure paths | Bus-scoped, no separate sink |
| `RuntimeEventPublisher` | Orchestration protocols / `NexusRuntimeEventPublisher` | Uses injected bus |
| `ExecutionEnvironment` | `EffectiveExecutionEnvironment` (sandbox) | Unrelated to observability sink |
| `ApplicationComposition` | `AgentCapabilityApplicationComposition` | AC-4 agent platform; shares process bus via host wiring |

## Target pipeline (W5-B2)

```text
Composition Root
      |
      +-- AcceptingObservabilityEventSink (terminal)
      +-- BoundedEventSink (policy from ObservabilityProfile)
      +-- RuntimeEventBus(event_sink=bounded)
      |
      v
Consumers (subscribers + durable persistence on bus — unchanged)
```

## Qualification

`tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py`
