# Layer boundary hardening (HARDENING-3)

## Target dependency model

```text
vendor implementations
        ↑
infrastructure / adapters
        ↑
runtime
        ↑
contracts
```

`intergrax/contracts` holds Protocols, enums, immutable models, and shared value types.
`intergrax/runtime` implements ports and may import contracts only.

## Fixes in HARDENING-3

| Violation | Resolution |
|-----------|------------|
| Decision deliberation modules imported `InferenceProfileId` from `runtime.execution.inference_profile` | Use `intergrax.contracts.inference_profile_id` (canonical home). |
| `PolicyRuleAction` lived in runtime policy schema but was referenced from contracts | Moved enum to `intergrax.contracts.declarative_policy_rule_action`; runtime schema re-imports it. |
| `stable_payload_hash` pulled contracts into runtime attestation | Moved digest helpers to `intergrax.contracts.canonical_payload_hash`; runtime attestation re-exports. |
| `AsOfBoundary` / `ExecutionEventPosition` in runtime but used from contracts | Moved to `intergrax.contracts.execution_event_position`; runtime `execution_position` keeps `PositionedRuntimeEvent`. |
| Observability export ports typed on `RuntimeEvent` | Port methods use `object` at the contract seam; adapters remain runtime-typed. |
| `historical_reconstruction` imported as-of types from runtime | Imports from `execution_event_position` contract module. |

Regression gate: `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py`.

## Deferred (requires broader migration — do not expand scope ad hoc)

| Module | Coupling | Planned direction |
|--------|----------|-------------------|
| `host_profile_slices.py` | Runtime adaptive / nexus profile types | Hoist shared profile slices into contracts or split host config package. |
| `runtime_mapping.py` | Nexus `RuntimeAnswer`, interrupt handler types | Lazy legacy bridge; replace with contract DTOs + runtime-only mappers. |
| `runtime_cost.py` | `RuntimeAnswer` | Introduce neutral cost contract type; map in runtime. |
| `runtime_execution_context.py` | Lazy cancellation / runtime event emission | Extract port hooks; keep execution behaviour unchanged. |
| `execution_evidence/persistence_port.py` | `RuntimeEvent` in persistence Protocol | Introduce contract-level evidence event surface or shared neutral event model (NPSC-5F follow-up). |

Self-healing (`contracts/self_healing/**`) remains free of runtime imports (covered by gate).

## Persistence and execution authority

No change to Domain → Repository Port → Adapter → Provider wiring.
Execution authority and plugin SPI boundaries are unchanged; this pass only removes improper upward imports.
