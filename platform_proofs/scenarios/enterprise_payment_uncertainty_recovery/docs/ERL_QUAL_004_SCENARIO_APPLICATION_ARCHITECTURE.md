# ERL-QUAL-004 Scenario Application Architecture

> **Status:** Application skeleton (business simulation only). **No reliability decisions** are implemented in this layer.

## Responsibility

The scenario application under `application/` simulates a realistic enterprise purchase-order payment workflow for the ERL-QUAL-004 standalone proof:

1. An order already exists for **Nordic Industrial Components Sp. z o.o.** (`PO-2026-004872`).
2. The application requests payment capture (`PAY-20260912-8F31A`).
3. External uncertainty and Integrax reliability processing are **out of scope** for this skeleton.

**Scenario Application = business simulation.**  
**Integrax Runtime = reliability intelligence** (wired later via `application/runtime/`).  
**Infrastructure / provisioning = state materialization** (PostgreSQL lab adapter and dataset).

The scenario application **does not** implement reconciliation, resolution, compensation, governance, HITL, or ERL workflows.

## Boundaries

| Layer | Location | Owns |
| --- | --- | --- |
| Domain | `application/domain/` | Orders, payment capture request shapes, application failure codes |
| Ports | `application/ports/` | `OrderAccessPort`, `PaymentWorkflowPort` |
| Services | `application/services/` | `EnterprisePaymentWorkflowService` |
| Composition | `application/composition/` | `ScenarioExecutionContext`, `ScenarioApplicationCompositionRoot` |
| Observability | `application/observability.py` | Started / action / state-prepared signals (not ERL evidence) |
| Runtime integration | `application/runtime/` | Platform lab runtime facade (future ERL plugin wiring) |
| Shared references | `contracts/application/` | Production-like lab identifiers |

### Database rule

`application/` **must not** import PostgreSQL drivers or scenario database/provisioning adapters. Data access flows:

```text
application → application port → infrastructure adapter (future)
```

## Lifecycle

1. Harness builds `ScenarioExecutionContext` (scenario id, variant id, execution reference, correlation ids).
2. `ScenarioApplicationCompositionRoot` validates context and records `scenario_started`.
3. `EnterprisePaymentWorkflowService` loads the order via `OrderAccessPort`, then requests capture via `PaymentWorkflowPort`.
4. Observability emits `business_action_executed` for order load and payment request.
5. Composition records `scenario_state_prepared` with `PaymentWorkflowOutcome` (`payment_requested` phase).

Application failures at this stage: invalid context, missing business entity, unavailable dependency.

## Ownership

- **Scenario package** owns business workflow and ports.
- **Proof / evaluator** (future) consumes runtime and application observability artifacts — does not drive business logic.
- **Integrax platform** supplies ERL and execution runtime; composition in `application/runtime/` will register plugins without moving decisions into the proof layer.

## Future Integrax integration

`application/runtime/platform_lab_runtime.py` continues to use the shared scenario lab runtime baseline. A later task will:

- Connect ERL reconciliation, governance, and external-effect metadata to the payment capture boundary.
- Keep material continuation decisions in the platform/runtime, not in `proof/`.

Until then, the composition root executes only the business skeleton described above.

## Related documentation

- [Proof Architecture Design](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md)
- [Scenario Data Architecture](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md)
- [PostgreSQL Data Model Architecture](ERL_QUAL_004_POSTGRESQL_DATA_MODEL_ARCHITECTURE.md)
