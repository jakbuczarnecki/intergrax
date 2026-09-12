# ERL-QUAL-004 External Payment Boundary

> **Purpose:** Represent enterprise payment capture reality for qualification — **not** to prove a platform feature.

## Business problem

High-value B2B orders depend on an external acquirer or PSP to capture funds. The commerce application sends a capture request and must act on the integration response. When communication fails after the provider accepts work, the application cannot know whether money moved. That ambiguity is a **distributed systems** problem: external truth may exist while application knowledge remains unknown until reconciliation or governed escalation.

## External system role

The `external_payment/` package is a **replaceable enterprise integration simulator** (PSP / acquiring / settlement boundary). It is intentionally **not** a mock API with hardcoded demo responses. It models:

- correlation id and external references
- request and processing timestamps
- external lifecycle (requested → processing → terminal channel outcome)
- authoritative system-of-record truth persisted separately from the integration response

## Lifecycle

| Stage | Meaning |
| --- | --- |
| `REQUESTED` | Capture accepted at the boundary |
| `PROCESSING` | Acquirer processing (implicit in simulator timing) |
| `COMPLETED` / `FAILED` | Definitive integration response delivered |
| `UNKNOWN` | Communication uncertainty — **not** payment failure |

Variant slices in `dataset/` drive SoR terminal outcomes (`PAYMENT_COMPLETED`, `PAYMENT_FAILED`, `TRUTH_INDETERMINATE`) without variant-specific code branches in the processor.

## Ownership and boundaries

| Concern | Owner |
| --- | --- |
| Capture command / integration result contracts | `external_payment/contracts/` |
| Lifecycle and failure semantics | `external_payment/domain/` |
| Acquirer processing | `external_payment/services/` |
| Dataset profiles, PostgreSQL SoR adapter, application port wiring | `external_payment/adapters/` |
| Application knowledge, orders, reconciliation | **Not** this boundary |
| Integrax runtime / ERL | **Not** this boundary |

External reality is persisted through `external_sor.external_payment_effects` and `external_sor.external_reality` (PostgreSQL lab adapter or in-memory store for tests). Application tables are not updated by this component.

## Scenario variants (data-driven)

| Variant | SoR truth | Application channel |
| --- | --- | --- |
| `payment_completed_after_unknown` | Payment completed | UNKNOWN |
| `payment_failed_after_unknown` | Payment failed | UNKNOWN |
| `payment_truth_unavailable` | Truth indeterminate / unavailable | UNKNOWN |

## Application integration

`ScenarioExternalPaymentWorkflow` implements `PaymentWorkflowPort`: the scenario application issues capture; the boundary processes, persists SoR artifacts, and returns the enterprise-facing `PaymentCaptureRequest` record.

## Future Integrax integration

A later task will connect ERL reconciliation, governance, and evidence to this boundary. The external simulator must remain **business-first**: platform capabilities evaluate against this reality; gaps are evolved in `intergrax/`, not by bending the scenario.

## Platform gap notes (observation)

- **First-class UNKNOWN containment in application workflow:** the scenario application skeleton does not yet admit UNKNOWN as application knowledge when the boundary returns an uncertain integration outcome; that belongs in application + ERL wiring, not in the external simulator.
- **Cross-boundary correlation registry:** durable correlation between lab execution references, payment intents, and external effect ids will be needed for proof-runner orchestration — universal for enterprise proofs, belongs in scenario harness / platform lab runtime, not in the acquirer simulator alone.
