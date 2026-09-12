# ERL reconciliation plugin integration

> **Purpose:** Document how Enterprise Reliability obtains external truth through a replaceable reconciliation plugin without embedding business domain logic in `intergrax/`.

## Problem framing

Enterprise workflows often face **unknown external effect outcomes**—communication gaps, partial responses, or delayed authoritative state. The platform must answer a generic question:

**How do we obtain evidence about an uncertain external effect?**

Payment capture in ERL-QUAL-004 is one qualification instance of that pattern. Reconciliation logic for acquirer ledgers lives in the **scenario integration boundary**, not in the Enterprise Reliability Layer.

## Boundary architecture

```text
Reliability case (UNKNOWN)
        ↓
Reconciliation orchestration (intergrax/runtime/enterprise_reliability)
        ↓
EnterpriseReliabilityPluginGateway
        ↓
ReconciliationStrategy + ReconciliationProbeExecutor (plugin SPI)
        ↓
Scenario external-reality adapter (platform_proofs/.../erl_integration)
        ↓
Authoritative store (e.g. PostgreSQL external_sor.external_reality)
        ↓
ExternalEffectEvidence + ReconciliationProbeResult
```

## Ownership

| Concern | Owner |
| --- | --- |
| Lifecycle, planning, probe scheduling, evidence materialization | `intergrax/runtime/enterprise_reliability` |
| Plugin SPI, probe request/result contracts, evidence models | `intergrax/contracts/enterprise_reliability` |
| Domain truth lookup, SoR schema, dataset variants | Scenario (`platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/`) |
| Payment capture simulator | `external_payment/` (separate boundary) |

ERL never imports scenario packages. Scenarios register plugins on `EnterpriseReliabilityPluginRegistry` at composition time.

## Plugin contract (platform)

- **`ReconciliationStrategy`** — optional probe advice (`probe_ref`) within contract-declared refs.
- **`ReconciliationProbeExecutor`** — read-only probe I/O; returns `ReconciliationProbeResult` (`verdict`, `evidence_ref`, `rationale`).

Registry resolves both protocols from the same `plugin_id` when the implementation provides `execute_probe`.

Generic probe reference for ERL-QUAL-004: `external_effect.system_of_record_read` (declared on the scenario `ExternalEffectContract`).

## Scenario adapter (ERL-QUAL-004)

Package: `platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/erl_integration/`

| Component | Role |
| --- | --- |
| `ExternalRealityLookupPort` | Correlation-scoped SoR read |
| `PostgreSqlExternalRealityLookup` / `InMemoryExternalRealityLookup` | Replaceable stores |
| `map_snapshot_to_probe_result` | Payment SoR columns → platform verdict |
| `ScenarioExternalRealityReconciliationPlugin` | SPI implementation |
| `register_scenario_reconciliation_plugins` | Bootstrap reconciliation + minimal resolution strategy |

Variant behavior (A/B/C) is **data-driven** from dataset slices (`external_reality.system_of_record_truth`); the mapper interprets normalized SoR fields, not `variant_id` branches.

## Lifecycle interaction

1. UNKNOWN admission → reconciliation planning (`RECONCILIATION_RUNNING` / pending resolution phases).
2. Gateway `execute_reconciliation_probe` invokes the registered executor.
3. Runtime materializes `ExternalEffectEvidence`.
4. Evidence evaluation (`evaluate_external_effect_evidence`) determines resolution readiness — see [`ERL_EVIDENCE_EVALUATION.md`](ERL_EVIDENCE_EVALUATION.md).
5. Resolution orchestration runs when evaluation outcome is `READY_FOR_DECISION` (or defers on insufficient evidence).

Failures are explicit in `rationale` (`source_unavailable`, `record_missing`, inconsistent SoR, indeterminate truth)—plugins return `INSUFFICIENT` rather than omitting probe results.

## Replaceability

Applications and proofs swap plugins by `plugin_id` on the same registry port. Contract tests assert ERL contracts remain free of payment- or vendor-specific vocabulary.

## Related documents

- [`ENTERPRISE_RELIABILITY_LAYER.md`](ENTERPRISE_RELIABILITY_LAYER.md) — capability kinds and gateway
- [`RECONCILIATION.md`](RECONCILIATION.md) — verification flow canon
- Scenario: `platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/docs/ERL_QUAL_004_EXTERNAL_PAYMENT_BOUNDARY.md`
