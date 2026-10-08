# TRACE-X-P5-R2-P2 — Durable Configuration Opportunity & Execution Provenance

| Field | Value |
|---|---|
| **START_HEAD** | `9351cd7ff8697e38a169afb24b14865b2206a98a` |
| **Status** | **BLOCKED ON P2-R1 INDEPENDENT AUDIT** |
| **Owner** | TRACE-X-P5-R2-P2 |

## Closed-world inventory (pre-edit)

- P1 contracts present on START_HEAD: `ExistingCapabilityConfigurationOpportunity`, `ConfigurationOpportunityRef`, `ExistingCapabilityConfigurationOpportunityReadPort`, `ExecutionIntegrationConfigurationProvenance`, `IntegrationConfigurationSubject`, `ExecutionIntegrationConfigurationProvenanceReader`, `ExecutionId`.
- Reference persistence pattern: `intergrax/applications/_shared/profile_resolution/persistence.py` (KV CAS + ConditionalDocumentStore `put_if_absent`, schema_version envelopes).
- No pre-existing `ExistingCapabilityConfigurationOpportunityStore` / `ExecutionIntegrationConfigurationPinningStore` / Integrations application persistence locus.
- **Payload pre-check:** no platform-wide unsafe `IntegrationConfigurationPayload` serializer; sanctioned **typed per-`configuration_type` codec registry** (`IntegrationConfigurationPayloadCodec` + SQLite provider codec) — not pickle/reflection.

## Store contracts

| Contract | Owner | Module |
|---|---|---|
| `ExistingCapabilityConfigurationOpportunityStore` | Integrations | `intergrax/integrations/contracts/existing_capability_configuration_opportunity.py` |
| `ExecutionIntegrationConfigurationPinningStore` | Integrations | `intergrax/integrations/contracts/execution_integration_configuration_pinning.py` |
| `ExecutionIntegrationConfigurationProvenanceReader` | Neutral read-only | `intergrax/contracts/execution_integration_configuration_provenance.py` |

## Adapters (composition)

| Adapter | Backing | `is_durable` |
|---|---|---|
| `KvExistingCapabilityConfigurationOpportunityStore` | `DistributedKVStore` | `True` |
| `DocumentStoreExistingCapabilityConfigurationOpportunityStore` | `ConditionalDocumentStore` | `True` |
| `KvExecutionIntegrationConfigurationPinningStore` | `DistributedKVStore` | `True` |
| `DocumentStoreExecutionIntegrationConfigurationPinningStore` | `ConditionalDocumentStore` | `True` |
| `InMemory*` reference stores | in-process | `False` |
| `PinningStoreExecutionIntegrationConfigurationProvenanceReader` | pinning store projection | N/A (read-only) |

Module: `intergrax/applications/_shared/integrations/persistence.py`

## Storage keys

| Mechanism | Identity |
|---|---|
| Opportunity | `tenant_id` + `configuration_ref` |
| Provenance pin | `tenant_id` + `ExecutionId` + `IntegrationConfigurationSubject` |

## Codec / schema

| Record | `schema_version` |
|---|---|
| Opportunity envelope | `1` |
| Provenance pin envelope | `1` |

## Idempotency / conflict

| Case | Behavior |
|---|---|
| absent key | persist succeeds |
| same key + identical immutable record | idempotent success |
| same key + different content | fail closed (`CONFLICT`) |

## Tenant matrix

- KV tenant partition + decoded `tenant_id` validation on read.
- Cross-tenant negative tests in `tests/unit/applications/integrations/test_trace_x_p5_r2_p2_persistence.py`.

## STATE-X delta (`R2-P2-STATE-X-DELTA-CLASSIFICATION-01`)

Explicit classifications added in `tests/qualification/state_x/_state_x_explicit_mechanism_classifications.py` for:

- `path:intergrax/applications/_shared/integrations/persistence.py` → `outside_state_x` (composition provider; Integrations owns semantic truth)
- `InMemoryExistingCapabilityConfigurationOpportunityStore` → `non_durable_reference_only`
- `InMemoryExecutionIntegrationConfigurationPinningStore` → `non_durable_reference_only`

Gate: `tests/qualification/trace_x/test_trace_x_p5_r2_p2_persistence_gates.py::test_txp5r2p2_q08_state_x_current_head_delta_classification`

## FRZ

- **FRZ-TRC-11** remains **OPEN** (P2 contributes durable evidence only).

## Tests

- `tests/unit/applications/integrations/test_trace_x_p5_r2_p2_persistence.py`
- `tests/qualification/trace_x/test_trace_x_p5_r2_p2_persistence_gates.py`

## TRACE-X-P5-R2-P2-R1 — Durable atomicity & codec registry hardening

| Blocker | Root cause | Before | Remediation | Tests | Status |
|---|---|---|---|---|---|
| **R2-P2-KV-PROVENANCE-ATOMICITY-01** | Record CAS before index CAS allowed durable orphan records invisible to `read_all()` | `pin`: record → index | Index marker first, then record CAS; incomplete index-without-record fails closed; retry completes; legacy record-without-index healed on retry | `test_kv_provenance_crash_after_index_before_record_fails_closed_then_repair`, `test_kv_provenance_legacy_orphan_record_without_index_repaired_on_retry`, gates `q12` | **READY FOR AUDIT** |
| **R2-P2-CODEC-DEFAULT-SELECTION-02** | `wire_*` accepted `payload_codecs=None` and imported SQLite codec in shared persistence | Implicit `default_integration_configuration_payload_codec_registry()` | Mandatory `payload_codecs`; SQLite import removed from shared persistence; test helpers compose SQLite explicitly | `test_wire_opportunity_store_requires_explicit_payload_codecs`, gates `q09`–`q10` | **READY FOR AUDIT** |
| **R2-P2-CODEC-REGISTRY-MUTABILITY-03** | Registry held mutable `dict` | Frozen dataclass with mutable dict field | `MappingProxyType` + construction validation; encode/decode identity checks | `tests/unit/integrations/test_integration_configuration_payload_codec_registry.py`, gate `q11` | **READY FOR AUDIT** |

**P2 remains not CLOSED.** `R2-P2-STATE-X-DELTA-CLASSIFICATION-01` = **READY FOR AUDIT** (atomicity text updated; discovery gate `q08`).
