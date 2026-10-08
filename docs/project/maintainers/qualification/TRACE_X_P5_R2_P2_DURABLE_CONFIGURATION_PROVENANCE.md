# TRACE-X-P5-R2-P2 — Durable Configuration Opportunity & Execution Provenance

| Field | Value |
|---|---|
| **START_HEAD** | `9351cd7ff8697e38a169afb24b14865b2206a98a` |
| **Status** | **CLOSED / independently accepted** |
| **Final accepted baseline** | `660d9d9cd237ca91a6c4e662389b3ec0cf9fd20c` |
| **Owner** | TRACE-X-P5-R2-P2 |

## Child lineage (implementation evidence)

| SHA | Role |
|---|---|
| `6896ea5f60c2d1afef15a5a148c0f018dfbc214e` | initial P2 implementation |
| `e0d5209825de532363b62f0e1e58f53e1ac30e47` | **P2-R1** hardening — **CLOSED / independently accepted** |
| `660d9d9cd237ca91a6c4e662389b3ec0cf9fd20c` | **P2-R2** pagination completeness + final P2 implementation baseline — **CLOSED / independently accepted** |

Bookkeeping commits after `660d9d9c…` (e.g. **TRACE-X-P5-R2-P2-CLOSE**) are docs-only — **not** P2 implementation evidence.

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

**Status:** **SATISFIED / independently accepted** @ `660d9d9cd237ca91a6c4e662389b3ec0cf9fd20c`.

**Evidence:** `discovered durable mechanisms == classified mechanisms`; **unclassified mechanism count = 0**.

**Historical STATE-X accepted baseline** (preserved — not replaced by P2 SHA): `bd54941d933069b8bfb2819bb819c1cbdbe71576`. P2 evidence is **current-head delta only**; does **not** claim new STATE-X parent closure.

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
| **R2-P2-KV-PROVENANCE-ATOMICITY-01** | Record CAS before index CAS allowed durable orphan records invisible to `read_all()` | `pin`: record → index | Index marker first, then record CAS; incomplete index-without-record fails closed; retry completes; legacy record-without-index healed on retry | `test_kv_provenance_crash_after_index_before_record_fails_closed_then_repair`, `test_kv_provenance_legacy_orphan_record_without_index_repaired_on_retry`, gates `q12` | **RESOLVED / independently accepted** @ `e0d5209825de532363b62f0e1e58f53e1ac30e47` |
| **R2-P2-CODEC-DEFAULT-SELECTION-02** | `wire_*` accepted `payload_codecs=None` and imported SQLite codec in shared persistence | Implicit `default_integration_configuration_payload_codec_registry()` | Mandatory `payload_codecs`; SQLite import removed from shared persistence; test helpers compose SQLite explicitly | `test_wire_opportunity_store_requires_explicit_payload_codecs`, gates `q09`–`q10` | **RESOLVED / independently accepted** @ `e0d5209825de532363b62f0e1e58f53e1ac30e47` |
| **R2-P2-CODEC-REGISTRY-MUTABILITY-03** | Registry held mutable `dict` | Frozen dataclass with mutable dict field | `MappingProxyType` + construction validation; encode/decode identity checks | `tests/unit/integrations/test_integration_configuration_payload_codec_registry.py`, gate `q11` | **RESOLVED / independently accepted** @ `e0d5209825de532363b62f0e1e58f53e1ac30e47` |

## TRACE-X-P5-R2-P2-R2 — DocumentStore provenance pagination completeness

| Blocker | Root cause | Before | Remediation | Tests | Status |
|---|---|---|---|---|---|
| **R2-P2-DOCUMENT-PROVENANCE-PAGINATION-04** | `DocumentStoreExecutionIntegrationConfigurationPinningStore.read_all()` issued one `query(..., limit=1000)` and ignored `DocumentQueryPageV1.next_cursor`, returning only the first storage page | Single-page truncation for partitions &gt; 1000 rows (or provider-capped pages) | Bounded `while` traversal using opaque `cursor=` + `next_cursor`; fail-closed on repeated/cyclic cursors; duplicate row-key/subject guards; row-key ↔ decoded-subject integrity; deterministic sort unchanged | `test_document_provenance_read_all_paginates_all_records`, `test_document_provenance_provider_caps_page_below_requested_limit`, `test_document_provenance_read_all_matches_kv_with_pagination`, cursor adversarial + corruption tests, gates `q14`–`q15` | **RESOLVED / independently accepted** @ `660d9d9cd237ca91a6c4e662389b3ec0cf9fd20c` |

**Parent P2 closure** @ `660d9d9cd237ca91a6c4e662389b3ec0cf9fd20c` covers typed Opportunity persistence; durable exact `tenant_id` + `configuration_ref` access; execution provenance persistence; exact `tenant_id` + `ExecutionId` + `IntegrationConfigurationSubject` identity; immutable/idempotent writes; conflict fail-closed; explicit schema versions; corruption fail-closed; restart continuity; KV crash repair; explicit codec composition; immutable codec registry; DocumentStore complete pagination; KV/DocumentStore semantic parity; local tenant isolation; STATE-X current-head delta classification.

**Architecture Health Signal (informational):** **GREEN** — P2 remediations were durable correctness, crash consistency, codec activation hardening, and pagination completeness; no ownership/authority redesign or layer inversion.

**Future debt (not P2 blockers):** **CONFIG-X** — explicit codec/provider activation and composition; **COMPAT-X** — persisted schema v1 / codec contract evolution; **PROD-Q** — operational behavior of complete `read_all()` for large subject sets (memory, latency, resource usage; potential future streaming/paging interface). Current contract requires complete `read_all()`.

**P5-GAP-04** remains **IMPLEMENTATION IN PROGRESS** (P2 = persistence foundation only; production configured→effective→`ExecutionId`→pin chain = **TRACE-X-P5-R2-P3** onward).

**Next stage:** **TRACE-X-P5-R2-P3** = **NEXT / REQUIRED / NOT ENTERED** (production sequencing per [`TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md`](TRACE_X_P5_R2_CONFIGURED_EFFECTIVE_EXECUTION_PROVENANCE_ARCHITECTURE_LOCK.md); no P3 implementation in P2-CLOSE bookkeeping).

**Cursor safety:** `next_cursor == cursor` or any repeated cursor token → `CORRUPT_RECORD` (no partial provenance). Empty `documents` with `next_cursor` continues traversal.

**Backend parity:** KV `read_all()` semantics preserved; DocumentStore adapter now returns the complete subject set with the same deterministic ordering.

**Operational note (&gt;1000 rows):** Multi-page proof uses provider-capped pages (`max_page_size=2`, five records) rather than materializing 1001 heavyweight rows; default adapter page size remains `1000`.
