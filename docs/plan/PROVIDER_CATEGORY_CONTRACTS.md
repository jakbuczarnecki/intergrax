# Provider Category Contracts — Implementation Plan

**Architecture (1:1):** [`architecture/INTEGRATIONS.md`](../architecture/INTEGRATIONS.md)  
**Taxonomy source:** `intergrax/integrations/providers/layout.py` (`SLUG_CATEGORY`)  
**Code:** `intergrax/runtime/integrations/categories/`  
**Task:** INTEGRATIONS-2A

**Last updated:** 2026-06-28 — INTEGRATIONS-2A provider category contracts implemented.

---

## Status

| ID | Type | Priority | Status | Deliverable |
|----|------|----------|--------|-------------|
| **INTEGRATIONS-2A** | Code | P1 | **Done** | Category-specific base contracts for all `SLUG_CATEGORY` folders |
| **INTEGRATIONS-2B** | Code | P1 | **Done (Langfuse reference pilot)** | Concrete provider migration to category contracts (per slug) |
| **INTEGRATIONS-2B-FOLLOWUP** | Code | P1 | **Done** | Provider package pattern + scaffold hardening |

**INTEGRATIONS-2A acceptance:** every `SLUG_CATEGORY` folder has a category contract or explicit alias; all derive from **`PlatformIntegrationContract`**; `observability_backend` aliases **`ObservabilityVendorIntegrationContract`**; no concrete provider migration; no LKW change; focused tests green.

**Provider package pattern (INTEGRATIONS-2B-FOLLOWUP):** category contracts define the **base**; each concrete provider class in `integration.py` **derives from the category-specific contract** for its folder. The provider package layout (`integration.py`, `bundle.py`, `manifest.py`, `register.py`, `__init__.py`, `USAGE.md`) is the **implementation convention** — see [`architecture/INTEGRATIONS.md`](../architecture/INTEGRATIONS.md#provider-package-pattern-integrations-2b-follow-up). Langfuse is the reference pilot; batch migration of remaining slugs remains deferred.

**Scaffold hardening (INTEGRATIONS-SCAFFOLD-P5-P7-CONTRACT-AWARE):** maintenance provider shell generators (`wire_p2` through `wire_p7`) preserve contract-aware packages when `integration.py` exists. Provider category migration can proceed in waves after scaffold validation; full batch migration remains deferred.

**Deferred:** batch provider migration, registry v2 / contract registry wiring, vendor SDK adapters, LKW wiring.

---

## Category → contract mapping

Full list from `layout.py` `SLUG_CATEGORY` unique folder values (31 categories).

| Provider folder (`SLUG_CATEGORY`) | Contract class | `schema_id` | `integration_kind` | Notes |
|-----------------------------------|----------------|-------------|-------------------|-------|
| `relational_store` | `RelationalStoreIntegrationContract` | `relational_store_integration_contract.v1` | `relational_store` | |
| `document_store` | `DocumentStoreIntegrationContract` | `document_store_integration_contract.v1` | `document_store` | |
| `key_value_cache` | `KeyValueCacheIntegrationContract` | `key_value_cache_integration_contract.v1` | `key_value_cache` | |
| `message_bus` | `MessageBusIntegrationContract` | `message_bus_integration_contract.v1` | `message_bus` | |
| `object_storage` | `ObjectStorageIntegrationContract` | `object_storage_integration_contract.v1` | `object_storage` | Legacy shorthand: `PlatformIntegrationKind.STORAGE` |
| `vector_store` | `VectorStoreIntegrationContract` | `vector_store_integration_contract.v1` | `vector_store` | |
| `search_provider` | `SearchProviderIntegrationContract` | `search_provider_integration_contract.v1` | `search_provider` | Legacy shorthand: `PlatformIntegrationKind.SEARCH` |
| `notification_channel` | `NotificationChannelIntegrationContract` | `notification_channel_integration_contract.v1` | `notification_channel` | Legacy shorthand: `PlatformIntegrationKind.NOTIFICATION` |
| `interaction_surface` | `InteractionSurfaceIntegrationContract` | `interaction_surface_integration_contract.v1` | `interaction_surface` | |
| `collaboration_suite` | `CollaborationSuiteIntegrationContract` | `collaboration_suite_integration_contract.v1` | `collaboration_suite` | |
| `issue_tracker` | `IssueTrackerIntegrationContract` | `issue_tracker_integration_contract.v1` | `issue_tracker` | |
| `wiki_knowledge` | `WikiKnowledgeIntegrationContract` | `wiki_knowledge_integration_contract.v1` | `wiki_knowledge` | |
| `observability_backend` | **`ObservabilityVendorIntegrationContract`** | `observability_vendor_integration_contract.v1` | **`observability_vendor`** | **Alias** — no duplicate contract; folder ≠ `integration_kind` |
| `browser_automation` | `BrowserAutomationIntegrationContract` | `browser_automation_integration_contract.v1` | `browser_automation` | |
| `cloud_platform` | `CloudPlatformIntegrationContract` | `cloud_platform_integration_contract.v1` | `cloud_platform` | |
| `secrets_store` | `SecretsStoreIntegrationContract` | `secrets_store_integration_contract.v1` | `secrets_store` | |
| `graph_store` | `GraphStoreIntegrationContract` | `graph_store_integration_contract.v1` | `graph_store` | |
| `document_parser` | `DocumentParserIntegrationContract` | `document_parser_integration_contract.v1` | `document_parser` | |
| `rerank_provider` | `RerankProviderIntegrationContract` | `rerank_provider_integration_contract.v1` | `rerank_provider` | |
| `feature_flag` | `FeatureFlagIntegrationContract` | `feature_flag_integration_contract.v1` | `feature_flag` | |
| `ci_cd` | `CiCdIntegrationContract` | `ci_cd_integration_contract.v1` | `ci_cd` | |
| `security_scanner` | `SecurityScannerIntegrationContract` | `security_scanner_integration_contract.v1` | `security_scanner` | |
| `sandbox_host` | `SandboxHostIntegrationContract` | `sandbox_host_integration_contract.v1` | `sandbox_host` | |
| `identity_provider` | `IdentityProviderIntegrationContract` | `identity_provider_integration_contract.v1` | `identity_provider` | |
| `speech_provider` | `SpeechProviderIntegrationContract` | `speech_provider_integration_contract.v1` | `speech_provider` | |
| `workflow_orchestrator` | `WorkflowOrchestratorIntegrationContract` | `workflow_orchestrator_integration_contract.v1` | `workflow_orchestrator` | |
| `billing_meter` | `BillingMeterIntegrationContract` | `billing_meter_integration_contract.v1` | `billing_meter` | |
| `crm` | `CrmIntegrationContract` | `crm_integration_contract.v1` | `crm` | |
| `vision_serving` | `VisionServingIntegrationContract` | `vision_serving_integration_contract.v1` | `vision_serving` | |
| `ml_inference_host` | `MlInferenceHostIntegrationContract` | `ml_inference_host_integration_contract.v1` | `ml_inference_host` | |
| `llm_guardrail` | `LlmGuardrailIntegrationContract` | `llm_guardrail_integration_contract.v1` | `llm_guardrail` | |

**Registry:** `PROVIDER_CATEGORY_CONTRACT_REGISTRY` in `categories/__init__.py` maps folder name → contract class.

---

## Alias and mapping notes

| Concept | Value | Maps to |
|---------|-------|---------|
| Provider folder | `observability_backend` | `ObservabilityVendorIntegrationContract` |
| Runtime `integration_kind` | `observability_vendor` | Used by observability vendor integrations |
| `PlatformIntegrationKind` | `OBSERVABILITY_BACKEND` | Documents folder taxonomy |
| `PlatformIntegrationKind` | `OBSERVABILITY_VENDOR` | Runtime integration kind for observability contracts |
| Legacy shorthand | `search` | Prefer `search_provider` for new code |
| Legacy shorthand | `storage` | Prefer `object_storage` for new code |
| Legacy shorthand | `notification` | Prefer `notification_channel` for new code |

**Multi-category providers:** same `provider_id` (for example `elasticsearch`) must use **separate integration classes** per category — never one multi-category class.

---

## Recommended migration order (INTEGRATIONS-2B+)

1. **observability_backend** — Langfuse, Arize, Phoenix, Elasticsearch (extends existing **`ObservabilityVendorIntegrationContract`**; OTLP done in INTEGRATIONS-1C)
2. **vector_store** — Pinecone, Qdrant, Weaviate (high RAG demand)
3. **object_storage** — S3, GCS, MinIO
4. **search_provider** — Tavily, Algolia
5. **message_bus** — Kafka, RabbitMQ
6. **relational_store** / **document_store** — Postgres, MongoDB
7. **notification_channel** — Slack, email
8. **secrets_store** — Vault, cloud secret managers
9. Remaining categories per product priority (issue_tracker, ci_cd, llm_guardrail, …)

**Per-slug rule:** one PR per slug (or small harden wave ≤4 slugs); subclass the category contract; no vendor SDK in runtime hot paths; config disabled by default.

---

## Tests

`tests/unit/runtime/integrations/test_provider_category_contracts.py` — registry coverage, inheritance, observability alias, schema_id stability, config safety, multi-category identity separation.

Focused run:

```bash
uv run pytest tests/unit/runtime/integrations/test_platform_integration_contract.py tests/unit/runtime/integrations/test_observability_vendor_integration_contract.py tests/unit/runtime/integrations/test_provider_category_contracts.py -q
```
