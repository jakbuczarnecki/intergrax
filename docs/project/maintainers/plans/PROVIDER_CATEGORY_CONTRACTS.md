# Provider Category Contracts — Implementation Plan

**Architecture (1:1):** [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md)
**Taxonomy source:** `intergrax/integrations/providers/layout.py` (`SLUG_CATEGORY`)  
**Code:** `intergrax/runtime/integrations/categories`
**Current follow-up:** INTEGRATIONS-3A-CONTRACT-REGISTRY-V2

**Last updated:** 2026-06-29 — INTEGRATIONS-3A contract registry v2 **In progress**; INTEGRATIONS-2E runtime cutover **Done** (185 slugs); 9 `llm_guardrail` slugs deferred.

---

## Status

| ID | Type | Priority | Status | Deliverable |
|----|------|----------|--------|-------------|
| **INTEGRATIONS-2A** | Code | P1 | **Done** | Category-specific base contracts for all `SLUG_CATEGORY` folders |
| **INTEGRATIONS-2B** | Code | P1 | **Done (Langfuse reference pilot)** | Concrete provider migration to category contracts (per slug) |
| **INTEGRATIONS-2B-FOLLOWUP** | Code | P1 | **Done** | Provider package pattern + scaffold hardening |
| **INTEGRATIONS-2C** | Code | P1 | **Done** | All `observability_backend` slugs migrated |
| **INTEGRATIONS-2D** | Code | P1 | **Done** | All remaining non-observability slugs migrated (160) or explicitly deferred (9 `llm_guardrail`) |
| **INTEGRATIONS-2E** | Code | P1 | **Done** | Runtime cutover — Integration is single public entrypoint; legacy factories are shims only (9 `llm_guardrail` deferred) |
| **INTEGRATIONS-3A-CONTRACT-REGISTRY-V2** | Code | P1 | **In progress** | Additive, contract-aware registry metadata for provider/category registrations; no runtime binding yet |

**INTEGRATIONS-2A acceptance:** every `SLUG_CATEGORY` folder has a category contract or explicit alias; all derive from **`PlatformIntegrationContract`**; `observability_backend` aliases **`ObservabilityVendorIntegrationContract`**; no LKW change; focused tests green.

**Contract migration vs runtime cutover (INTEGRATIONS-2E):** INTEGRATIONS-2D added contract-based classes in `integration.py` while legacy runtime adapters often remained as parallel public APIs (same class name in `adapter.py` and `integration.py` in some categories). INTEGRATIONS-2E completes migration: `<ProviderPascal><CategoryPascal>Integration` owns runtime behavior; legacy factories delegate to it; old public adapter/facade classes are removed or renamed private (`_ProviderRuntime`).

**Legacy factory shim policy:** `create_<slug>_<category>()` and slug-specific legacy names (e.g. `create_pinecone_vector_store`) may remain for import stability but MUST construct or return the new Integration class (directly or via `.from_store()` / `.as_*()` view owned by Integration). They must NOT return a separate public adapter class.

**Private helper policy:** SDK clients, bridges, mappers, and runtime helpers MAY remain as `_`-prefixed private modules/classes inside the provider package. They must not appear in `__init__.py` `__all__`, bundle public exports, or registry factory return types (unless the factory explicitly documents returning an Integration-owned view).

**Provider package pattern (INTEGRATIONS-2B-FOLLOWUP / 2D):** category contracts define the **base**; each concrete provider class in `integration.py` **derives from the category-specific contract** for its folder. The provider package layout (`integration.py`, `bundle.py`, `manifest.py`, `register.py`, `__init__.py`, `USAGE.md`) is the **implementation convention** — see [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md#provider-package-pattern-integrations-2b-follow-up). Langfuse is the observability reference pilot; **INTEGRATIONS-2D** applied the same package pattern across all non-observability categories.

**Completeness guard (INTEGRATIONS-2D):** `test_provider_category_contract_migration.py` derives expected slugs from `SLUG_CATEGORY` (excluding `observability_backend`) and compares against discovered `integration.py` packages; deferred slugs documented in test + plan.

**Scaffold hardening (INTEGRATIONS-SCAFFOLD-P5-P7-CONTRACT-AWARE):** maintenance provider shell generators (`wire_p2` through `wire_p7`) preserve contract-aware packages when `integration.py` exists.

**Deferred:** `llm_guardrail` per-slug packages (shared bundles layout), runtime binding/profile resolution, vendor SDK adapters, LKW wiring.

---

## Contract Registry v2 — INTEGRATIONS-3A

`intergrax/runtime/integrations/registry_v2.py` introduces an additive metadata registry for contract-aware provider registrations. It does **not** replace the legacy catalog yet and does **not** perform runtime binding.

**Registration identity:** `(provider_id, category)`.

- `provider_id` may repeat across categories, for example `elasticsearch` can later exist as `vector_store`, `search_provider`, and `observability_backend` registrations.
- `provider_id + category` must be unique; duplicate registrations raise `DuplicateIntegrationRegistrationError`.
- `category` comes from `SLUG_CATEGORY` / provider folder taxonomy.
- `integration_kind` is stored separately and validated against the category contract. For `observability_backend`, the category remains `observability_backend` while `integration_kind` remains `observability_vendor`.

**Typed metadata:** `IntegrationRegistration` stores `provider_id`, `slug`, `category`, `integration_kind`, `contract_class`, `integration_class`, contract factory, `config_class`, display name, capabilities, security posture, `default_enabled=False`, health/runtime-binding support flags, and safe metadata.

**Builder behavior:** `build_integration_registration(slug)` derives provider metadata from:

- `intergrax/integrations/providers/layout.py` (`SLUG_CATEGORY`)
- `PROVIDER_CATEGORY_CONTRACT_REGISTRY`
- provider `integration.py` single public Integration class
- provider `bundle.py` contract factory (`create_<slug>_<category>_integration` or `create_<slug>_observability_integration`)

The builder calls the contract factory only with `enabled=False` to validate the disabled integration shape. It must not initialize vendor SDK clients, read secrets, perform network I/O, activate providers from environment, replace bootstrap, or create tenant/workspace/application bindings.

**Registry behavior:** `IntegrationRegistry` can register, get, list all, list by category, and list by provider. It is deterministic and isolated; no global provider bootstrap is changed in INTEGRATIONS-3A.

**Deferred/excluded:** the 9 `llm_guardrail` slugs remain explicitly excluded from registry v2 completeness until INTEGRATIONS-2F package normalization.

**Next phase:** INTEGRATIONS-3B will own explicit integration binding / provider profile resolution. LKW adoption remains after INTEGRATIONS-3B and LKW-H1.

---

## Category → contract mapping

Full list from `layout.py` `SLUG_CATEGORY` unique folder values (31 categories).

| Provider folder (`SLUG_CATEGORY`) | Contract class | `schema_id` | `integration_kind` | Notes |
|-----------------------------------|----------------|-------------|-------------------|-------|
| `relational_store` | `RelationalStoreIntegrationContract` | `relational_store_integration_contract.v1` | `relational_store` | |
| `document_store` | **`DocumentStoreVendorIntegrationContract`** | `document_store_vendor_integration_contract.v1` | `document_store` | **Alias** — replaces removed `DocumentStoreIntegrationContract` |
| `key_value_cache` | `KeyValueCacheIntegrationContract` | `key_value_cache_integration_contract.v1` | `key_value_cache` | |
| `message_bus` | `MessageBusIntegrationContract` | `message_bus_integration_contract.v1` | `message_bus` | |
| `object_storage` | `ObjectStorageIntegrationContract` | `object_storage_integration_contract.v1` | `object_storage` | Legacy shorthand: `PlatformIntegrationKind.STORAGE` |
| `vector_store` | `VectorStoreIntegrationContract` | `vector_store_integration_contract.v1` | `vector_store` | |
| `search_provider` | `SearchProviderIntegrationContract` | `search_provider_integration_contract.v1` | `search_provider` | Legacy shorthand: `PlatformIntegrationKind.SEARCH` |
| `notification_channel` | `NotificationChannelIntegrationContract` | `notification_channel_integration_contract.v1` | `notification_channel` | Legacy shorthand: `PlatformIntegrationKind.NOTIFICATION` |
| `conversation_channel` | `ConversationChannelIntegrationContract` | `conversation_channel_integration_contract.v1` | `conversation_channel` | Near-real-time bidirectional human↔app chat; distinct from notify-only `notification_channel` — [`CONVERSATION_CHANNEL_CONTRACT.md`](../../architecture/CONVERSATION_CHANNEL_CONTRACT.md) |
| `model_serving_runtime` | `ModelServingRuntimeIntegrationContract` | `model_serving_runtime_integration_contract.v1` | `model_serving_runtime` | Replaces removed `interaction_surface`; self-hosted model hosts (Ollama, …) |
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
| Provider folder | `document_store` | `DocumentStoreVendorIntegrationContract` |
| Runtime `integration_kind` | `observability_vendor` | Used by observability vendor integrations |
| `PlatformIntegrationKind` | `OBSERVABILITY_BACKEND` | Documents folder taxonomy |
| `PlatformIntegrationKind` | `OBSERVABILITY_VENDOR` | Runtime integration kind for observability contracts |
| Legacy shorthand | `search` | Prefer `search_provider` for new code |
| Legacy shorthand | `storage` | Prefer `object_storage` for new code |
| Legacy shorthand | `notification` | Prefer `notification_channel` for new code |

**Multi-category providers:** same `provider_id` (for example `elasticsearch`, `slack`) must use **separate integration classes** per category — never one multi-category class. Primary package folder remains in `SLUG_CATEGORY`; secondary memberships use `SECONDARY_PROVIDER_CATEGORIES` (for example `slack` → `notification_channel` primary + `conversation_channel` secondary). Registry identity remains `(provider_id, category)`.

---

## Recommended migration order (INTEGRATIONS-2B+)

1. **observability_backend** — Langfuse, Arize, Phoenix, Elasticsearch (extends existing **`ObservabilityVendorIntegrationContract`**; OTLP done in INTEGRATIONS-1C)
2. **vector_store** — Pinecone, Qdrant, Weaviate (high RAG demand)
3. **object_storage** — S3, GCS, MinIO
4. **search_provider** — Tavily, Algolia
5. **message_bus** — Kafka, RabbitMQ
6. **relational_store** / **document_store** — Postgres, MongoDB
7. **notification_channel** — Slack webhook/email notify
8. **conversation_channel** — Slack (Socket Mode + Web API runtime), Teams/Discord/Telegram/Mattermost/Rocket.Chat/Google Chat (contract-defined; vendor runtime unbound)
9. **secrets_store** — Vault, cloud secret managers
10. Remaining categories per product priority (issue_tracker, ci_cd, llm_guardrail, …)

**Per-slug rule:** one PR per slug (or small harden wave ≤4 slugs); subclass the category contract; no vendor SDK in runtime hot paths; config disabled by default.

---

## Tests

`tests/unit/runtime/integrations/test_provider_category_contracts.py` — registry coverage, inheritance, observability alias, schema_id stability, config safety, multi-category identity separation.

`tests/unit/runtime/integrations/test_contract_registry_v2.py` — registry v2 model validation, duplicate identity, same provider across categories, disabled factory compatibility, `observability_backend` alias handling, deferred `llm_guardrail` exclusion, and no SDK/network initialization guard.

Focused run:

```bash
uv run pytest tests/unit/runtime/integrations/test_contract_registry_v2.py tests/unit/runtime/integrations/test_platform_integration_contract.py tests/unit/runtime/integrations/test_observability_vendor_integration_contract.py tests/unit/runtime/integrations/test_provider_category_contracts.py -q
```
