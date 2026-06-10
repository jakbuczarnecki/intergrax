# Integrations

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 13–14  
**Audit instruction:** [`guides/audit/INTEGRATIONS.md`](../guides/audit/INTEGRATIONS.md)  
---

# 18. Slack / Teams / Communication Integration Philosophy

Intergrax should support Slack and Teams as interaction surfaces.

This follows the Viktor-like idea where an AI worker can live inside organizational communication tools.

Slack and Teams should be implemented as adapters.

They may provide:

- task intake
- notifications
- approval requests
- progress updates
- final responses
- interactive buttons
- user context
- channel context

They should NOT own the runtime.

Correct model:

```text
Slack message
    -> SlackAdapter
    -> normalized Task
    -> Nexus Runtime
    -> Agent execution
    -> Nexus final result
    -> SlackAdapter sends response
```

Incorrect model:

```text
Slack bot contains orchestration logic
Slack bot directly manages agents
Slack bot stores global task state
```

---


---

# 46. Checklist For New Adapter Implementation

Before implementing a new adapter, answer:

```text
1. What external system does it connect to?
2. What operations does it expose?
3. What permissions are required?
4. Is it read-only or write-capable?
5. What are risk levels?
6. What errors can happen?
7. What timeout/retry policy is needed?
8. What data should be logged?
9. What data must be protected?
10. Which agents or runtime components may use it?
```

Adapters should be generic and reusable.

---

---

## Catalog


**Last updated:** 2026-06-10 (RAG domain pair split · M.7 P7 **Done** 18/18)

The **Integration Library** (`intergrax/integrations/`) is Intergrax’s modular catalog of external systems — databases, queues, search APIs, vector indexes, cloud platforms, and collaboration tools. Agents and applications wire backends **by category**, not by vendor SDK, so the same agent code can run in a local lab, a customer VPC, or a multi-cloud deployment.

**Related docs:**

| Document | Purpose |
|----------|---------|
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1 | Architecture canon — tiers, contracts, registry rules |
| [plan/INTEGRATIONS.md) Phase M | Phase status, backlog, delivery workflow |
| [guides/AGENT_CREATION_GUIDE.md](guides/AGENT_CREATION_GUIDE.md) Appendix E | How agents vs applications use integrations |
| [architecture/RAG.md](RAG.md) | RAG retrieval engine (consumes integration slugs) |
| [architecture/TOOLS.md](architecture/TOOLS.md) | Agent-facing tools that compose these integrations |
| Per-provider guides | `intergrax/integrations/providers/<category>/<slug>/USAGE.md` |
| [../infra/README.md](../infra/README.md) | **Local Docker infrastructure** — compose profiles, manage scripts |
| [../infra/PORTS.md](../infra/PORTS.md) | Host port matrix for integration tests |
| [guides/HARNESS_ENVIRONMENT.md](guides/HARNESS_ENVIRONMENT.md) | Lab harness stack, OTLP, verification |

---

## Harness lab stable stack (Phase S / T)

The **lab harness environment** treats these catalog slugs as **`stable`** (production-ready for the reference lab stack). Source of truth: `intergrax/integrations/registry/harness_lab_stack.py`.

| Slug | Category |
|------|----------|
| `sqlite` | relational_store |
| `postgresql` | relational_store (Tier-2 product apps) |
| `redis` | key_value_cache |
| `qdrant` | vector_store |
| `slack` | notification_channel + interaction_surface |
| `sentry` | observability_backend |
| `otel` | observability_backend |
| `lab_json` | interaction_surface |
| `log` | notification_channel |

```bash
uv run pytest tests/unit/integrations/test_harness_lab_stable_stack.py -m gate -q
```

Other slugs remain **`beta`** unless promoted explicitly. Do not mark all 185 providers stable in one release.

### M.6 P5 — Harness integration depth (Done — 33/34)

**Register:** [intergrax_runtime_architecture.md — M.6 P5](../plan/INTEGRATIONS.md#m6-p5--harness-integration-depth-done--3334) · Band **2ab**

| Wave | Focus | Status |
|------|--------|--------|
| H-INT-6 | Ops, metrics, multi-CI, local cloud | **Done** (10/10) |
| H-INT-7 | Eval observability, async bus, artifacts | **Done** (10/10) |
| H-INT-8 | Data plane lab (graph, document, logs, vectors) | **Done** (8/8) |
| H-INT-9 | P2 reserve | **Done** (5/6 — `trivy` deferred) |

**Delivered:** 25 harden (STABLE + `IntegrationHealthProbe`) · 8 greenfield (`_shared/p6`) · 4 Tier-3 presets · `HARNESS_M6_P5_PROBE_SLUGS` · debug API `GET /debug/integrations/health?stack=m6_p5`.

**Tier-3 presets (P5):** `harness_metrics_stack()`, `harness_eval_stack()`, `harness_async_stack()`, `harness_ci_stack()` — CLI: `intergrax integrations-pick harness_metrics|harness_eval|harness_async|harness_ci`.

**Deferred:** `trivy` — absorbed into **M.6 P6** [M-P6.1](../plan/INTEGRATIONS.md#m6-p6--master-register-32-slugs) (`security_scanner` / **M-P6-CAT.1**).

### M.6 P6 — Harness integration expansion (Done — 32/32)

**Register:** [intergrax_runtime_architecture.md — M.6 P6](../plan/INTEGRATIONS.md#m6-p6--harness-integration-expansion-planned) · Band **2ac** · Queue **[§6.1y](../plan/INTEGRATIONS.md#61y-harness-implementation-queue--integration-expansion-m6-p6-planned)**

| Wave | Focus | Slugs | Status |
|------|--------|-------|--------|
| H-INT-10 | Security + secrets | `trivy`, `snyk`, `semgrep`, `infisical` | **Done** |
| H-INT-11 | Cloud sandbox | `e2b`, `modal`, `daytona` | **Done** |
| H-INT-12 | Identity / tenant IAM | `auth0`, `keycloak`, `workos` | **Done** |
| H-INT-13 | GitOps CI | `argocd`, `buildkite`, `jenkins` | **Done** |
| H-INT-14 | Speech catalog | `elevenlabs`, `deepgram` | **Done** |
| H-INT-15 | Enterprise ops | `newrelic`, `splunk`, `zendesk`, `statsig` | **Done** |
| H-INT-16 | Data / workflow | `prefect`, `airflow`, `typesense`, `neon`, `pulsar` | **Done** |
| H-INT-17 | Reserve | `algolia`, `confluent`, `backblaze_b2`, `triton`, `replicate`, `stripe`, `salesforce`, `hubspot` | **Done** |

**New categories (9):** `security_scanner`, `sandbox_host`, `identity_provider`, `speech_provider`, `workflow_orchestrator`, `vision_serving`, `ml_inference_host`, `billing_meter`, `crm`.

**Post-catalog wiring (M-P6-WIRE):** `wire_integration_tool_context()` resolves P6 slots into `ToolWiringContext`; `extend_tool_profile_for_integration()` auto-enables `security.scan`, `workflow.*`, and `sandbox.exec` when matching categories are configured. Speech catalog slugs bridge to Tier-0 speech tools via `IntegrationSpeechAdapter`.

**Delivered:** 32 STABLE slugs (`_shared/p7`) · 9 category contracts · 4 Tier-3 presets · `HARNESS_M6_P6_PROBE_SLUGS` · debug API `GET /debug/integrations/health?stack=m6_p6`.

**Tier-3 presets (P6):** `harness_security_stack()`, `harness_sandbox_stack()`, `harness_identity_stack()`, `harness_gitops_stack()` — CLI: `intergrax integrations-pick harness_security|harness_sandbox|harness_identity|harness_gitops`.

### M.7 P7 — Agent-developer integration expansion (Done — 18/18)

**Register:** [plan/INTEGRATIONS.md — M.7 P7](../plan/INTEGRATIONS.md#m7-p7--agent-developer-integration-expansion-done--1818) · Band **2ad**

| Wave | Focus | Slugs | Status |
|------|--------|-------|--------|
| H-INT-P7-1 | Research + RAG | `perplexity`, `arxiv`, `semantic_scholar`, `llamaparse`, `lancedb` | **Done** |
| H-INT-P7-2 | Interaction + browser + storage | `telegram`, `browserbase`, `google_drive`, `apify` | **Done** |
| H-INT-P7-3 | Workflow + wiki + identity + cache | `n8n`, `wikipedia`, `clerk`, `upstash_redis`, `upstash_qstash` | **Done** |
| H-INT-P7-4 | Data warehouse | `okta`, `bigquery`, `motherduck`, `airbyte` | **Done** |

**Delivered:** 18 STABLE slugs (`_shared/p8`) · 3 Tier-3 agent presets · `HARNESS_M7_P7_PROBE_SLUGS` · auto-wiring `search_provider` / `document_parser` / `vector_store` → catalog tools.

**Tier-3 presets (P7):** `research_web_stack()`, `document_ingest_stack()`, `chat_bot_stack()` — CLI: `intergrax integrations-pick research_web|document_ingest|chat_bot`.

**Catalog:** **185** slugs in `layout.py` (**12** core / **185** full preset).

---

## Local infrastructure (Docker)

Run backing services locally before integration tests or lab hosts. Unified stack: `infra/integration/` with **compose profiles** (`core`, `queue`, `rag`, `data`, `secrets`, `observability`, `cloud`, `heavy`, `p6`).

```bash
cd infra/integration && ./manage.sh start          # default profiles
cd infra/integration && ./manage.sh start rag      # vectors + neo4j + ollama + docling
cd infra/integration && ./manage.sh start p6       # keycloak + typesense + airflow + core (PostgreSQL)
cd infra/integration && ./manage.sh start all      # full stack
```

See [infra/PORTS.md](../infra/PORTS.md) for host ports (e.g. Redis `6379`, Qdrant `6333`, Neo4j Bolt `7687`, Weaviate `8080`, MinIO `9000`, Vault `8200`, ClickHouse HTTP `8123` / native `9002`).

**SaaS-only slugs** (no local container — use mocks or API keys): `slack`, `jira`, `confluence`, `google_cse`, `pinecone`, `cohere_rerank`, `sentry` (cloud), most `observability_backend` HTTP proxies unless self-hosted image is listed in infra.

---

## Provider layout (by category)

Integrations are grouped under **contract category** folders — the same grouping used when generating P2/P3 provider stubs:

```text
intergrax/integrations/providers/
├── layout.py                 # slug → category map
├── relational_store/         # sqlite, postgresql, mysql, …
├── document_store/           # mongodb, cassandra, dynamodb
├── key_value_cache/          # redis, memcached, elasticache
├── message_bus/              # kafka, sqs, pubsub, …
├── object_storage/           # s3, azure_blob, gcs
├── vector_store/             # pinecone, qdrant, chroma, weaviate, milvus, inmemory, vespa
├── search_provider/          # google_cse, bing, reddit, google_places, brave, serpapi, tavily, exa
├── notification_channel/     # slack, teams, discord, twilio, pagerduty, opsgenie, …
├── interaction_surface/      # lab_json, slash_command (slack/teams also register here)
├── collaboration_suite/      # ms365_graph, google_workspace
├── issue_tracker/            # jira, github, linear, azure_devops, gitlab
├── wiki_knowledge/           # confluence, notion, sharepoint
├── observability_backend/    # prometheus, elasticsearch, otel, langfuse, datadog, clickhouse, sentry, langsmith, …
├── document_parser/          # docling, pymupdf, unstructured, python_docx, openpyxl, whisper, yt_dlp
├── rerank_provider/          # cohere_rerank, jina_rerank
├── browser_automation/       # playwright, firecrawl, selenium
├── secrets_store/            # vault
├── graph_store/              # neo4j
└── cloud_platform/           # aws, azure, gcp
```

**Import path:** `from intergrax.integrations.providers.object_storage.s3.bundle import create_s3_object_storage`

Catalog identity is the string **slug** (`"s3"`, `"postgresql"`) registered at runtime — not a central enum.

---

## Open catalog (no slug enum)

| Mechanism | Role |
|-----------|------|
| `providers/<category>/<slug>/manifest.py` | `MANIFEST = IntegrationManifest(slug=…)` — canonical metadata per provider |
| `register_from_manifest(MANIFEST, factory)` | Registers in runtime catalog (`registry/catalog.py`) |
| `IntegrationProfile` | Declares slot: manifest, plugin class, slug `str`, or pre-built instance |
| `profile.resolve(IntegrationCategory.…)` | Instantiates via registered factory |
| `catalog_manifests.py` | Lightweight **preset** copies for lab/product profiles only (not exhaustive) |
| `IntegrationPlugin` | External packages: `integration_manifest()` + `create_integration()` |
| `bootstrap_catalogs()` | Unified Tier-3 bootstrap; `integration_preset="core"` or `"full"` |

Third-party integrations **must not** extend a core enum. Register a plugin or call `register_from_manifest` from application startup.

**Shipped vs plugin class:** ~167 providers register via `register_from_manifest(MANIFEST, create_*)`. External pip packages should implement `IntegrationPlugin` (`integration_manifest()` + `create_integration()`). `SqliteIntegrationPlugin` in `providers/relational_store/sqlite/plugin.py` documents the class-based pattern; shipped `register.py` keeps the manifest path for bootstrap performance.

Tier-3 hosts should call `bootstrap_application_integration_catalog()` (not bare `register_default_integrations()`).

### Named integration presets (Phase DX-4.3)

Typed factories in `intergrax.integrations.registry.presets` — use in `ApplicationManifest` / `host/environment_profile.py`:

| Preset function | Returns | Typical use |
|-----------------|---------|-------------|
| `lab_stack(enable_otel=True)` | `IntegrationProfile.lab_harness_preset` | Default lab / scaffold hosts |
| `legal_stack()` | `IntegrationProfile.legal_product()` | Legal product relational + vector |
| `research_stack()` | `IntegrationProfile.research_product()` | Research product search + vector |
| `data_stack(enable_redis=True, enable_qdrant=False)` | Lab harness + optional redis/qdrant | Data-heavy experiments |
| `observability_stack(enable_otel=True, enable_grafana_stack=False)` | Lab harness OTEL-first; optional Grafana/Loki/Tempo triad | Trace/metrics focus |
| `harness_production_stack(secrets_slug="doppler", enable_grafana_stack=True)` | PostgreSQL + pgvector + secrets + Grafana stack + Unleash + GitHub Actions | Harness production Tier-3 (no business agents) |

CLI fragment helper: `uv run intergrax integrations pick postgres` (presets: `lab`, `legal`, `research`, `data`, `observability`, `harness_production`). See `intergrax/cli/integrations_pick.py`.

See [guides/EXTENSION_AUTHOR_GUIDE.md](guides/EXTENSION_AUTHOR_GUIDE.md), `intergrax/integrations/examples/custom_memory_kv/`, and `tests/unit/integrations/test_external_plugin.py`.

Scaffold a new provider tree: `python -m intergrax.scaffold new-integration <slug> --category <category>`.

---

## Design principles

| Principle | What it means |
|-----------|---------------|
| **Universal contracts** | Each category (`relational_store`, `vector_store`, `message_bus`, …) defines a small Protocol. Providers implement the contract; agent logic depends on the contract only. |
| **Modular providers** | One slug = one package under `providers/<category>/<slug>/` (category = contract name). Swap Redis for ElastiCache, SQLite for PostgreSQL, or Chroma for Pinecone by changing `IntegrationProfile` — no agent refactor. |
| **Environment portability** | Tier-3 applications compose integrations at startup (`IntegrationProfile`, env vars). The same Tier-2 agent runs against lab defaults (`sqlite`, `log`, `lab_json`) or production stacks (`postgresql`, `slack`, `s3`, `qdrant`). |
| **Single entry for SDKs** | Vendor SDKs (boto3, PyMongo, chromadb, redis, …) are imported only in boundary modules: `opens.py`, `rag_store.py`, `web_client.py`, `client.py`, and `_shared/p2|p3|p4/factories.py`. CI enforces this via `scripts/check_integration_vendor_imports.py`. Tier-2 agents must **not** import provider slugs or vendor libraries. |
| **Catalog registration** | `register_default_integrations(preset="full")` or `preset="core"` (lab). Resolution: explicit slug → profile field → env → cloud defaults. |

---

## How wiring works

```text
Tier-3 application (integration_wiring.py)
        │
        ▼
IntegrationProfile  ──►  IntegrationRegistry.resolve(category)
        │                        │
        │                        ▼
        │                 providers/<slug>/bundle.py
        │                        │
        ▼                        ▼
   env + options            category contract instance
                                   │
                                   ▼
                         passed into runtime / RAG / tools
```

Agents consume integrations **through catalog tools** ([architecture/TOOLS.md](architecture/TOOLS.md)), not by importing provider adapters. Tier-3 may also pass resolved contracts into `ToolWiringContext` for tool handlers.

**Example — declarative profile:**

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.catalog_manifests import POSTGRESQL, QDRANT
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(
    relational_store=POSTGRESQL,
    vector_store=QDRANT,
    object_storage="s3",
    notification_channel="slack",
    options={
        "s3": {"bucket": "intergrax-artifacts", "prefix": "tenant-a"},
    },
)

store = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

**Example — lab defaults (no external vendors):**

```python
profile = IntegrationProfile.lab()
# relational_store → sqlite, notification_channel → log, interaction_surface → lab_json
```

**Example — product profile with SQLite observability fallback:**

Product profiles such as `IntegrationProfile.legal_product()` may omit `relational_store`. Tier-3 factories pass the profile to `wire_nexus_observability()`; when SQLite is not declared on the profile, trace and runtime-event stores fall back to default `build/` SQLite paths (same as pre-profile wiring).

```python
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.integrations.registry.profile import IntegrationProfile

observability = wire_nexus_observability(
    integration_profile=IntegrationProfile.legal_product(),
)
```

Explicit SQLite bundle (lab / tests):

```python
from intergrax.runtime.persistence.integration_profile_wiring import open_trace_store_from_profile

profile = IntegrationProfile(
    relational_store="sqlite",
    options={"sqlite": {"data_dir": "build/lab"}},
)
trace_store = open_trace_store_from_profile(profile)
```

---

## Category contracts

| Category | Contract | Typical use |
|----------|----------|-------------|
| `relational_store` | `RelationalStore` | SQL persistence, analytics warehouses |
| `document_store` | `DocumentStore` | Flexible JSON / wide-column documents |
| `key_value_cache` | `KeyValueCache` | Idempotency, rate limits, locks — **not** session or user LTM memory (see Phase MEM / `AGENT_CREATION_GUIDE` Appendix G) |
| `message_bus` | `MessageBus` | Async task queues, worker transport |
| `object_storage` | `ObjectStorage` | Artifacts, exports, large file handoff |
| `vector_store` | `VectorStore` | RAG embedding indexes |
| `search_provider` | `SearchProvider` | Web / API research |
| `notification_channel` | `NotificationChannel` | Outbound alerts (HITL, progress) |
| `interaction_surface` | `InteractionSurface` | Inbound webhooks / chat intake |
| `collaboration_suite` | `CollaborationSuite` | Mail, calendar, directory |
| `issue_tracker` | `IssueTracker` | Issues, comments, search |
| `wiki_knowledge` | `WikiKnowledge` | Runbooks, internal docs |
| `observability_backend` | `ObservabilityBackend` | Metrics, log search, error tracking (Sentry) |
| `browser_automation` | `BrowserAutomation` | Dynamic web pages (JS-heavy sites) |
| `secrets_store` | `SecretsStore` | Tenant API keys, credentials (Vault, …) |
| `graph_store` | `GraphStore` | Agent memory, tool dependency graphs |
| `document_parser` | `DocumentParser` | Document/media parsing (Docling, PyMuPDF, Unstructured, python-docx, openpyxl, whisper, yt_dlp) |
| `rerank_provider` | `RerankProvider` | Vendor reranking APIs (cohere_rerank, jina_rerank) — consumed by RAG `rerankers/` |
| `security_scanner` | `SecurityScannerBackend` | SAST/dependency scans (trivy, snyk, semgrep) — CI and release gates |
| `llm_guardrail` | `LlmGuardrailBackend` | LLM I/O safety scanners (LLM Guard, Guardrails AI, NeMo, OpenGuardrails) — §47 |
| `cloud_platform` | `CloudPlatform` | Multi-service auth + category defaults |

Contract modules: `intergrax/integrations/contracts/`.

---

## Cloud platform facades

Platform providers resolve **default slugs** for infrastructure categories when an application sets `cloud_platform` and leaves category fields empty.

| Platform | Auth | Default `object_storage` | Default `message_bus` | Default `document_store` | Default `key_value_cache` |
|----------|------|--------------------------|------------------------|--------------------------|---------------------------|
| `aws` | IAM keys, profile, STS assume-role | `s3` | `sqs` | `dynamodb` | `elasticache` |
| `azure` | Managed identity, service principal | `azure_blob` | `service_bus` | — | — |
| `gcp` | ADC, service account JSON | `gcs` | `pubsub` | — | — |

Service-level slugs (`s3`, `azure_blob`, `gcs`, …) remain available for explicit or multi-cloud setups.

---

## Implemented providers (185)

All providers below are registered in `register_default_integrations()`.  
**Status:** `stable` = production-ready catalog entry; `beta` = shipped, API may evolve.

- **P2** slugs delegate to `intergrax/integrations/_shared/p2/factories.py`
- **P3 / M.7** harness slugs delegate to `intergrax/integrations/_shared/p3/factories.py`
- **P5 / M.6 P4** (2026-06-02) delegate to `intergrax/integrations/_shared/p5/factories.py`
- Thin shells live under `providers/<category>/<slug>/` — see `providers/layout.py`

### Summary by category

| Category | Count | Slugs |
|----------|------:|-------|
| `relational_store` | 14 | `sqlite`, `postgresql`, `mysql`, `databricks`, `oracle`, `mssql`, `azure_sql`, `cloud_sql`, `snowflake`, `supabase`, `duckdb`, `timescaledb`, `bigquery`, `motherduck` |
| `document_store` | 3 | `cassandra`, `mongodb`, `dynamodb` |
| `key_value_cache` | 4 | `redis`, `memcached`, `elasticache`, `upstash_redis` |
| `message_bus` | 10 | `kafka`, `celery`, `rabbitmq`, `sqs`, `service_bus`, `pubsub`, `temporal`, `nats`, `redpanda`, `upstash_qstash` |
| `object_storage` | 8 | `s3`, `azure_blob`, `gcs`, `minio`, `filesystem`, `cloudflare_r2`, `huggingface_hub`, `google_drive` |
| `vector_store` | 9 | `pinecone`, `qdrant`, `chroma`, `weaviate`, `milvus`, `inmemory`, `vespa`, `pgvector`, `lancedb` |
| `search_provider` | 11 | `google_cse`, `bing`, `reddit`, `google_places`, `brave`, `serpapi`, `tavily`, `exa`, `perplexity`, `arxiv`, `semantic_scholar` |
| `notification_channel` | 12 | `slack`, `teams`, `webhook`, `log`, `email_smtp`, `discord`, `twilio`, `pagerduty`, `opsgenie`, `incident_io`, `sendgrid`, `telegram` |
| `interaction_surface` | 5 (+3 dual) | `lab_json`, `slash_command`, `mailgun`, `ollama`; `slack` / `teams` / `telegram` also register this category |
| `collaboration_suite` | 2 | `ms365_graph`, `google_workspace` |
| `issue_tracker` | 8 | `jira`, `github`, `linear`, `azure_devops`, `gitlab`, `servicenow`, `bitbucket`, `asana` |
| `wiki_knowledge` | 4 | `confluence`, `notion`, `sharepoint`, `wikipedia` |
| `observability_backend` | 22 | `prometheus`, `elasticsearch`, `otel`, `langfuse`, `datadog`, `clickhouse`, `sentry`, `langsmith`, `helicone`, `posthog`, `braintrust`, `signoz`, `honeycomb`, `arize`, `phoenix`, `wandb`, `opensearch`, `influxdb`, `grafana`, `loki`, `tempo`, `mlflow` |
| `document_parser` | 8 | `docling`, `pymupdf`, `unstructured`, `python_docx`, `openpyxl`, `whisper`, `yt_dlp`, `llamaparse` |
| `rerank_provider` | 2 | `cohere_rerank`, `jina_rerank` |
| `browser_automation` | 5 | `playwright`, `firecrawl`, `selenium`, `browserbase`, `apify` |
| `secrets_store` | 5 | `vault`, `aws_secrets_manager`, `azure_key_vault`, `gcp_secret_manager`, `doppler` |
| `graph_store` | 3 | `neo4j`, `memgraph`, `falkordb` |
| `cloud_platform` | 4 | `aws`, `azure`, `gcp`, `kubernetes` |
| `feature_flag` | 2 | `unleash`, `launchdarkly` |
| `ci_cd` | 1 | `github_actions` |

**Total unique slugs:** 185.

### Implementation depth (code audit)

| Depth | Count | Meaning |
|-------|------:|---------|
| **full** | 30+ | Dedicated adapter + opens (+ config) |
| **full-partial** | 6 | opens/client without full adapter split |
| **thin-p2** | 22 | `_shared/p2/factories.py` |
| **thin-p3** | 21 | `_shared/p3/factories.py` (M.7 harness + sentry) |
| **thin-p4** | 14 | `_shared/p4/factories.py` (M.8 harness gap) |

**Thin-p4 slugs:** `langsmith`, `helicone`, `posthog`, `braintrust`, `signoz`, `honeycomb`, `arize`, `phoenix`, `wandb`, `opensearch`, `pagerduty`, `opsgenie`, `gitlab`, `vespa`.

**Vector-store bridges** (`pinecone`, `qdrant`, `chroma`): RAG implementation in `intergrax/rag/vectorstore/`.

Unit tests: `tests/unit/integrations/` — **335+** cases including `test_p2_providers.py` and `test_p3_providers.py`.

### Reserved slugs (manifest constants) but not registered in catalog

None — all enum slugs in `FIELD_SLUGS` are registered except cloud-platform-only defaults resolved via `IntegrationProfile.with_cloud_platform()`.

---

### Relational store

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `sqlite` | stable | `create_sqlite_relational_store()` | `INTERGRAX_SQLITE` | Lab default; also trace, checkpoint, HITL stores via bundle helpers |
| `postgresql` | beta | `create_postgresql_relational_store()` | `INTERGRAX_POSTGRESQL` | psycopg3; DSN or host/port/user/password |
| `mysql` | beta | `create_mysql_relational_store()` | `INTERGRAX_MYSQL` | pymysql |
| `databricks` | beta | `create_databricks_relational_store()` | `INTERGRAX_DATABRICKS` | SQL Warehouse / Unity Catalog |
| `oracle` | beta | `create_oracle_relational_store()` | `INTERGRAX_ORACLE` | oracledb DSN |
| `mssql` | beta | `create_mssql_relational_store()` | `INTERGRAX_MSSQL` | pyodbc |
| `azure_sql` | beta | `create_azure_sql_relational_store()` | `INTERGRAX_AZURE_SQL` | Azure SQL via pyodbc |
| `cloud_sql` | beta | `create_cloud_sql_relational_store()` | `INTERGRAX_CLOUD_SQL` | Cloud SQL via pg8000 |

| Provider | Usage guide |
|----------|-------------|
| `sqlite` | [USAGE.md](../intergrax/integrations/providers/relational_store/sqlite/USAGE.md) |
| `postgresql` | [USAGE.md](../intergrax/integrations/providers/relational_store/postgresql/USAGE.md) |
| `mysql` | [USAGE.md](../intergrax/integrations/providers/relational_store/mysql/USAGE.md) |
| `databricks` | [USAGE.md](../intergrax/integrations/providers/relational_store/databricks/USAGE.md) |
| `oracle` | [USAGE.md](../intergrax/integrations/providers/relational_store/oracle/USAGE.md) |
| `mssql` | [USAGE.md](../intergrax/integrations/providers/relational_store/mssql/USAGE.md) |
| `azure_sql` | [USAGE.md](../intergrax/integrations/providers/relational_store/azure_sql/USAGE.md) |
| `cloud_sql` | [USAGE.md](../intergrax/integrations/providers/relational_store/cloud_sql/USAGE.md) |

---

### Document store

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `cassandra` | beta | `create_cassandra_document_store()` | `INTERGRAX_CASSANDRA` | CQL partition-scoped CRUD |
| `mongodb` | beta | `create_mongodb_document_store()` | `INTERGRAX_MONGODB` | Flexible JSON documents via PyMongo |
| `dynamodb` | beta | `create_dynamodb_document_store()` | `INTERGRAX_DYNAMODB` | AWS DynamoDB partition-scoped CRUD |

| Provider | Usage guide |
|----------|-------------|
| `cassandra` | [USAGE.md](../intergrax/integrations/providers/document_store/cassandra/USAGE.md) |
| `mongodb` | [USAGE.md](../intergrax/integrations/providers/document_store/mongodb/USAGE.md) |
| `dynamodb` | [USAGE.md](../intergrax/integrations/providers/document_store/dynamodb/USAGE.md) |

---

### Key-value cache

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `redis` | stable | `create_redis_key_value_cache()` | `INTERGRAX_REDIS` | Also idempotency, rate limit, semaphore via `create_redis_integration()` |
| `memcached` | beta | `create_memcached_key_value_cache()` | `INTERGRAX_MEMCACHED` | pymemcache |
| `elasticache` | beta | `create_elasticache_key_value_cache()` | `INTERGRAX_ELASTICACHE` | Redis-compatible endpoint (same adapter as memcached duck client) |

| Provider | Usage guide |
|----------|-------------|
| `redis` | [USAGE.md](../intergrax/integrations/providers/key_value_cache/redis/USAGE.md) |
| `memcached` | [USAGE.md](../intergrax/integrations/providers/key_value_cache/memcached/USAGE.md) |
| `elasticache` | [USAGE.md](../intergrax/integrations/providers/key_value_cache/elasticache/USAGE.md) |

---

### Message bus

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `kafka` | stable | `create_kafka_message_bus()` | `INTERGRAX_KAFKA` | Runtime transport delegates here |
| `celery` | stable | `create_celery_message_bus()` | `INTERGRAX_CELERY` | Broker/backend via env or injected app |
| `rabbitmq` | stable | `create_rabbitmq_message_bus()` | `INTERGRAX_RABBITMQ` | Requires KV store for ack semantics |
| `sqs` | beta | `create_sqs_message_bus()` | `INTERGRAX_SQS` | AWS SQS (also via `aws` facade) |
| `service_bus` | beta | `create_service_bus_message_bus()` | `INTERGRAX_SERVICE_BUS` | Azure Service Bus |
| `pubsub` | beta | `create_pubsub_message_bus()` | `INTERGRAX_PUBSUB` | GCP Pub/Sub |

| Provider | Usage guide |
|----------|-------------|
| `kafka` | [USAGE.md](../intergrax/integrations/providers/message_bus/kafka/USAGE.md) |
| `celery` | [USAGE.md](../intergrax/integrations/providers/message_bus/celery/USAGE.md) |
| `rabbitmq` | [USAGE.md](../intergrax/integrations/providers/message_bus/rabbitmq/USAGE.md) |
| `sqs` | [USAGE.md](../intergrax/integrations/providers/message_bus/sqs/USAGE.md) |
| `service_bus` | [USAGE.md](../intergrax/integrations/providers/message_bus/service_bus/USAGE.md) |
| `pubsub` | [USAGE.md](../intergrax/integrations/providers/message_bus/pubsub/USAGE.md) |

---

### Object storage

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `s3` | beta | `create_s3_object_storage()` | `INTERGRAX_S3` | put / get / delete / presigned_url; boto3 in `opens.py` only |
| `azure_blob` | beta | `create_azure_blob_object_storage()` | `INTERGRAX_AZURE_BLOB` | Azure Blob; SDK only in `opens.py` |
| `gcs` | beta | `create_gcs_object_storage()` | `INTERGRAX_GCS` | Google Cloud Storage; SDK only in factory open path |

| Provider | Usage guide |
|----------|-------------|
| `s3` | [USAGE.md](../intergrax/integrations/providers/object_storage/s3/USAGE.md) |
| `azure_blob` | [USAGE.md](../intergrax/integrations/providers/object_storage/azure_blob/USAGE.md) |
| `gcs` | [USAGE.md](../intergrax/integrations/providers/object_storage/gcs/USAGE.md) |

---

## RAG engine (cross-reference)

**Canonical domain pair:** [`architecture/RAG.md`](RAG.md) ↔ [`plan/RAG.md`](../plan/RAG.md) — retrieval orchestration, ingest, eval, M-RAG register.

This catalog doc covers **integration slugs** consumed by RAG (`vector_store`, `document_parser`, `rerank_provider`, `graph_store`). Do not duplicate engine canon here.

---

### Vector store (RAG)

Vector implementations remain in `intergrax/rag/vectorstore/`. Integration providers are **thin catalog bridges** — select backend via `IntegrationProfile.vector_store` or RAG bootstrap.

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `pinecone` | beta | `create_pinecone_vector_store()` | `INTERGRAX_PINECONE` | Managed cloud index |
| `qdrant` | beta | `create_qdrant_vector_store()` | `INTERGRAX_QDRANT` | Self-hosted or Qdrant Cloud |
| `chroma` | beta | `create_chroma_vector_store()` | `INTERGRAX_CHROMA` | Embedded or HTTP Chroma |

| Provider | Usage guide |
|----------|-------------|
| `pinecone` | [USAGE.md](../intergrax/integrations/providers/vector_store/pinecone/USAGE.md) |
| `qdrant` | [USAGE.md](../intergrax/integrations/providers/vector_store/qdrant/USAGE.md) |
| `chroma` | [USAGE.md](../intergrax/integrations/providers/vector_store/chroma/USAGE.md) |

RAG bootstrap: `create_default_vectorstore_manager()` in `intergrax/rag/vectorstore/bootstrap/` resolves via the integration catalog when `vector_store` is configured.

---

### Search provider

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `google_cse` | stable | `create_google_cse_search_provider()` | `INTERGRAX_GOOGLE_CSE` | Google Custom Search |
| `bing` | stable | `create_bing_search_provider()` | `INTERGRAX_BING` | Bing Web Search v7 |
| `brave` | beta | `create_brave_search_provider()` | `INTERGRAX_BRAVE` | Brave Search API |
| `serpapi` | beta | `create_serpapi_search_provider()` | `INTERGRAX_SERPAPI` | SerpAPI organic results |

| Provider | Usage guide |
|----------|-------------|
| `google_cse` | [USAGE.md](../intergrax/integrations/providers/search_provider/google_cse/USAGE.md) |
| `bing` | [USAGE.md](../intergrax/integrations/providers/search_provider/bing/USAGE.md) |
| `brave` | [USAGE.md](../intergrax/integrations/providers/search_provider/brave/USAGE.md) |
| `serpapi` | [USAGE.md](../intergrax/integrations/providers/search_provider/serpapi/USAGE.md) |

---

### Notification channel

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `slack` | stable | `create_slack_notification_channel()` | `INTERGRAX_SLACK` | Also registered as `interaction_surface` |
| `teams` | stable | `create_teams_notification_channel()` | `INTERGRAX_TEAMS` | Also registered as `interaction_surface` |
| `webhook` | stable | `create_webhook_notification_channel()` | `INTERGRAX_WEBHOOK` | Generic HTTP outbound |
| `log` | stable | `create_log_notification_channel()` | `INTERGRAX_LOG` | Process log; lab profile default |
| `email_smtp` | beta | `create_email_smtp_notification_channel()` | `INTERGRAX_EMAIL_SMTP` | Outbound SMTP mail |

| Provider | Usage guide |
|----------|-------------|
| `slack` | [USAGE.md](../intergrax/integrations/providers/notification_channel/slack/USAGE.md) |
| `teams` | [USAGE.md](../intergrax/integrations/providers/notification_channel/teams/USAGE.md) |
| `webhook` | [USAGE.md](../intergrax/integrations/providers/notification_channel/webhook/USAGE.md) |
| `log` | [USAGE.md](../intergrax/integrations/providers/notification_channel/log/USAGE.md) |
| `email_smtp` | [USAGE.md](../intergrax/integrations/providers/notification_channel/email_smtp/USAGE.md) |

---

### Interaction surface

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `slack` | stable | `create_slack_interaction_surface()` | `INTERGRAX_SLACK` | Inbound slash commands / events |
| `teams` | stable | `create_teams_interaction_surface()` | `INTERGRAX_TEAMS` | Microsoft Teams activity intake |
| `lab_json` | stable | `create_lab_json_interaction_surface()` | `INTERGRAX_LAB_JSON` | Laboratory JSON webhook intake |
| `slash_command` | beta | `create_slash_command_interaction_surface()` | `INTERGRAX_SLASH_COMMAND` | Generic slash-command payloads (Slack/Teams/CLI) |

| Provider | Usage guide |
|----------|-------------|
| `slack` | [USAGE.md](../intergrax/integrations/providers/notification_channel/slack/USAGE.md) |
| `teams` | [USAGE.md](../intergrax/integrations/providers/notification_channel/teams/USAGE.md) |
| `lab_json` | [USAGE.md](../intergrax/integrations/providers/interaction_surface/lab_json/USAGE.md) |
| `slash_command` | [USAGE.md](../intergrax/integrations/providers/interaction_surface/slash_command/USAGE.md) |

---

### Collaboration suite

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `ms365_graph` | beta | `create_ms365_graph_collaboration_suite()` | `INTERGRAX_MS365_GRAPH` | Mail, calendar, directory via Microsoft Graph |
| `google_workspace` | beta | `create_google_workspace_collaboration_suite()` | `INTERGRAX_GOOGLE_WORKSPACE` | Gmail / Calendar / Directory via Google APIs |

| Provider | Usage guide |
|----------|-------------|
| `ms365_graph` | [USAGE.md](../intergrax/integrations/providers/collaboration_suite/ms365_graph/USAGE.md) |
| `google_workspace` | [USAGE.md](../intergrax/integrations/providers/collaboration_suite/google_workspace/USAGE.md) |

---

### Issue tracker

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `jira` | beta | `create_jira_issue_tracker()` | `INTERGRAX_JIRA` | REST v3 — issues, comments, search |
| `github` | beta | `create_github_issue_tracker()` | `INTERGRAX_GITHUB` | GitHub Issues REST |
| `linear` | beta | `create_linear_issue_tracker()` | `INTERGRAX_LINEAR` | Linear issues API |
| `azure_devops` | beta | `create_azure_devops_issue_tracker()` | `INTERGRAX_AZURE_DEVOPS` | Azure DevOps work items |

| Provider | Usage guide |
|----------|-------------|
| `jira` | [USAGE.md](../intergrax/integrations/providers/issue_tracker/jira/USAGE.md) |
| `github` | [USAGE.md](../intergrax/integrations/providers/issue_tracker/github/USAGE.md) |
| `linear` | [USAGE.md](../intergrax/integrations/providers/issue_tracker/linear/USAGE.md) |
| `azure_devops` | [USAGE.md](../intergrax/integrations/providers/issue_tracker/azure_devops/USAGE.md) |

---

### Wiki / knowledge

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `confluence` | beta | `create_confluence_wiki_knowledge()` | `INTERGRAX_CONFLUENCE` | Pages and search for RAG / runbooks |
| `notion` | beta | `create_notion_wiki_knowledge()` | `INTERGRAX_NOTION` | Notion pages API |
| `sharepoint` | beta | `create_sharepoint_wiki_knowledge()` | `INTERGRAX_SHAREPOINT` | SharePoint pages / search |

| Provider | Usage guide |
|----------|-------------|
| `confluence` | [USAGE.md](../intergrax/integrations/providers/wiki_knowledge/confluence/USAGE.md) |
| `notion` | [USAGE.md](../intergrax/integrations/providers/wiki_knowledge/notion/USAGE.md) |
| `sharepoint` | [USAGE.md](../intergrax/integrations/providers/wiki_knowledge/sharepoint/USAGE.md) |

---

### Observability backend

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `prometheus` | beta | `create_prometheus_observability_backend()` | `INTERGRAX_PROMETHEUS` | PromQL instant / range queries |
| `elasticsearch` | beta | `create_elasticsearch_observability_backend()` | `INTERGRAX_ELASTICSEARCH` | Log search and aggregations |
| `otel` | beta | `create_otel_observability_backend()` | `INTERGRAX_OTEL` | OTLP-oriented metrics facade |
| `langfuse` | beta | `create_langfuse_observability_backend()` | `INTERGRAX_LANGFUSE` | LLM/agent trace export |
| `datadog` | beta | `create_datadog_observability_backend()` | `INTERGRAX_DATADOG` | APM metrics API |
| `clickhouse` | beta | `create_clickhouse_observability_backend()` | `INTERGRAX_CLICKHOUSE` | High-volume event analytics |
| `sentry` | beta | `create_sentry_observability_backend()` | `INTERGRAX_SENTRY` | Error tracking + issue stats; `capture_exception` / `capture_message` |

| Provider | Usage guide |
|----------|-------------|
| `prometheus` | [USAGE.md](../intergrax/integrations/providers/observability_backend/prometheus/USAGE.md) |
| `elasticsearch` | [USAGE.md](../intergrax/integrations/providers/observability_backend/elasticsearch/USAGE.md) |
| `otel` | [USAGE.md](../intergrax/integrations/providers/observability_backend/otel/USAGE.md) |
| `langfuse` | [USAGE.md](../intergrax/integrations/providers/observability_backend/langfuse/USAGE.md) |
| `datadog` | [USAGE.md](../intergrax/integrations/providers/observability_backend/datadog/USAGE.md) |
| `clickhouse` | [USAGE.md](../intergrax/integrations/providers/observability_backend/clickhouse/USAGE.md) |
| `sentry` | [USAGE.md](../intergrax/integrations/providers/observability_backend/sentry/USAGE.md) |

---

### Browser automation

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `playwright` | beta | `create_playwright_browser_automation()` | `INTERGRAX_PLAYWRIGHT` | Headless Chromium via Playwright |

---

### Cloud platform

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `aws` | beta | `create_aws_cloud_platform()` | `INTERGRAX_AWS` | IAM / STS; defaults for S3, SQS, DynamoDB, ElastiCache |
| `azure` | beta | `create_azure_cloud_platform()` | `INTERGRAX_AZURE` | MI / service principal; defaults for Blob, Service Bus |
| `gcp` | beta | `create_gcp_cloud_platform()` | `INTERGRAX_GCP` | ADC / service account; defaults for GCS, Pub/Sub |

| Provider | Usage guide |
|----------|-------------|
| `aws` | [USAGE.md](../intergrax/integrations/providers/cloud_platform/aws/USAGE.md) |
| `azure` | [USAGE.md](../intergrax/integrations/providers/cloud_platform/azure/USAGE.md) |
| `gcp` | [USAGE.md](../intergrax/integrations/providers/cloud_platform/gcp/USAGE.md) |

---

## Full provider index

Alphabetical reference — all shipped integrations in one table.

| Slug | Category (categories) | Status | Catalog factory | Usage |
|------|----------------------|--------|-----------------|-------|
| `aws` | `cloud_platform` | beta | `create_aws_cloud_platform()` | [USAGE](../intergrax/integrations/providers/cloud_platform/aws/USAGE.md) |
| `azure` | `cloud_platform` | beta | `create_azure_cloud_platform()` | [USAGE](../intergrax/integrations/providers/cloud_platform/azure/USAGE.md) |
| `azure_blob` | `object_storage` | beta | `create_azure_blob_object_storage()` | [USAGE](../intergrax/integrations/providers/object_storage/azure_blob/USAGE.md) |
| `azure_devops` | `issue_tracker` | beta | `create_azure_devops_issue_tracker()` | [USAGE](../intergrax/integrations/providers/issue_tracker/azure_devops/USAGE.md) |
| `azure_sql` | `relational_store` | beta | `create_azure_sql_relational_store()` | [USAGE](../intergrax/integrations/providers/relational_store/azure_sql/USAGE.md) |
| `bing` | `search_provider` | stable | `create_bing_search_provider()` | [USAGE](../intergrax/integrations/providers/search_provider/bing/USAGE.md) |
| `brave` | `search_provider` | beta | `create_brave_search_provider()` | [USAGE](../intergrax/integrations/providers/search_provider/brave/USAGE.md) |
| `cassandra` | `document_store` | beta | `create_cassandra_document_store()` | [USAGE](../intergrax/integrations/providers/document_store/cassandra/USAGE.md) |
| `celery` | `message_bus` | stable | `create_celery_message_bus()` | [USAGE](../intergrax/integrations/providers/message_bus/celery/USAGE.md) |
| `chroma` | `vector_store` | beta | `create_chroma_vector_store()` | [USAGE](../intergrax/integrations/providers/vector_store/chroma/USAGE.md) |
| `cloud_sql` | `relational_store` | beta | `create_cloud_sql_relational_store()` | [USAGE](../intergrax/integrations/providers/relational_store/cloud_sql/USAGE.md) |
| `confluence` | `wiki_knowledge` | beta | `create_confluence_wiki_knowledge()` | [USAGE](../intergrax/integrations/providers/wiki_knowledge/confluence/USAGE.md) |
| `databricks` | `relational_store` | beta | `create_databricks_relational_store()` | [USAGE](../intergrax/integrations/providers/relational_store/databricks/USAGE.md) |
| `dynamodb` | `document_store` | beta | `create_dynamodb_document_store()` | [USAGE](../intergrax/integrations/providers/document_store/dynamodb/USAGE.md) |
| `elasticache` | `key_value_cache` | beta | `create_elasticache_key_value_cache()` | [USAGE](../intergrax/integrations/providers/key_value_cache/elasticache/USAGE.md) |
| `elasticsearch` | `observability_backend` | beta | `create_elasticsearch_observability_backend()` | [USAGE](../intergrax/integrations/providers/observability_backend/elasticsearch/USAGE.md) |
| `email_smtp` | `notification_channel` | beta | `create_email_smtp_notification_channel()` | [USAGE](../intergrax/integrations/providers/notification_channel/email_smtp/USAGE.md) |
| `gcp` | `cloud_platform` | beta | `create_gcp_cloud_platform()` | [USAGE](../intergrax/integrations/providers/cloud_platform/gcp/USAGE.md) |
| `gcs` | `object_storage` | beta | `create_gcs_object_storage()` | [USAGE](../intergrax/integrations/providers/object_storage/gcs/USAGE.md) |
| `github` | `issue_tracker` | beta | `create_github_issue_tracker()` | [USAGE](../intergrax/integrations/providers/issue_tracker/github/USAGE.md) |
| `google_cse` | `search_provider` | stable | `create_google_cse_search_provider()` | [USAGE](../intergrax/integrations/providers/search_provider/google_cse/USAGE.md) |
| `google_workspace` | `collaboration_suite` | beta | `create_google_workspace_collaboration_suite()` | [USAGE](../intergrax/integrations/providers/collaboration_suite/google_workspace/USAGE.md) |
| `jira` | `issue_tracker` | beta | `create_jira_issue_tracker()` | [USAGE](../intergrax/integrations/providers/issue_tracker/jira/USAGE.md) |
| `kafka` | `message_bus` | stable | `create_kafka_message_bus()` | [USAGE](../intergrax/integrations/providers/message_bus/kafka/USAGE.md) |
| `lab_json` | `interaction_surface` | stable | `create_lab_json_interaction_surface()` | [USAGE](../intergrax/integrations/providers/interaction_surface/lab_json/USAGE.md) |
| `linear` | `issue_tracker` | beta | `create_linear_issue_tracker()` | [USAGE](../intergrax/integrations/providers/issue_tracker/linear/USAGE.md) |
| `log` | `notification_channel` | stable | `create_log_notification_channel()` | [USAGE](../intergrax/integrations/providers/notification_channel/log/USAGE.md) |
| `memcached` | `key_value_cache` | beta | `create_memcached_key_value_cache()` | [USAGE](../intergrax/integrations/providers/key_value_cache/memcached/USAGE.md) |
| `mongodb` | `document_store` | beta | `create_mongodb_document_store()` | [USAGE](../intergrax/integrations/providers/document_store/mongodb/USAGE.md) |
| `ms365_graph` | `collaboration_suite` | beta | `create_ms365_graph_collaboration_suite()` | [USAGE](../intergrax/integrations/providers/collaboration_suite/ms365_graph/USAGE.md) |
| `mssql` | `relational_store` | beta | `create_mssql_relational_store()` | [USAGE](../intergrax/integrations/providers/relational_store/mssql/USAGE.md) |
| `mysql` | `relational_store` | beta | `create_mysql_relational_store()` | [USAGE](../intergrax/integrations/providers/relational_store/mysql/USAGE.md) |
| `notion` | `wiki_knowledge` | beta | `create_notion_wiki_knowledge()` | [USAGE](../intergrax/integrations/providers/wiki_knowledge/notion/USAGE.md) |
| `oracle` | `relational_store` | beta | `create_oracle_relational_store()` | [USAGE](../intergrax/integrations/providers/relational_store/oracle/USAGE.md) |
| `otel` | `observability_backend` | beta | `create_otel_observability_backend()` | [USAGE](../intergrax/integrations/providers/observability_backend/otel/USAGE.md) |
| `pinecone` | `vector_store` | beta | `create_pinecone_vector_store()` | [USAGE](../intergrax/integrations/providers/vector_store/pinecone/USAGE.md) |
| `playwright` | `browser_automation` | beta | `create_playwright_browser_automation()` | [USAGE](../intergrax/integrations/providers/browser_automation/playwright/USAGE.md) |
| `postgresql` | `relational_store` | beta | `create_postgresql_relational_store()` | [USAGE](../intergrax/integrations/providers/relational_store/postgresql/USAGE.md) |
| `prometheus` | `observability_backend` | beta | `create_prometheus_observability_backend()` | [USAGE](../intergrax/integrations/providers/observability_backend/prometheus/USAGE.md) |
| `pubsub` | `message_bus` | beta | `create_pubsub_message_bus()` | [USAGE](../intergrax/integrations/providers/message_bus/pubsub/USAGE.md) |
| `qdrant` | `vector_store` | beta | `create_qdrant_vector_store()` | [USAGE](../intergrax/integrations/providers/vector_store/qdrant/USAGE.md) |
| `rabbitmq` | `message_bus` | stable | `create_rabbitmq_message_bus()` | [USAGE](../intergrax/integrations/providers/message_bus/rabbitmq/USAGE.md) |
| `redis` | `key_value_cache` | stable | `create_redis_key_value_cache()` | [USAGE](../intergrax/integrations/providers/key_value_cache/redis/USAGE.md) |
| `s3` | `object_storage` | beta | `create_s3_object_storage()` | [USAGE](../intergrax/integrations/providers/object_storage/s3/USAGE.md) |
| `serpapi` | `search_provider` | beta | `create_serpapi_search_provider()` | [USAGE](../intergrax/integrations/providers/search_provider/serpapi/USAGE.md) |
| `sentry` | `observability_backend` | beta | `create_sentry_observability_backend()` | [USAGE](../intergrax/integrations/providers/observability_backend/sentry/USAGE.md) |
| `service_bus` | `message_bus` | beta | `create_service_bus_message_bus()` | [USAGE](../intergrax/integrations/providers/message_bus/service_bus/USAGE.md) |
| `sharepoint` | `wiki_knowledge` | beta | `create_sharepoint_wiki_knowledge()` | [USAGE](../intergrax/integrations/providers/wiki_knowledge/sharepoint/USAGE.md) |
| `slack` | `notification_channel`, `interaction_surface` | stable | `create_slack_catalog_factory()` | [USAGE](../intergrax/integrations/providers/notification_channel/slack/USAGE.md) |
| `sqs` | `message_bus` | beta | `create_sqs_message_bus()` | [USAGE](../intergrax/integrations/providers/message_bus/sqs/USAGE.md) |
| `sqlite` | `relational_store` | stable | `create_sqlite_relational_store()` | [USAGE](../intergrax/integrations/providers/relational_store/sqlite/USAGE.md) |
| `teams` | `notification_channel`, `interaction_surface` | stable | `create_teams_catalog_factory()` | [USAGE](../intergrax/integrations/providers/notification_channel/teams/USAGE.md) |
| `webhook` | `notification_channel` | stable | `create_webhook_notification_channel()` | [USAGE](../intergrax/integrations/providers/notification_channel/webhook/USAGE.md) |

---

## Phase M.7 harness backlog — Done (beta)

All slugs from the 2026-05 harness recommendation (high / medium / low) are registered. Shared implementation: `intergrax/integrations/_shared/p3/factories.py`. New categories: `secrets_store`, `graph_store`.

## Phase M.9 harness depth — Done (beta)

**Full adapters** (replacing thin-p4 shells): `langsmith`, `opensearch`, `vespa`, `gitlab`, `pagerduty`, `braintrust` — dedicated `config/client/adapter/opens/bundle` packages.

**New tools:** `gitlab.create_issue`, `pagerduty.trigger_incident`, `braintrust.log_eval`.

**`slash_command`** interaction surface registered (catalog **167**).

**Lab harness:** `IntegrationProfile.harness_lab()` + `wire_lab_integrations(harness=True)` — composite observability (Sentry + LangSmith) + PagerDuty + harness tool bundle.

## Phase M.10 harness Tier A — Done (beta)

**Composite observability:** `ToolWiringContext.observability_backends` populated from profile primary + `options` observability slugs. Role-based resolution in `resolve_observability_backend()` — `errors.capture` → Sentry, `observability.query_traces` → LangSmith, `braintrust.log_eval` → Braintrust.

**HITL → PagerDuty (runtime):** `wire_lab_integrations(harness=True)` resolves notification adapter directly from profile (`create_harness_notification_adapter`). Long-running tasks with `notify_channel="pagerduty"` escalate via `LongRunningCoordinator.notify_escalation()`. Factory: `LAB_HARNESS=true`.

**Tests:** `tests/unit/tools/providers/observability/test_composite_observability.py`, `tests/integration/runtime/test_harness_hitl_pagerduty.py`.

## Phase M.11 harness — Done (beta)

**Default notify channel:** `make_lab_harness_task_enricher()` + `apply_default_long_running_notify_channel()` inject profile default (`pagerduty` in harness) when `long_running.enabled` and `notify_channel` unset. Wired in `create_lab_application()` for `POST /v1/lab/run` (`long_running: true`) and interaction intake.

**Next gaps (M.13+):** full adapters for remaining thin-p4 slugs (Helicone, PostHog, …), network smoke CI for non-guardrail harness integrations. **Guardrails (M.12):** Done — see §47.

**CI:** `harness-smoke` job in `.github/workflows/unit-tests.yml` (`integrations-harness` extra).

**Legacy → catalog:** notification factory resolves PagerDuty/Opsgenie via catalog; `distributed/bootstrap.resolve_redis_kv_store()` via `IntegrationProfile`.

---

## Phase M.8 harness gap — Done (beta)

**+14 slugs** via `intergrax/integrations/_shared/p4/factories.py`: `langsmith`, `helicone`, `posthog`, `braintrust`, `signoz`, `honeycomb`, `arize`, `phoenix`, `wandb`, `opensearch`, `pagerduty`, `opsgenie`, `gitlab`, `vespa`. Wire script: `scripts/wire_p4_harness_providers.py`.

---

## Harness AI gap analysis (closed)

Audit against typical agent stacks (LangGraph, CrewAI, LlamaIndex, enterprise VPC). **Done** means the slug is registered in the catalog; **Missing** means consider in a future iteration.

| Priority | Slug | Category | Status | Why harness |
|----------|------|----------|--------|-------------|
| **High** | `langsmith` | observability_backend | **Done** | LLM trace/debug (LangChain ecosystem) |
| **High** | `helicone` | observability_backend | **Done** | LLM cost/latency proxy |
| **High** | `pagerduty` / `opsgenie` | notification_channel | **Done** | On-call escalation after HITL / agent failure |
| **Medium** | `posthog` | observability_backend | **Done** | Product analytics + feature flags |
| **Medium** | `braintrust` | observability_backend | **Done** | Evals + prompt regression |
| **Medium** | `gitlab` | issue_tracker | **Done** | DevOps issue flow |
| **Medium** | `signoz` / `honeycomb` | observability_backend | **Done** | OTEL-native APM |
| **Medium** | `arize` / `phoenix` | observability_backend | **Done** | ML/RAG eval + drift |
| **Future** | `opensearch`, `vespa`, `wandb` | various | **Done** (M.8) | Search/RAG scale, experiment tracking |
| **Low** | `reddit`, `google_places` | search_provider | **Done** | Social/geo search (full packages) |
| **Future** | `slash_command` | interaction_surface | **Done** (M.9) | Generic slash intake |

**Strong harness coverage today:** **185** integrations — observability (24+), notification (11+), issue trackers (9), vectors (9 incl. typesense), secrets (6 incl. infisical), feature flags (3), CI/CD (8 incl. argocd), security scanners (3), sandbox hosts (3), identity (3), speech (2), workflow (2), CRM (2), plus M.7 stack (Vault, Neo4j, Temporal, …).

**Tool Library:** `errors.capture`, `gitlab.create_issue`, `pagerduty.trigger_incident`, `braintrust.log_eval`. Optional deps: ``uv pip install 'Intergrax-ai[integrations-harness]'``.

---

All **185** shipped providers include an English usage guide at `intergrax/integrations/providers/<category>/<slug>/USAGE.md`. Regenerate after catalog changes:

```bash
uv run python scripts/generate_integration_usage_docs.py
```

---

## 47. LLM guardrail integrations

**Canon cross-refs:** UAEP guardrail catalog §42.11.6 · Policy bundle §42.11.4 · Middleware hooks §42.42 · Security defenses §42.45 · CVL [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md).

Third-party **guardrail engines** belong in the **Integration Library** — not in Tier-2 agents and not as parallel Nexus modules. Intergrax composes **native harness defenses** (`prompt_security`, `tool_security`, `PolicyEngine`, CVL L0) with **optional vendor backends** selected per Tier-3 host.

### 47.1 Design rules

| Rule | Rationale |
|------|-----------|
| **Category `llm_guardrail`** | One contract (`LlmGuardrailBackend`) — swap NeMo / Guardrails AI / LLM Guard / OpenGuardrails without agent changes |
| **Vendor imports only in `opens.py`** | Same boundary as §Design principles — `scripts/check_integration_vendor_imports.py` |
| **Middleware invocation only** | Tier-1 `guardrail_runtime_bridge` (planned M.12) calls integration at §42.42 hooks — agents never import slugs |
| **Policy consequences unified** | Scanner result → `GuardrailScanResult` → `PolicyEngine` / `ValidationResult` — no ad-hoc `raise` in adapters |
| **Defense in depth** | Fast deterministic scanners first (LLM Guard, native `prompt_security`); orchestration frameworks second (NeMo); semantic validators third (Guardrails AI Hub) |
| **Not LLM adapters** | Guardrail engines are **not** `intergrax/llm_adapters/` — they scan or constrain calls, not replace the producer model |

### 47.2 Category contract (M-P12-CAT.1)

```text
LlmGuardrailBackend:
    slug: str
    scan_input(text, *, context: GuardrailContext) -> GuardrailScanResult
    scan_output(text, *, context: GuardrailContext) -> GuardrailScanResult
    scan_tool_call(request: ToolRequest, *, context: GuardrailContext) -> GuardrailScanResult  # optional
    health_check() -> bool

GuardrailScanResult:
    allowed: bool
    risk_level: low | medium | high | critical
    categories: list[str]          # e.g. prompt_injection, pii, toxicity
    matched_rules: list[str]
    sanitized_text: str | null     # when redaction/masking applied
    audit_payload: dict
```

`IntegrationProfile.llm_guardrail: IntegrationBinding | None` — resolved at Tier-3 startup; wired through `security_runtime_bridge` + `guardrail_wiring` → `LlmGuardrailMiddleware` (M.12). Tier-3 `GuardrailProfile` toggles scan_input / scan_output / scan_tool_calls.

### 47.3 Recommended vendor libraries (2026)

Use this matrix to **pick engines by problem shape**, then register the matching catalog slug. Production stacks often **layer** multiple engines (fast scanner + programmable rails + output validator).

| Library | PyPI / package | Primary strength | Typical hook | Latency class | Deploy posture |
|---------|----------------|------------------|--------------|---------------|----------------|
| **[LLM Guard](https://github.com/protectai/llm-guard)** | `llm-guard` | Fast I/O scanning — prompt injection, secrets, toxicity, anonymization | pre-LLM, post-LLM | **Low** (~&lt;50ms CPU) | On-prem, no GPU required |
| **[Guardrails AI](https://github.com/guardrails-ai/guardrails)** | `guardrails-ai` | Composable validators, Hub validators, RAIL, structured-output re-ask | post-LLM, completion | Medium (20–200ms+) | On-prem; validators may call models |
| **[NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)** | `nemoguardrails` | Colang dialog flows, multi-rail pipelines, tool-call constraints | pre/post-LLM, multi-turn | Medium–high (150–500ms+) | GPU optional; strong for conversational agents |
| **[OpenGuardrails](https://github.com/openguardrails/openguardrails)** | `openguardrails` | Enterprise gateway — DLP, masking, multi-tenant policy, OpenAI-compatible proxy | gateway or SDK scan | Medium (API / self-hosted) | SaaS, private cloud, or on-prem gateway |
| **[Meta Llama Guard](https://huggingface.co/meta-llama/Llama-Guard-3-8B)** | via inference host / `llm_guardrail` wrapper | Content-safety classifier (input/output) | pre/post-LLM | Medium (model inference) | Self-hosted model; often composed inside NeMo or custom adapter |
| **[Microsoft Presidio](https://github.com/microsoft/presidio)** | `presidio-analyzer`, `presidio-anonymizer` | PII detect/redact (deterministic + NER) | pre-LLM, post-LLM, logs | Low–medium | On-prem; pairs with LLM Guard |
| **[Lakera Guard](https://www.lakera.ai/)** | REST API | Prompt injection / jailbreak API | pre-LLM | Low (~&lt;150ms API) | Managed API |
| **Azure AI Content Safety** | `azure-ai-contentsafety` | Moderation categories, blocklists | pre/post-LLM | Medium (cloud API) | Azure tenants |
| **AWS Bedrock Guardrails** | boto3 bedrock | Content, prompt attack, PII, contextual grounding | provider-side or pre-call | Variable | AWS Bedrock workloads |

**Also worth tracking:** `open-guardrail` (deterministic multi-language guard pack, edge-friendly), **Rebuff** (prompt injection focus), **Patronus** / **Aporia** (managed eval+safety APIs) — add catalog slugs when a product host requires them; do not fork ad-hoc SDK calls in agents.

### 47.4 Layering pattern (recommended)

```text
                    ┌─────────────────────────────────────┐
  User input ──────►│ L1: LLM Guard / prompt_security     │ fast deny / redact
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ L2: NeMo Guardrails (optional)      │ dialog / tool rails
                    └─────────────────┬───────────────────┘
                                      ▼
                              Producer LLM (llm_adapters)
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ L3: Guardrails AI / L0 CVL          │ schema + validators
                    └─────────────────┬───────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ PolicyEngine → HITL if REQUIRED     │
                    └─────────────────────────────────────┘
```

### 47.5 Shipped catalog slugs (Phase M.12 — Done)

| Slug | Backend library | Priority | Notes |
|------|-----------------|----------|-------|
| `llm_guard` | Protect AI LLM Guard | **P0** | Default on-prem fast scanner; `scan_prompt` / `scan_output` wrappers |
| `guardrails_ai` | Guardrails AI | **P0** | Hub validators; map to `GuardrailScanResult` |
| `nemo_guardrails` | NVIDIA NeMo Guardrails | **P1** | Colang config path via Tier-3 `GuardrailProfile` |
| `openguardrails` | OpenGuardrails SDK / gateway | **P1** | API key or self-hosted gateway URL |
| `presidio` | Microsoft Presidio | **P1** | PII-only adapter; composable with `llm_guard` |
| `llama_guard` | Llama Guard inference | **P2** | Requires `ml_inference_host` or Triton slug |
| `lakera` | Lakera Guard API | **P2** | Managed API adapter |
| `azure_content_safety` | Azure Content Safety | **P2** | Azure profile hosts |
| `bedrock_guardrails` | AWS Bedrock Guardrails | **P2** | `aws` cloud_platform hosts |

**Tier-3 preset (shipped):** `harness_guardrail_stack(primary="llm_guard", semantic="guardrails_ai")` in `registry/presets.py` — `semantic` becomes chained secondary scanner.

### 47.6 Wiring sketch (M.12 — Done)

```python
# Tier-3 host — applications/<product>/host/integration_wiring.py
from intergrax.integrations.registry.presets import harness_guardrail_stack  # M.12

profile = harness_guardrail_stack(
    primary="llm_guard",
    semantic="guardrails_ai",
    enable_presidio_pii=True,
)
# profile.llm_guardrail resolved → guardrail_runtime_bridge on RuntimeConfig
```

```python
# Tier-3 — guardrail_runtime_bridge + LlmGuardrailMiddleware (priority 52)
# Hooks: BEFORE_CONTEXT_BUILD, BEFORE_LLM_INFERENCE, BEFORE_TOOL_CALL,
#        AFTER_LLM_OUTPUT, AFTER_FINALIZATION
```

**Verification:**

```bash
uv run pytest tests/unit/integrations/providers/llm_guardrail/ -m gate -q
uv run pytest tests/unit/runtime/test_guardrail_runtime_bridge.py -m gate -q
uv run pytest tests/unit/applications/test_guardrail_output_hooks.py -m gate -q
python scripts/check_harness_guardrail_wiring.py
```

**Implementation tracker:** [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) Phase **M.12** · UAEP doc Phase **GR-DOC**.

---

## Adding a new provider

1. Implement `intergrax/integrations/providers/<category>/<slug>/` (config, adapter, opens, bundle, register).
2. Register in `register_default_integrations()`.
3. Add unit tests under `tests/unit/integrations/providers/`.
4. Add an entry to `scripts/generate_integration_usage_docs.py` and run the generator (English `USAGE.md`).
5. Update this catalog and the implementation plan tracker.

Delivery checklist: [plan/INTEGRATIONS.md) — Phase M.4 workflow.

---

## Tests

Integration catalog regression:

```bash
uv run pytest tests/unit/integrations/ -q
```

Vendor SDK boundary (CI + local gate — integrations, rag, agents):

```bash
uv run python scripts/check_integration_vendor_imports.py
uv run pytest tests/unit/integrations/test_vendor_import_governance.py -q
```

Allowed vendor import modules: `opens.py`, `rag_store.py`, `client.py`, `web_client.py`, `_shared/p3/factories.py`, `_shared/p3/clients.py` (integrations); `parser_trace_exporter.py` (rag). **No** vendor imports under `agents/`.

Conformance helpers: `intergrax/integrations/_shared/conformance.py`.

