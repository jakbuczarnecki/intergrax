# Intergrax Integration Library

**Last updated:** 2026-05-29

The **Integration Library** (`intergrax/integrations/`) is Intergrax’s modular catalog of external systems — databases, queues, search APIs, vector indexes, cloud platforms, and collaboration tools. Agents and applications wire backends **by category**, not by vendor SDK, so the same agent code can run in a local lab, a customer VPC, or a multi-cloud deployment.

**Related docs:**

| Document | Purpose |
|----------|---------|
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1 | Architecture canon — tiers, contracts, registry rules |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase M | Phase status, backlog, delivery workflow |
| [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) Appendix E | How agents vs applications use integrations |
| [TOOLS.md](TOOLS.md) | Agent-facing tools that compose these integrations |
| Per-provider guides | `intergrax/integrations/providers/<category>/<slug>/USAGE.md` |

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
├── vector_store/             # pinecone, qdrant, chroma, weaviate, milvus, inmemory
├── search_provider/          # google_cse, bing, brave, serpapi, tavily, exa
├── notification_channel/     # slack, teams, discord, twilio, …
├── interaction_surface/      # lab_json (slack/teams also register here)
├── collaboration_suite/      # ms365_graph, google_workspace
├── issue_tracker/            # jira, github, linear, azure_devops
├── wiki_knowledge/           # confluence, notion, sharepoint
├── observability_backend/    # prometheus, elasticsearch, otel, langfuse, datadog, clickhouse, sentry
├── browser_automation/       # playwright, firecrawl, selenium
├── secrets_store/            # vault
├── graph_store/              # neo4j
└── cloud_platform/           # aws, azure, gcp
```

**Import path:** `from intergrax.integrations.providers.object_storage.s3.bundle import create_s3_object_storage`

Catalog slugs (`IntegrationSlug.S3`) are unchanged — only the Python package path includes the category folder.

---

## Design principles

| Principle | What it means |
|-----------|---------------|
| **Universal contracts** | Each category (`relational_store`, `vector_store`, `message_bus`, …) defines a small Protocol. Providers implement the contract; agent logic depends on the contract only. |
| **Modular providers** | One slug = one package under `providers/<category>/<slug>/` (category = contract name). Swap Redis for ElastiCache, SQLite for PostgreSQL, or Chroma for Pinecone by changing `IntegrationProfile` — no agent refactor. |
| **Environment portability** | Tier-3 applications compose integrations at startup (`IntegrationProfile`, env vars). The same Tier-2 agent runs against lab defaults (`sqlite`, `log`, `lab_json`) or production stacks (`postgresql`, `slack`, `s3`, `qdrant`). |
| **Single entry for SDKs** | Vendor SDKs (boto3, PyMongo, httpx, …) are imported only in each provider’s `opens.py`. Tier-2 agents must **not** import provider slugs or vendor libraries. |
| **Catalog registration** | `register_default_integrations()` registers all shipped providers. Resolution: explicit slug → profile field → env (`INTERGRAX_INTEGRATION_<CATEGORY>`) → cloud-platform defaults. |

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

Agents consume integrations **through catalog tools** ([TOOLS.md](TOOLS.md)), not by importing provider adapters. Tier-3 may also pass resolved contracts into `ToolWiringContext` for tool handlers.

**Example — declarative profile:**

```python
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug
from intergrax.integrations.contracts.base import IntegrationCategory

profile = IntegrationProfile(
    relational_store=IntegrationSlug.POSTGRESQL,
    vector_store=IntegrationSlug.QDRANT,
    object_storage=IntegrationSlug.S3,
    notification_channel=IntegrationSlug.SLACK,
    options={
        IntegrationSlug.S3: {"bucket": "intergrax-artifacts", "prefix": "tenant-a"},
    },
)

store = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

**Example — lab defaults (no external vendors):**

```python
profile = IntegrationProfile.lab()
# relational_store → sqlite, notification_channel → log, interaction_surface → lab_json
```

---

## Category contracts

| Category | Contract | Typical use |
|----------|----------|-------------|
| `relational_store` | `RelationalStore` | SQL persistence, analytics warehouses |
| `document_store` | `DocumentStore` | Flexible JSON / wide-column documents |
| `key_value_cache` | `KeyValueCache` | Idempotency, rate limits, locks |
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

## Implemented providers (73)

All providers below are registered in `register_default_integrations()`.  
**Status:** `stable` = production-ready catalog entry; `beta` = shipped, API may evolve.

- **P2** slugs delegate to `intergrax/integrations/_shared/p2/factories.py`
- **P3 / M.7** harness slugs delegate to `intergrax/integrations/_shared/p3/factories.py` (2026-05-29)
- Thin shells live under `providers/<category>/<slug>/` — see `providers/layout.py`

### Summary by category

| Category | Count | Slugs |
|----------|------:|-------|
| `relational_store` | 10 | `sqlite`, `postgresql`, `mysql`, `databricks`, `oracle`, `mssql`, `azure_sql`, `cloud_sql`, `snowflake`, `supabase` |
| `document_store` | 3 | `cassandra`, `mongodb`, `dynamodb` |
| `key_value_cache` | 3 | `redis`, `memcached`, `elasticache` |
| `message_bus` | 8 | `kafka`, `celery`, `rabbitmq`, `sqs`, `service_bus`, `pubsub`, `temporal`, `nats` |
| `object_storage` | 5 | `s3`, `azure_blob`, `gcs`, `minio`, `filesystem` |
| `vector_store` | 6 | `pinecone`, `qdrant`, `chroma`, `weaviate`, `milvus`, `inmemory` |
| `search_provider` | 6 | `google_cse`, `bing`, `brave`, `serpapi`, `tavily`, `exa` |
| `notification_channel` | 7 | `slack`, `teams`, `webhook`, `log`, `email_smtp`, `discord`, `twilio` |
| `interaction_surface` | 1 (+2 dual) | `lab_json`; `slack` / `teams` also register this category |
| `collaboration_suite` | 2 | `ms365_graph`, `google_workspace` |
| `issue_tracker` | 4 | `jira`, `github`, `linear`, `azure_devops` |
| `wiki_knowledge` | 3 | `confluence`, `notion`, `sharepoint` |
| `observability_backend` | 7 | `prometheus`, `elasticsearch`, `otel`, `langfuse`, `datadog`, `clickhouse`, **`sentry`** |
| `browser_automation` | 3 | `playwright`, `firecrawl`, `selenium` |
| `secrets_store` | 1 | `vault` |
| `graph_store` | 1 | `neo4j` |
| `cloud_platform` | 3 | `aws`, `azure`, `gcp` |

**Total unique slugs:** 73.

### Implementation depth (code audit)

| Depth | Count | Meaning |
|-------|------:|---------|
| **full** | 24 | Dedicated `adapter` + `opens` (+ `config`); SDK in `opens.py` |
| **full-partial** | 6 | `opens` / `client` without full adapter split |
| **thin-p2** | 22 | `_shared/p2/factories.py` |
| **thin-p3** | 21 | `_shared/p3/factories.py` (Phase M.7 harness + **sentry**) |

**Thin-p3 slugs:** `tavily`, `exa`, `weaviate`, `milvus`, `inmemory`, `vault`, `langfuse`, `datadog`, `clickhouse`, **`sentry`**, `temporal`, `nats`, `neo4j`, `snowflake`, `supabase`, `minio`, `filesystem`, `discord`, `twilio`, `firecrawl`, `selenium`.

**Vector-store bridges** (`pinecone`, `qdrant`, `chroma`): RAG implementation in `intergrax/rag/vectorstore/`.

Unit tests: `tests/unit/integrations/` — **335+** cases including `test_p2_providers.py` and `test_p3_providers.py`.

### Reserved in `IntegrationSlug` but not registered

| Slug | Category | Notes |
|------|----------|-------|
| `reddit` | search_provider | Social search API |
| `google_places` | search_provider | Places API |
| `slash_command` | interaction_surface | Generic slash-command intake |

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

| Provider | Usage guide |
|----------|-------------|
| `slack` | [USAGE.md](../intergrax/integrations/providers/notification_channel/slack/USAGE.md) |
| `teams` | [USAGE.md](../intergrax/integrations/providers/notification_channel/teams/USAGE.md) |
| `lab_json` | [USAGE.md](../intergrax/integrations/providers/interaction_surface/lab_json/USAGE.md) |

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

---

## Recommended next integrations (harness AI gap analysis)

Audyt względem typowych stacków agentowych (LangGraph, CrewAI, LlamaIndex, enterprise VPC). **Zaimplementowane** oznacza slug w katalogu; **Brak** — warto rozważyć w kolejnej iteracji.

| Priorytet | Slug | Kategoria | Status | Dlaczego harness |
|-----------|------|-----------|--------|------------------|
| **Wysoki** | `sentry` | observability_backend | **Done** | Error tracking, release health, agent failure alerts |
| **Wysoki** | `langsmith` | observability_backend | Brak | LLM trace/debug (LangChain ecosystem) — obok `langfuse` |
| **Wysoki** | `helicone` | observability_backend | Brak | LLM cost/latency proxy — popularny w prod agentach |
| **Wysoki** | `pagerduty` / `opsgenie` | notification_channel | Brak | On-call escalation po HITL / agent failure |
| **Średni** | `posthog` | observability_backend | Brak | Product analytics + feature flags dla agent apps |
| **Średni** | `braintrust` | observability_backend | Brak | Evals + regression dla promptów agentów |
| **Średni** | `gitlab` | issue_tracker | Brak | DevOps issue flow (mamy GitHub/Azure DevOps) |
| **Średni** | `signoz` / `honeycomb` | observability_backend | Brak | OTEL-native APM (częściowo pokryte przez `otel`) |
| **Średni** | `arize` / `phoenix` | observability_backend | Brak | ML/RAG eval + drift (w planie jako `arize`) |
| **Niski** | `reddit` | search_provider | Reserved | Social research |
| **Niski** | `google_places` | search_provider | Reserved | Geo POI |
| **Future** | `opensearch`, `vespa`, `wandb` | various | Brak | Search/RAG scale, experiment tracking |

**Już silne pokrycie harness:** 73 integracje — SQL (10), kolejki (8), blob (5), wektory (6), search (6), observability (7 incl. Sentry), Vault, Neo4j, Temporal/NATS, Firecrawl, Tavily/Exa, Discord/Twilio.

---

All **73** shipped providers include an English usage guide at `intergrax/integrations/providers/<category>/<slug>/USAGE.md`. Regenerate after catalog changes:

```bash
uv run python scripts/generate_integration_usage_docs.py
```

## Adding a new provider

1. Implement `intergrax/integrations/providers/<category>/<slug>/` (config, adapter, opens, bundle, register).
2. Register in `register_default_integrations()`.
3. Add unit tests under `tests/unit/integrations/providers/`.
4. Add an entry to `scripts/generate_integration_usage_docs.py` and run the generator (English `USAGE.md`).
5. Update this catalog and the implementation plan tracker.

Delivery checklist: [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) — Phase M.4 workflow.

---

## Tests

Integration catalog regression:

```bash
uv run pytest tests/unit/integrations/ -q
```

Conformance helpers: `intergrax/integrations/_shared/conformance.py`.
