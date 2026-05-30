# Intergrax Integration Library

**Last updated:** 2026-05-29

The **Integration Library** (`intergrax/integrations/`) is Intergrax’s modular catalog of external systems — databases, queues, search APIs, vector indexes, cloud platforms, and collaboration tools. Agents and applications wire backends **by category**, not by vendor SDK, so the same agent code can run in a local lab, a customer VPC, or a multi-cloud deployment.

**Related docs:**

| Document | Purpose |
|----------|---------|
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1 | Architecture canon — tiers, contracts, registry rules |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase M | Phase status, backlog, delivery workflow |
| [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) Appendix E | How agents vs applications use integrations |
| Per-provider guides | `intergrax/integrations/providers/<slug>/USAGE.md` |

---

## Design principles

| Principle | What it means |
|-----------|---------------|
| **Universal contracts** | Each category (`relational_store`, `vector_store`, `message_bus`, …) defines a small Protocol. Providers implement the contract; agent logic depends on the contract only. |
| **Modular providers** | One slug = one package under `providers/<slug>/`. Swap Redis for ElastiCache, SQLite for PostgreSQL, or Chroma for Pinecone by changing `IntegrationProfile` — no agent refactor. |
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
| `observability_backend` | `ObservabilityBackend` | Metrics and log search |
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

## Implemented providers (29)

All providers below are registered in `register_default_integrations()`.  
**Status:** `stable` = production-ready catalog entry; `beta` = shipped, API may evolve.

### Summary by category

| Category | Count | Slugs |
|----------|------:|-------|
| `relational_store` | 4 | `sqlite`, `postgresql`, `mysql`, `databricks` |
| `document_store` | 2 | `cassandra`, `mongodb` |
| `key_value_cache` | 1 | `redis` |
| `message_bus` | 3 | `kafka`, `celery`, `rabbitmq` |
| `object_storage` | 1 | `s3` |
| `vector_store` | 3 | `pinecone`, `qdrant`, `chroma` |
| `search_provider` | 2 | `google_cse`, `bing` |
| `notification_channel` | 4 | `slack`, `teams`, `webhook`, `log` |
| `interaction_surface` | 3 | `slack`, `teams`, `lab_json` |
| `collaboration_suite` | 1 | `ms365_graph` |
| `issue_tracker` | 1 | `jira` |
| `wiki_knowledge` | 1 | `confluence` |
| `observability_backend` | 2 | `prometheus`, `elasticsearch` |
| `cloud_platform` | 3 | `aws`, `azure`, `gcp` |

---

### Relational store

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `sqlite` | stable | `create_sqlite_relational_store()` | `INTERGRAX_SQLITE` | Lab default; also trace, checkpoint, HITL stores via bundle helpers |
| `postgresql` | beta | `create_postgresql_relational_store()` | `INTERGRAX_POSTGRESQL` | psycopg3; DSN or host/port/user/password |
| `mysql` | beta | `create_mysql_relational_store()` | `INTERGRAX_MYSQL` | pymysql |
| `databricks` | beta | `create_databricks_relational_store()` | `INTERGRAX_DATABRICKS` | SQL Warehouse / Unity Catalog |

| Provider | Usage guide |
|----------|-------------|
| `sqlite` | [USAGE.md](../intergrax/integrations/providers/sqlite/USAGE.md) |
| `postgresql` | [USAGE.md](../intergrax/integrations/providers/postgresql/USAGE.md) |
| `mysql` | [USAGE.md](../intergrax/integrations/providers/mysql/USAGE.md) |
| `databricks` | [USAGE.md](../intergrax/integrations/providers/databricks/USAGE.md) |

---

### Document store

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `cassandra` | beta | `create_cassandra_document_store()` | `INTERGRAX_CASSANDRA` | CQL partition-scoped CRUD |
| `mongodb` | beta | `create_mongodb_document_store()` | `INTERGRAX_MONGODB` | Flexible JSON documents via PyMongo |

| Provider | Usage guide |
|----------|-------------|
| `cassandra` | [USAGE.md](../intergrax/integrations/providers/cassandra/USAGE.md) |
| `mongodb` | [USAGE.md](../intergrax/integrations/providers/mongodb/USAGE.md) |

---

### Key-value cache

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `redis` | stable | `create_redis_key_value_cache()` | `INTERGRAX_REDIS` | Also idempotency, rate limit, semaphore via `create_redis_integration()` |

| Provider | Usage guide |
|----------|-------------|
| `redis` | [USAGE.md](../intergrax/integrations/providers/redis/USAGE.md) |

---

### Message bus

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `kafka` | stable | `create_kafka_message_bus()` | `INTERGRAX_KAFKA` | Runtime transport delegates here |
| `celery` | stable | `create_celery_message_bus()` | `INTERGRAX_CELERY` | Broker/backend via env or injected app |
| `rabbitmq` | stable | `create_rabbitmq_message_bus()` | `INTERGRAX_RABBITMQ` | Requires KV store for ack semantics |

| Provider | Usage guide |
|----------|-------------|
| `kafka` | [USAGE.md](../intergrax/integrations/providers/kafka/USAGE.md) |
| `celery` | [USAGE.md](../intergrax/integrations/providers/celery/USAGE.md) |
| `rabbitmq` | [USAGE.md](../intergrax/integrations/providers/rabbitmq/USAGE.md) |

---

### Object storage

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `s3` | beta | `create_s3_object_storage()` | `INTERGRAX_S3` | put / get / delete / presigned_url; boto3 in `opens.py` only |

| Provider | Usage guide |
|----------|-------------|
| `s3` | [USAGE.md](../intergrax/integrations/providers/s3/USAGE.md) |

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
| `pinecone` | [USAGE.md](../intergrax/integrations/providers/pinecone/USAGE.md) |
| `qdrant` | [USAGE.md](../intergrax/integrations/providers/qdrant/USAGE.md) |
| `chroma` | [USAGE.md](../intergrax/integrations/providers/chroma/USAGE.md) |

RAG bootstrap: `create_default_vectorstore_manager()` in `intergrax/rag/vectorstore/bootstrap/` resolves via the integration catalog when `vector_store` is configured.

---

### Search provider

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `google_cse` | stable | `create_google_cse_search_provider()` | `INTERGRAX_GOOGLE_CSE` | Google Custom Search |
| `bing` | stable | `create_bing_search_provider()` | `INTERGRAX_BING` | Bing Web Search v7 |

| Provider | Usage guide |
|----------|-------------|
| `google_cse` | [USAGE.md](../intergrax/integrations/providers/google_cse/USAGE.md) |
| `bing` | [USAGE.md](../intergrax/integrations/providers/bing/USAGE.md) |

---

### Notification channel

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `slack` | stable | `create_slack_notification_channel()` | `INTERGRAX_SLACK` | Also registered as `interaction_surface` |
| `teams` | stable | `create_teams_notification_channel()` | `INTERGRAX_TEAMS` | Also registered as `interaction_surface` |
| `webhook` | stable | `create_webhook_notification_channel()` | `INTERGRAX_WEBHOOK` | Generic HTTP outbound |
| `log` | stable | `create_log_notification_channel()` | `INTERGRAX_LOG` | Process log; lab profile default |

| Provider | Usage guide |
|----------|-------------|
| `slack` | [USAGE.md](../intergrax/integrations/providers/slack/USAGE.md) |
| `teams` | [USAGE.md](../intergrax/integrations/providers/teams/USAGE.md) |
| `webhook` | [USAGE.md](../intergrax/integrations/providers/webhook/USAGE.md) |
| `log` | [USAGE.md](../intergrax/integrations/providers/log/USAGE.md) |

---

### Interaction surface

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `slack` | stable | `create_slack_interaction_surface()` | `INTERGRAX_SLACK` | Inbound slash commands / events |
| `teams` | stable | `create_teams_interaction_surface()` | `INTERGRAX_TEAMS` | Microsoft Teams activity intake |
| `lab_json` | stable | `create_lab_json_interaction_surface()` | `INTERGRAX_LAB_JSON` | Laboratory JSON webhook intake |

| Provider | Usage guide |
|----------|-------------|
| `slack` | [USAGE.md](../intergrax/integrations/providers/slack/USAGE.md) |
| `teams` | [USAGE.md](../intergrax/integrations/providers/teams/USAGE.md) |
| `lab_json` | [USAGE.md](../intergrax/integrations/providers/lab_json/USAGE.md) |

---

### Collaboration suite

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `ms365_graph` | beta | `create_ms365_graph_collaboration_suite()` | `INTERGRAX_MS365_GRAPH` | Mail, calendar, directory via Microsoft Graph |

| Provider | Usage guide |
|----------|-------------|
| `ms365_graph` | [USAGE.md](../intergrax/integrations/providers/ms365_graph/USAGE.md) |

---

### Issue tracker

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `jira` | beta | `create_jira_issue_tracker()` | `INTERGRAX_JIRA` | REST v3 — issues, comments, search |

| Provider | Usage guide |
|----------|-------------|
| `jira` | [USAGE.md](../intergrax/integrations/providers/jira/USAGE.md) |

---

### Wiki / knowledge

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `confluence` | beta | `create_confluence_wiki_knowledge()` | `INTERGRAX_CONFLUENCE` | Pages and search for RAG / runbooks |

| Provider | Usage guide |
|----------|-------------|
| `confluence` | [USAGE.md](../intergrax/integrations/providers/confluence/USAGE.md) |

---

### Observability backend

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `prometheus` | beta | `create_prometheus_observability_backend()` | `INTERGRAX_PROMETHEUS` | PromQL instant / range queries |
| `elasticsearch` | beta | `create_elasticsearch_observability_backend()` | `INTERGRAX_ELASTICSEARCH` | Log search and aggregations |

| Provider | Usage guide |
|----------|-------------|
| `prometheus` | [USAGE.md](../intergrax/integrations/providers/prometheus/USAGE.md) |
| `elasticsearch` | [USAGE.md](../intergrax/integrations/providers/elasticsearch/USAGE.md) |

---

### Cloud platform

| Slug | Status | Catalog factory | Env prefix | Notes |
|------|--------|-----------------|------------|-------|
| `aws` | beta | `create_aws_cloud_platform()` | `INTERGRAX_AWS` | IAM / STS; defaults for S3, SQS, DynamoDB, ElastiCache |
| `azure` | beta | `create_azure_cloud_platform()` | `INTERGRAX_AZURE` | MI / service principal; defaults for Blob, Service Bus |
| `gcp` | beta | `create_gcp_cloud_platform()` | `INTERGRAX_GCP` | ADC / service account; defaults for GCS, Pub/Sub |

| Provider | Usage guide |
|----------|-------------|
| `aws` | [USAGE.md](../intergrax/integrations/providers/aws/USAGE.md) |
| `azure` | [USAGE.md](../intergrax/integrations/providers/azure/USAGE.md) |
| `gcp` | [USAGE.md](../intergrax/integrations/providers/gcp/USAGE.md) |

---

## Full provider index

Alphabetical reference — all shipped integrations in one table.

| Slug | Category (categories) | Status | Catalog factory | Usage |
|------|----------------------|--------|-----------------|-------|
| `aws` | `cloud_platform` | beta | `create_aws_cloud_platform()` | [USAGE](../intergrax/integrations/providers/aws/USAGE.md) |
| `azure` | `cloud_platform` | beta | `create_azure_cloud_platform()` | [USAGE](../intergrax/integrations/providers/azure/USAGE.md) |
| `bing` | `search_provider` | stable | `create_bing_search_provider()` | [USAGE](../intergrax/integrations/providers/bing/USAGE.md) |
| `cassandra` | `document_store` | beta | `create_cassandra_document_store()` | [USAGE](../intergrax/integrations/providers/cassandra/USAGE.md) |
| `celery` | `message_bus` | stable | `create_celery_message_bus()` | [USAGE](../intergrax/integrations/providers/celery/USAGE.md) |
| `chroma` | `vector_store` | beta | `create_chroma_vector_store()` | [USAGE](../intergrax/integrations/providers/chroma/USAGE.md) |
| `confluence` | `wiki_knowledge` | beta | `create_confluence_wiki_knowledge()` | [USAGE](../intergrax/integrations/providers/confluence/USAGE.md) |
| `databricks` | `relational_store` | beta | `create_databricks_relational_store()` | [USAGE](../intergrax/integrations/providers/databricks/USAGE.md) |
| `elasticsearch` | `observability_backend` | beta | `create_elasticsearch_observability_backend()` | [USAGE](../intergrax/integrations/providers/elasticsearch/USAGE.md) |
| `gcp` | `cloud_platform` | beta | `create_gcp_cloud_platform()` | [USAGE](../intergrax/integrations/providers/gcp/USAGE.md) |
| `google_cse` | `search_provider` | stable | `create_google_cse_search_provider()` | [USAGE](../intergrax/integrations/providers/google_cse/USAGE.md) |
| `jira` | `issue_tracker` | beta | `create_jira_issue_tracker()` | [USAGE](../intergrax/integrations/providers/jira/USAGE.md) |
| `kafka` | `message_bus` | stable | `create_kafka_message_bus()` | [USAGE](../intergrax/integrations/providers/kafka/USAGE.md) |
| `lab_json` | `interaction_surface` | stable | `create_lab_json_interaction_surface()` | [USAGE](../intergrax/integrations/providers/lab_json/USAGE.md) |
| `log` | `notification_channel` | stable | `create_log_notification_channel()` | [USAGE](../intergrax/integrations/providers/log/USAGE.md) |
| `mongodb` | `document_store` | beta | `create_mongodb_document_store()` | [USAGE](../intergrax/integrations/providers/mongodb/USAGE.md) |
| `ms365_graph` | `collaboration_suite` | beta | `create_ms365_graph_collaboration_suite()` | [USAGE](../intergrax/integrations/providers/ms365_graph/USAGE.md) |
| `mysql` | `relational_store` | beta | `create_mysql_relational_store()` | [USAGE](../intergrax/integrations/providers/mysql/USAGE.md) |
| `pinecone` | `vector_store` | beta | `create_pinecone_vector_store()` | [USAGE](../intergrax/integrations/providers/pinecone/USAGE.md) |
| `postgresql` | `relational_store` | beta | `create_postgresql_relational_store()` | [USAGE](../intergrax/integrations/providers/postgresql/USAGE.md) |
| `prometheus` | `observability_backend` | beta | `create_prometheus_observability_backend()` | [USAGE](../intergrax/integrations/providers/prometheus/USAGE.md) |
| `qdrant` | `vector_store` | beta | `create_qdrant_vector_store()` | [USAGE](../intergrax/integrations/providers/qdrant/USAGE.md) |
| `rabbitmq` | `message_bus` | stable | `create_rabbitmq_message_bus()` | [USAGE](../intergrax/integrations/providers/rabbitmq/USAGE.md) |
| `redis` | `key_value_cache` | stable | `create_redis_key_value_cache()` | [USAGE](../intergrax/integrations/providers/redis/USAGE.md) |
| `s3` | `object_storage` | beta | `create_s3_object_storage()` | [USAGE](../intergrax/integrations/providers/s3/USAGE.md) |
| `slack` | `notification_channel`, `interaction_surface` | stable | `create_slack_catalog_factory()` | [USAGE](../intergrax/integrations/providers/slack/USAGE.md) |
| `sqlite` | `relational_store` | stable | `create_sqlite_relational_store()` | [USAGE](../intergrax/integrations/providers/sqlite/USAGE.md) |
| `teams` | `notification_channel`, `interaction_surface` | stable | `create_teams_catalog_factory()` | [USAGE](../intergrax/integrations/providers/teams/USAGE.md) |
| `webhook` | `notification_channel` | stable | `create_webhook_notification_channel()` | [USAGE](../intergrax/integrations/providers/webhook/USAGE.md) |

---

## Planned providers (not yet implemented)

Slugs reserved in the catalog; see [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase M.6 P2/P3 for priority.

| Priority | Slugs | Category |
|----------|-------|----------|
| High | `azure_blob`, `gcs` | `object_storage` |
| High | `email_smtp` | `notification_channel` |
| Medium | `notion`, `sharepoint` | `wiki_knowledge` |
| Medium | `github`, `linear` | `issue_tracker` |
| Medium | `google_workspace` | `collaboration_suite` |
| Medium | `otel` | `observability_backend` |
| Medium | `brave`, `serpapi` | `search_provider` |
| Medium | `playwright` | `browser_automation` |
| Low | `sqs`, `service_bus`, `pubsub`, `dynamodb`, `elasticache`, `memcached` | various (often via cloud facades) |

---

## Adding a new provider

1. Implement `intergrax/integrations/providers/<slug>/` (config, adapter, opens, bundle, register).
2. Register in `register_default_integrations()`.
3. Add unit tests under `tests/unit/integrations/providers/`.
4. Add `providers/<slug>/USAGE.md` (English).
5. Update this catalog and the implementation plan tracker.

Delivery checklist: [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) — Phase M.4 workflow.

---

## Tests

Integration catalog regression:

```bash
uv run pytest tests/unit/integrations/ -q
```

Conformance helpers: `intergrax/integrations/_shared/conformance.py`.
