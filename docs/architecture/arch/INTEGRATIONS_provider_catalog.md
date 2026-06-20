# INTEGRATIONS — provider catalog

**Parent hub:** [`INTEGRATIONS.md`](../INTEGRATIONS.md)

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
| `graph_store` | 3 (+3 planned) | **Shipped:** `neo4j`, `memgraph`, `falkordb` · **Planned (M-RAG.49–51 / H-INT-GRAPH):** `neptune`, `orientdb`, `arangodb` — see [`plan/RAG.md`](../plan/RAG.md) Phase M-RAG-GRAPH |
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

| Slug | Status | Catalog factory | Env prefix | Prod SLO soak (M-RAG.30) |
|------|--------|-----------------|------------|--------------------------|
| `qdrant` | **stable** | `create_qdrant_vector_store()` | `INTERGRAX_QDRANT` | Required — `test_vectorstore_real_backends.py` |
| `pgvector` | **stable** | `create_pgvector_vector_store()` | `INTERGRAX_PGVECTOR` | Required — DSN optional (in-memory fallback for harness) |
| `chroma` | **stable** | `create_chroma_vector_store()` | `INTERGRAX_CHROMA` | Required — HTTP mode default probe `localhost:8000` |
| `weaviate` | **stable** | `create_weaviate_vector_store()` | `INTERGRAX_WEAVIATE` | Required — probe `INTERGRAX_WEAVIATE_URL` / `localhost:8080` |
| `lancedb` | **stable** | `create_lancedb_vector_store()` | `INTERGRAX_LANCEDB` | Harness stable; soak gate optional per deployment |
| `typesense` | **stable** | `create_typesense_vector_store()` | `INTERGRAX_TYPESENSE` | Harness stable; soak gate optional per deployment |
| `pinecone` | beta | `create_pinecone_vector_store()` | `INTERGRAX_PINECONE` | Promote to stable after soak passes in ops environment |
| `milvus` | beta | `create_milvus_vector_store()` | `INTERGRAX_MILVUS` | Promote to stable after soak passes in ops environment |
| `vespa` | beta | `create_vespa_vector_store()` | `INTERGRAX_VESPA` | Promote to stable after soak passes in ops environment |
| `inmemory` | beta | `create_inmemory_vector_store()` | `INTERGRAX_INMEMORY` | Harness/tests only — not for production |

**Soak contract:** `intergrax/rag/vectorstore/soak/prod_slo.py` — ingest → query → metadata filter → delete; p95 query latency budget.

**Gate (CI, no external services):** `uv run pytest tests/unit/rag/vectorstore/test_vectorstore_prod_slo_soak.py -m gate -q`

**Ops / nightly (real backends):** `uv run pytest tests/integration/rag/vectorstore/test_vectorstore_real_backends.py -m vectorstore_soak -q` — tests skip when backend unreachable.

| Provider | Usage guide |
|----------|-------------|
| `pinecone` | [USAGE.md](../intergrax/integrations/providers/vector_store/pinecone/USAGE.md) |
| `qdrant` | [USAGE.md](../intergrax/integrations/providers/vector_store/qdrant/USAGE.md) |
| `chroma` | [USAGE.md](../intergrax/integrations/providers/vector_store/chroma/USAGE.md) |
| `pgvector` | [USAGE.md](../intergrax/integrations/providers/vector_store/pgvector/USAGE.md) |
| `weaviate` | [USAGE.md](../intergrax/integrations/providers/vector_store/weaviate/USAGE.md) |

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

## Ingress / nginx bridge (INT-MAINT-04)

The **nginx / ingress controller** catalog slug is **not** owned by Integrations.
Capacity ingress is documented and implemented under
[`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) (ECP-6.*).
Decision: [ADR-SCALE-002](../adr/entries/2026-06-09/ADR-SCALE-002.md) — defer
standalone nginx slug; Kubernetes deployment path remains canonical.

Integrations cross-ref only. Host authors enable ingress via ECP profiles and
`kubernetes` integration — see [`intergrax/integrations/USAGE.md`](../../intergrax/integrations/USAGE.md).

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
