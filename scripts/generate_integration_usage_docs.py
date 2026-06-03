# © Artur Czarnecki. All rights reserved.
# Generator for intergrax/integrations/providers/*/USAGE.md (English only).

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS_DIR = ROOT / "intergrax" / "integrations" / "providers"

COMMON_HEADER = """\
> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile({profile_field}=IntegrationSlug.{slug_enum})
backend = profile.resolve(IntegrationCategory.{category_enum})
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.{category}.{slug}.bundle import {factory}

backend = {factory}(**config_overrides)
```
"""

PROVIDERS: list[dict[str, str]] = [
    {
        "slug": "aws",
        "category": "cloud_platform",
        "category_enum": "CLOUD_PLATFORM",
        "slug_enum": "AWS",
        "factory": "create_aws_cloud_platform",
        "env": "`INTERGRAX_AWS_REGION`, `INTERGRAX_AWS_PROFILE`; optional keys or `INTERGRAX_AWS_ROLE_ARN`",
        "example": """\
platform = create_aws_cloud_platform(region="eu-central-1")
health = platform.health()
s3_slug = platform.resolve("object_storage")  # -> "s3"
""",
        "notes": "boto3 SDK only in ``opens.py``. The facade does not implement S3/SQS — it returns default category slugs.",
    },
    {
        "slug": "azure",
        "category": "cloud_platform",
        "category_enum": "CLOUD_PLATFORM",
        "slug_enum": "AZURE",
        "factory": "create_azure_cloud_platform",
        "env": "`INTERGRAX_AZURE_TENANT_ID`, `INTERGRAX_AZURE_CLIENT_ID`, `INTERGRAX_AZURE_CLIENT_SECRET` (or managed identity)",
        "example": """\
platform = create_azure_cloud_platform()
blob_slug = platform.resolve("object_storage")  # -> "azure_blob"
""",
        "notes": "``azure-identity`` only in ``opens.py``.",
    },
    {
        "slug": "gcp",
        "category": "cloud_platform",
        "category_enum": "CLOUD_PLATFORM",
        "slug_enum": "GCP",
        "factory": "create_gcp_cloud_platform",
        "env": "`INTERGRAX_GCP_PROJECT_ID`, `INTERGRAX_GCP_REGION`, `INTERGRAX_GCP_CREDENTIALS_FILE` (or ADC)",
        "example": """\
platform = create_gcp_cloud_platform(project_id="my-project")
gcs_slug = platform.resolve("object_storage")  # -> "gcs"
""",
        "notes": "``google-auth`` only in ``opens.py``.",
    },
    {
        "slug": "sqlite",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "SQLITE",
        "factory": "create_sqlite_relational_store",
        "env": "`INTERGRAX_SQLITE_DATA_DIR` (directory for `.db` files)",
        "example": """\
store = create_sqlite_relational_store(data_dir="build/lab")
store.connect()
store.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT)")
store.fetch_all("SELECT * FROM items")
store.close()
""",
        "notes": "Bundle also exposes ``create_sqlite_trace_store()``, ``create_sqlite_runtime_event_store()``, etc.",
    },
    {
        "slug": "postgresql",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "POSTGRESQL",
        "factory": "create_postgresql_relational_store",
        "env": "`INTERGRAX_POSTGRESQL_DSN` or HOST/PORT/USER/PASSWORD/DATABASE; optional `INTERGRAX_POSTGRESQL_SCHEMA`",
        "example": """\
store = create_postgresql_relational_store(dsn="postgresql://user:pass@localhost:5432/app")
store.execute("INSERT INTO items (name) VALUES (%s)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
""",
        "notes": "``psycopg.connect`` only in ``opens.py``.",
    },
    {
        "slug": "mysql",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "MYSQL",
        "factory": "create_mysql_relational_store",
        "env": "`INTERGRAX_MYSQL_DSN` or component vars; optional `INTERGRAX_MYSQL_TENANT_DATABASE`",
        "example": """\
store = create_mysql_relational_store(host="127.0.0.1", user="app", password="secret", database="intergrax")
store.execute("INSERT INTO items (name) VALUES (%s)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
""",
        "notes": "``pymysql.connect`` only in ``opens.py``.",
    },
    {
        "slug": "databricks",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "DATABRICKS",
        "factory": "create_databricks_relational_store",
        "env": "`INTERGRAX_DATABRICKS_HOST`, `INTERGRAX_DATABRICKS_HTTP_PATH`, `INTERGRAX_DATABRICKS_TOKEN`; optional CATALOG/SCHEMA",
        "example": """\
store = create_databricks_relational_store(
    host="adb-123.4.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc",
    access_token="dapi-...",
)
rows = store.fetch_all("SELECT id, name FROM analytics.events LIMIT 10")
store.close()
""",
        "notes": "``databricks.sql.connect`` only in ``opens.py``.",
    },
    {
        "slug": "cassandra",
        "category": "document_store",
        "category_enum": "DOCUMENT_STORE",
        "slug_enum": "CASSANDRA",
        "factory": "create_cassandra_document_store",
        "env": "`INTERGRAX_CASSANDRA_CONTACT_POINTS`, `INTERGRAX_CASSANDRA_KEYSPACE`; optional USER/PASSWORD/TABLE",
        "example": """\
from intergrax.integrations.contracts.document_store import DocumentRecord

store = create_cassandra_document_store(contact_points="127.0.0.1", keyspace="intergrax")
store.put(DocumentRecord(partition_key="tenant-1", row_key="evt-1", data={"status": "ok"}))
doc = store.get("tenant-1", "evt-1")
result = store.query("tenant-1", limit=50, row_key_prefix="2026-")
store.close()
""",
        "notes": "Cassandra driver only in ``opens.py``.",
    },
    {
        "slug": "mongodb",
        "category": "document_store",
        "category_enum": "DOCUMENT_STORE",
        "slug_enum": "MONGODB",
        "factory": "create_mongodb_document_store",
        "env": "`INTERGRAX_MONGODB_URI`, `INTERGRAX_MONGODB_DATABASE`, `INTERGRAX_MONGODB_COLLECTION`",
        "example": """\
from intergrax.integrations.contracts.document_store import DocumentRecord

store = create_mongodb_document_store(uri="mongodb://localhost:27017")
store.put(DocumentRecord(partition_key="tenant-1", row_key="mem-1", data={"topic": "onboarding"}))
doc = store.get("tenant-1", "mem-1")
store.close()
""",
        "notes": "``pymongo.MongoClient`` only in ``opens.py``.",
    },
    {
        "slug": "redis",
        "category": "key_value_cache",
        "category_enum": "KEY_VALUE_CACHE",
        "slug_enum": "REDIS",
        "factory": "create_redis_key_value_cache",
        "env": "`INTERGRAX_REDIS_URL`; optional `INTERGRAX_REDIS_DB`, `INTERGRAX_REDIS_KEY_PREFIX`",
        "example": """\
cache = create_redis_key_value_cache(url="redis://localhost:6379/0")
cache.set("session:42", b"payload", ttl_seconds=3600)
value = cache.get("session:42")
cache.delete("session:42")
""",
        "notes": "Bundle also provides idempotency, rate limit, semaphore via ``create_redis_integration()``.",
    },
    {
        "slug": "kafka",
        "category": "message_bus",
        "category_enum": "MESSAGE_BUS",
        "slug_enum": "KAFKA",
        "factory": "create_kafka_message_bus",
        "env": "`INTERGRAX_KAFKA_BOOTSTRAP_SERVERS`, `INTERGRAX_KAFKA_TOPIC`, `INTERGRAX_KAFKA_CONSUMER_GROUP`",
        "example": """\
from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_kafka_message_bus(bootstrap_servers="localhost:9092", topic="intergrax.tasks")
handle = bus.enqueue(TaskRequest(task_id="t-1", payload={"agent": "echo"}))
status = bus.get_status(handle)
result = bus.get_result(handle)
""",
        "notes": "``confluent_kafka`` only in ``opens.py``.",
    },
    {
        "slug": "rabbitmq",
        "category": "message_bus",
        "category_enum": "MESSAGE_BUS",
        "slug_enum": "RABBITMQ",
        "factory": "create_rabbitmq_message_bus",
        "env": "`INTERGRAX_RABBITMQ_HOST`, `INTERGRAX_RABBITMQ_QUEUE`; optional USER/PASSWORD/VHOST",
        "example": """\
from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_rabbitmq_message_bus(host="localhost", queue="intergrax.tasks", kv_store=cache)
handle = bus.enqueue(TaskRequest(task_id="t-1", payload={"step": "run"}))
""",
        "notes": "Requires a ``kv_store`` (e.g. Redis) for task status. ``pika`` only in ``opens.py``.",
    },
    {
        "slug": "celery",
        "category": "message_bus",
        "category_enum": "MESSAGE_BUS",
        "slug_enum": "CELERY",
        "factory": "create_celery_message_bus",
        "env": "`INTERGRAX_CELERY_BROKER_URL`, `INTERGRAX_CELERY_BACKEND_URL`",
        "example": """\
from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_celery_message_bus(broker_url="redis://localhost:6379/1")
handle = bus.enqueue(TaskRequest(task_id="t-1", payload={"graph": "demo"}))
""",
        "notes": "You may inject an existing Celery ``app``. Workers: ``create_celery_worker_app()``.",
    },
    {
        "slug": "google_cse",
        "category": "search_provider",
        "category_enum": "SEARCH_PROVIDER",
        "slug_enum": "GOOGLE_CSE",
        "factory": "create_google_cse_search_provider",
        "env": "`INTERGRAX_GOOGLE_CSE_API_KEY`, `INTERGRAX_GOOGLE_CSE_CX`",
        "example": """\
search = create_google_cse_search_provider(api_key="...", cx="...")
hits = search.search("Intergrax agent orchestration", limit=5)
for hit in hits:
    print(hit.title, hit.url)
""",
        "notes": "Compatible with ``WebSearchExecutor`` via ``search.web_search_provider``.",
    },
    {
        "slug": "bing",
        "category": "search_provider",
        "category_enum": "SEARCH_PROVIDER",
        "slug_enum": "BING",
        "factory": "create_bing_search_provider",
        "env": "`INTERGRAX_BING_API_KEY` (legacy: `BING_SEARCH_V7_API_KEY`)",
        "example": """\
search = create_bing_search_provider(api_key="...")
hits = search.search("enterprise AI agents", limit=5)
""",
        "notes": "HTTP client only in ``opens.py``.",
    },
    {
        "slug": "slack",
        "category": "notification_channel + interaction_surface",
        "category_enum": "NOTIFICATION_CHANNEL",
        "slug_enum": "SLACK",
        "factory": "create_slack_notification_channel",
        "env": "`INTERGRAX_SLACK_WEBHOOK_URL`; optional `INTERGRAX_SLACK_SIGNING_SECRET` (inbound)",
        "example": """\
notifier = create_slack_notification_channel(webhook_url="https://hooks.slack.com/...")
notifier.notify("Task t-1 finished")

# Inbound (interaction):
from intergrax.integrations.providers.notification_channel.slack.bundle import create_slack_interaction_surface
surface = create_slack_interaction_surface(signing_secret="...")
# profile.resolve(IntegrationCategory.INTERACTION_SURFACE) when interaction_surface=SLACK
""",
        "notes": "Catalog registers both categories. ``create_slack_catalog_factory`` selects by ``IntegrationCategory``.",
    },
    {
        "slug": "teams",
        "category": "notification_channel + interaction_surface",
        "category_enum": "NOTIFICATION_CHANNEL",
        "slug_enum": "TEAMS",
        "factory": "create_teams_notification_channel",
        "env": "`INTERGRAX_TEAMS_WEBHOOK_URL`; optional `INTERGRAX_TEAMS_SECURITY_TOKEN`",
        "example": """\
notifier = create_teams_notification_channel(webhook_url="https://outlook.office.com/webhook/...")
notifier.notify("Nexus run completed")

# Inbound: create_teams_interaction_surface(security_token="...")
# profile.resolve(IntegrationCategory.INTERACTION_SURFACE) when interaction_surface=TEAMS
""",
        "notes": "Same dual-category pattern as Slack — separate factory for ``INTERACTION_SURFACE``.",
    },
    {
        "slug": "webhook",
        "category": "notification_channel",
        "category_enum": "NOTIFICATION_CHANNEL",
        "slug_enum": "WEBHOOK",
        "factory": "create_webhook_notification_channel",
        "env": "`INTERGRAX_WEBHOOK_URL`",
        "example": """\
notifier = create_webhook_notification_channel(url="https://example.com/hooks/intergrax")
notifier.notify({"event": "task.completed", "task_id": "t-1"})
""",
        "notes": "Generic HTTP POST; JSON formatting via ``GenericJsonPayloadFormatter``.",
    },
    {
        "slug": "log",
        "category": "notification_channel",
        "category_enum": "NOTIFICATION_CHANNEL",
        "slug_enum": "LOG",
        "factory": "create_log_notification_channel",
        "env": "None — uses the application logger",
        "example": """\
notifier = create_log_notification_channel()
notifier.notify("HITL: approval required for task t-1")
""",
        "notes": "Default channel in ``IntegrationProfile.lab()`` — no network.",
    },
    {
        "slug": "lab_json",
        "category": "interaction_surface",
        "category_enum": "INTERACTION_SURFACE",
        "slug_enum": "LAB_JSON",
        "factory": "create_lab_json_interaction_surface",
        "env": "Optional `INTERGRAX_LAB_JSON_DEFAULT_SOURCE`",
        "example": """\
surface = create_lab_json_interaction_surface()
if surface.can_handle(inbound_payload):
    message = surface.to_inbound(inbound_payload)
    print(message.text, message.channel)
""",
        "notes": "JSON intake for the lab; channel ``lab``.",
    },
    {
        "slug": "jira",
        "category": "issue_tracker",
        "category_enum": "ISSUE_TRACKER",
        "slug_enum": "JIRA",
        "factory": "create_jira_issue_tracker",
        "env": "`INTERGRAX_JIRA_BASE_URL`, `INTERGRAX_JIRA_EMAIL`, `INTERGRAX_JIRA_API_TOKEN`",
        "example": """\
tracker = create_jira_issue_tracker(base_url="https://acme.atlassian.net", email="bot@acme.com", api_token="...")
issue = tracker.get_issue("PROJ-123")
tracker.add_comment("PROJ-123", "Agent update: analysis complete")
results = tracker.search_issues('project = PROJ AND status = "In Progress"', limit=20)
""",
        "notes": "httpx only in ``opens.py``.",
    },
    {
        "slug": "confluence",
        "category": "wiki_knowledge",
        "category_enum": "WIKI_KNOWLEDGE",
        "slug_enum": "CONFLUENCE",
        "factory": "create_confluence_wiki_knowledge",
        "env": "`INTERGRAX_CONFLUENCE_BASE_URL`, `INTERGRAX_CONFLUENCE_EMAIL`, `INTERGRAX_CONFLUENCE_API_TOKEN`",
        "example": """\
wiki = create_confluence_wiki_knowledge(base_url="https://acme.atlassian.net/wiki", email="...", api_token="...")
page = wiki.get_page("123456")
results = wiki.search_pages("runbook deployment", limit=10)
""",
        "notes": "httpx only in ``opens.py``.",
    },
    {
        "slug": "prometheus",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "PROMETHEUS",
        "factory": "create_prometheus_observability_backend",
        "env": "`INTERGRAX_PROMETHEUS_BASE_URL`; optional `INTERGRAX_PROMETHEUS_BEARER_TOKEN`",
        "example": """\
obs = create_prometheus_observability_backend(base_url="http://prometheus:9090")
instant = obs.query_instant("up")
range_result = obs.query_range("rate(http_requests_total[5m])", start=1710000000, end=1710003600, step="1m")
""",
        "notes": "PromQL via HTTP API v1.",
    },
    {
        "slug": "elasticsearch",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "ELASTICSEARCH",
        "factory": "create_elasticsearch_observability_backend",
        "env": "`INTERGRAX_ELASTICSEARCH_URL`, `INTERGRAX_ELASTICSEARCH_INDEX`; optional USER/PASSWORD/API_KEY",
        "example": """\
obs = create_elasticsearch_observability_backend(base_url="http://localhost:9200", index="logs-*")
# The promql argument is a Lucene query_string (not PromQL):
instant = obs.query_instant('level:"error" AND service:"nexus"')
range_result = obs.query_range('status:500', start=1710000000, end=1710003600, step="15s")
""",
        "notes": "``promql`` in the API maps to Lucene ``query_string``. httpx only in ``opens.py``.",
    },
    {
        "slug": "ms365_graph",
        "category": "collaboration_suite",
        "category_enum": "COLLABORATION_SUITE",
        "slug_enum": "MS365_GRAPH",
        "factory": "create_ms365_graph_collaboration_suite",
        "env": "`INTERGRAX_MS365_TENANT_ID`, `INTERGRAX_MS365_CLIENT_ID`, `INTERGRAX_MS365_CLIENT_SECRET`",
        "example": """\
suite = create_ms365_graph_collaboration_suite(tenant_id="...", client_id="...", client_secret="...")
user = suite.get_user("user@contoso.com")
events = suite.list_calendar_events(user_id=user.id, start="2026-05-01", end="2026-05-31")
suite.send_mail(to=["user@contoso.com"], subject="Report", body="...")
""",
        "notes": "OAuth client credentials in ``opens.py``.",
    },
    {
        "slug": "pinecone",
        "category": "vector_store",
        "category_enum": "VECTOR_STORE",
        "slug_enum": "PINECONE",
        "factory": "create_pinecone_vector_store",
        "env": "`INTERGRAX_PINECONE_API_KEY`, `INTERGRAX_PINECONE_INDEX`; optional `INTERGRAX_PINECONE_TENANT_ID`, `INTERGRAX_PINECONE_COLLECTION`, `INTERGRAX_PINECONE_METRIC`",
        "example": """\
from langchain_core.documents import Document

store = create_pinecone_vector_store(api_key="pc-...", index_name="intergrax-rag", tenant_id="tenant-a")
store.add_documents(
    [Document(page_content="Intergrax overview", metadata={"source": "docs"})],
    [[0.01, 0.02, 0.03]],
    ids=["doc-1"],
)
hits = store.query([0.01, 0.02, 0.03], top_k=5)
store.delete(["doc-1"])
""",
        "notes": "Catalog bridge to ``intergrax/rag/`` — Pinecone SDK import only in ``opens.py``; RAG implementation unchanged.",
    },
    {
        "slug": "qdrant",
        "category": "vector_store",
        "category_enum": "VECTOR_STORE",
        "slug_enum": "QDRANT",
        "factory": "create_qdrant_vector_store",
        "env": "`INTERGRAX_QDRANT_URL` or `INTERGRAX_QDRANT_HOST`/`INTERGRAX_QDRANT_PORT`; optional `INTERGRAX_QDRANT_API_KEY`, `INTERGRAX_QDRANT_COLLECTION`, `INTERGRAX_QDRANT_TENANT_ID`, `INTERGRAX_QDRANT_METRIC`",
        "example": """\
store = create_qdrant_vector_store(
    collection_name="intergrax-rag",
    tenant_id="tenant-a",
    host="localhost",
    port=6333,
)
store.add_documents(
    [Document(page_content="Intergrax overview", metadata={"source": "docs"})],
    [[0.01, 0.02, 0.03]],
    ids=["doc-1"],
)
hits = store.query([0.01, 0.02, 0.03], top_k=5)
store.delete(["doc-1"])
""",
        "notes": "Catalog bridge to ``intergrax/rag/`` — ``qdrant_client`` import only in ``opens.py``; RAG ``QdrantVectorStore`` unchanged.",
    },
    {
        "slug": "chroma",
        "category": "vector_store",
        "category_enum": "VECTOR_STORE",
        "slug_enum": "CHROMA",
        "factory": "create_chroma_vector_store",
        "env": "`INTERGRAX_CHROMA_MODE` (`embedded`|`http`); optional `INTERGRAX_CHROMA_HOST`, `INTERGRAX_CHROMA_PORT`, `INTERGRAX_CHROMA_PERSIST_DIRECTORY`, `INTERGRAX_CHROMA_COLLECTION`, `INTERGRAX_CHROMA_TENANT_ID`",
        "example": """\
store = create_chroma_vector_store(
    collection_name="intergrax-rag",
    tenant_id="tenant-a",
    mode="embedded",
    persist_directory=None,
)
store.add_documents(
    [Document(page_content="Intergrax overview", metadata={"source": "docs"})],
    [[0.01, 0.02, 0.03]],
    ids=["doc-1"],
)
hits = store.query([0.01, 0.02, 0.03], top_k=5)
store.delete(["doc-1"])
""",
        "notes": "Catalog bridge to ``intergrax/rag/`` — ``chromadb`` import only in ``opens.py``; RAG ``ChromaVectorStore`` unchanged.",
    },
    {
        "slug": "s3",
        "category": "object_storage",
        "category_enum": "OBJECT_STORAGE",
        "slug_enum": "S3",
        "factory": "create_s3_object_storage",
        "env": "`INTERGRAX_S3_BUCKET` (required); optional `INTERGRAX_S3_REGION`, `INTERGRAX_S3_PREFIX`, `INTERGRAX_S3_ENDPOINT_URL`, AWS credential vars",
        "example": """\
store = create_s3_object_storage(bucket="intergrax-artifacts", region="eu-central-1", prefix="tenant-a")
store.put("exports/run-1.zip", file_bytes, content_type="application/zip")
obj = store.get("exports/run-1.zip")
url = store.presigned_url("exports/run-1.zip", expires_in_seconds=900)
store.delete("exports/run-1.zip")
""",
        "notes": "boto3 S3 client only in ``opens.py``. With ``IntegrationProfile(cloud_platform="aws")``, ``object_storage`` resolves to ``s3`` by default.",
    },
    # --- Phase M.6 P2/P3 (2026-05-30) ---
    {
        "slug": "azure_blob",
        "category": "object_storage",
        "category_enum": "OBJECT_STORAGE",
        "slug_enum": "AZURE_BLOB",
        "factory": "create_azure_blob_object_storage",
        "env": "`INTERGRAX_AZURE_BLOB_CONTAINER` (required); optional `INTERGRAX_AZURE_BLOB_PREFIX`, `INTERGRAX_AZURE_BLOB_CONNECTION_STRING`, `INTERGRAX_AZURE_BLOB_ACCOUNT_URL`",
        "example": """\
store = create_azure_blob_object_storage(container="artifacts", prefix="tenant-a")
store.put("exports/run-1.zip", file_bytes, content_type="application/zip")
obj = store.get("exports/run-1.zip")
store.delete("exports/run-1.zip")
""",
        "notes": "``azure-storage-blob`` only in ``opens.py``. Default ``object_storage`` slug when ``cloud_platform=azure``.",
    },
    {
        "slug": "gcs",
        "category": "object_storage",
        "category_enum": "OBJECT_STORAGE",
        "slug_enum": "GCS",
        "factory": "create_gcs_object_storage",
        "env": "`INTERGRAX_GCS_BUCKET` (required); optional `INTERGRAX_GCS_PREFIX`, `INTERGRAX_GCS_PROJECT_ID`; GCP ADC or service account",
        "example": """\
store = create_gcs_object_storage(bucket="intergrax-artifacts", prefix="tenant-a")
store.put("reports/summary.pdf", pdf_bytes, content_type="application/pdf")
obj = store.get("reports/summary.pdf")
url = store.presigned_url("reports/summary.pdf", expires_in_seconds=900)
store.close()
""",
        "notes": "``google-cloud-storage`` opened lazily in ``_shared/p2/``. Default ``object_storage`` when ``cloud_platform=gcp``.",
    },
    {
        "slug": "dynamodb",
        "category": "document_store",
        "category_enum": "DOCUMENT_STORE",
        "slug_enum": "DYNAMODB",
        "factory": "create_dynamodb_document_store",
        "env": "`INTERGRAX_DYNAMODB_TABLE`; optional `INTERGRAX_DYNAMODB_REGION`; AWS credential vars",
        "example": """\
from intergrax.integrations.contracts.document_store import DocumentRecord

store = create_dynamodb_document_store(table_name="intergrax-events", region="eu-central-1")
store.put(DocumentRecord(partition_key="tenant-1", row_key="evt-1", data={"status": "ok"}))
doc = store.get("tenant-1", "evt-1")
result = store.query("tenant-1", limit=50, row_key_prefix="2026-")
store.close()
""",
        "notes": "boto3 DynamoDB resource in ``_shared/p2/factories.py``. Default ``document_store`` when ``cloud_platform=aws``.",
    },
    {
        "slug": "sqs",
        "category": "message_bus",
        "category_enum": "MESSAGE_BUS",
        "slug_enum": "SQS",
        "factory": "create_sqs_message_bus",
        "env": "`INTERGRAX_SQS_QUEUE`; optional `INTERGRAX_SQS_REGION`; AWS credential vars",
        "example": """\
from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_sqs_message_bus(queue_name="intergrax-tasks", region="eu-central-1")
handle = bus.enqueue(TaskRequest(tenant_id="t1", run_id="r1", task_name="echo", payload=b"{}", idempotency_key=None))
status = bus.get_status(handle)
""",
        "notes": "``CloudTaskQueue`` over boto3 SQS. Default ``message_bus`` when ``cloud_platform=aws``.",
    },
    {
        "slug": "service_bus",
        "category": "message_bus",
        "category_enum": "MESSAGE_BUS",
        "slug_enum": "SERVICE_BUS",
        "factory": "create_service_bus_message_bus",
        "env": "`INTERGRAX_SERVICE_BUS_CONNECTION_STRING`, `INTERGRAX_SERVICE_BUS_QUEUE`",
        "example": """\
from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_service_bus_message_bus(
    connection_string="Endpoint=sb://....servicebus.windows.net/;...",
    queue_name="intergrax-tasks",
)
handle = bus.enqueue(TaskRequest(tenant_id="t1", run_id="r1", task_name="echo", payload=b"{}", idempotency_key=None))
""",
        "notes": "``azure-servicebus`` opened lazily. Default ``message_bus`` when ``cloud_platform=azure``.",
    },
    {
        "slug": "pubsub",
        "category": "message_bus",
        "category_enum": "MESSAGE_BUS",
        "slug_enum": "PUBSUB",
        "factory": "create_pubsub_message_bus",
        "env": "`INTERGRAX_PUBSUB_PROJECT_ID`, `INTERGRAX_PUBSUB_TOPIC`; GCP ADC or service account",
        "example": """\
from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_pubsub_message_bus(project_id="my-project", topic="intergrax-tasks")
handle = bus.enqueue(TaskRequest(tenant_id="t1", run_id="r1", task_name="echo", payload=b"{}", idempotency_key=None))
""",
        "notes": "``google-cloud-pubsub`` opened lazily. Default ``message_bus`` when ``cloud_platform=gcp``.",
    },
    {
        "slug": "memcached",
        "category": "key_value_cache",
        "category_enum": "KEY_VALUE_CACHE",
        "slug_enum": "MEMCACHED",
        "factory": "create_memcached_key_value_cache",
        "env": "`INTERGRAX_MEMCACHED_HOST` (default `localhost`), `INTERGRAX_MEMCACHED_PORT` (default `11211`)",
        "example": """\
cache = create_memcached_key_value_cache(host="127.0.0.1", port=11211)
cache.set("t1", "session:42", b"payload", ttl_seconds=3600)
value = cache.get("t1", "session:42")
cache.delete("t1", "session:42")
cache.close()
""",
        "notes": "``pymemcache`` opened lazily. Keys are tenant-scoped as ``{tenant_id}:{key}``.",
    },
    {
        "slug": "elasticache",
        "category": "key_value_cache",
        "category_enum": "KEY_VALUE_CACHE",
        "slug_enum": "ELASTICACHE",
        "factory": "create_elasticache_key_value_cache",
        "env": "Same as memcached — point ``INTERGRAX_ELASTICACHE_HOST`` / ``PORT`` at the ElastiCache Redis endpoint",
        "example": """\
cache = create_elasticache_key_value_cache(host="my-cluster.xxxxx.cache.amazonaws.com", port=6379)
cache.set("t1", "lock:graph", b"1", ttl_seconds=60)
""",
        "notes": "Uses the memcached-style duck client adapter. For full Redis semantics prefer ``"redis"`` with the cluster URL.",
    },
    {
        "slug": "oracle",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "ORACLE",
        "factory": "create_oracle_relational_store",
        "env": "`INTERGRAX_ORACLE_DSN` or `INTERGRAX_ORACLE_CONNECTION_STRING`",
        "example": """\
store = create_oracle_relational_store(dsn="user/pass@localhost:1521/ORCL")
store.execute("INSERT INTO items (name) VALUES (:1)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
""",
        "notes": "``oracledb.connect`` opened lazily in ``_shared/p2/factories.py``.",
    },
    {
        "slug": "mssql",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "MSSQL",
        "factory": "create_mssql_relational_store",
        "env": "`INTERGRAX_MSSQL_DSN` or `INTERGRAX_MSSQL_CONNECTION_STRING`",
        "example": """\
store = create_mssql_relational_store(connection_string="Driver={ODBC Driver 18 for SQL Server};Server=...")
store.execute("INSERT INTO items (name) VALUES (?)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
""",
        "notes": "``pyodbc.connect`` opened lazily.",
    },
    {
        "slug": "azure_sql",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "AZURE_SQL",
        "factory": "create_azure_sql_relational_store",
        "env": "`INTERGRAX_AZURE_SQL_CONNECTION_STRING` or DSN; optional `INTERGRAX_AZURE_SQL_SCHEMA`",
        "example": """\
store = create_azure_sql_relational_store(
    connection_string="Driver={ODBC Driver 18 for SQL Server};Server=tcp:....database.windows.net;..."
)
rows = store.fetch_all("SELECT TOP 10 id, name FROM items")
store.close()
""",
        "notes": "Default ``relational_store`` when ``cloud_platform=azure``. ``pyodbc`` opened lazily.",
    },
    {
        "slug": "cloud_sql",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "CLOUD_SQL",
        "factory": "create_cloud_sql_relational_store",
        "env": "`INTERGRAX_CLOUD_SQL_DSN` or connection string components (`HOST`, `USER`, `PASSWORD`, `DATABASE`)",
        "example": """\
store = create_cloud_sql_relational_store(dsn="host=127.0.0.1 user=app password=secret dbname=intergrax")
store.execute("INSERT INTO items (name) VALUES (%s)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
""",
        "notes": "Default ``relational_store`` when ``cloud_platform=gcp``. ``pg8000`` opened lazily.",
    },
    {
        "slug": "email_smtp",
        "category": "notification_channel",
        "category_enum": "NOTIFICATION_CHANNEL",
        "slug_enum": "EMAIL_SMTP",
        "factory": "create_email_smtp_notification_channel",
        "env": "`INTERGRAX_EMAIL_SMTP_HOST`, `INTERGRAX_EMAIL_SMTP_PORT` (default `587`); optional `USER`, `PASSWORD`, `FROM`",
        "example": """\
import asyncio
from intergrax.runtime.notifications.models import NotificationMessage

channel = create_email_smtp_notification_channel(
    smtp_host="smtp.example.com",
    smtp_port=587,
    user="bot@example.com",
    password="...",
    from_address="noreply@example.com",
)
asyncio.run(channel.notify(NotificationMessage(
    tenant_id="t1",
    channel="#alerts",
    task_id="task-1",
    subject="HITL approval required",
    body="Please review run r-42.",
    metadata={"to": "ops@example.com"},
)))
""",
        "notes": "stdlib ``smtplib`` in factory open path. Implements ``NotificationAdapter`` (async ``notify``).",
    },
    {
        "slug": "otel",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "OTEL",
        "factory": "create_otel_observability_backend",
        "env": "`INTERGRAX_OTEL_ENDPOINT` (default `http://localhost:4318`), `INTERGRAX_OTEL_SERVICE_NAME`",
        "example": """\
obs = create_otel_observability_backend(endpoint="http://otel-collector:4318", service_name="intergrax-nexus")
instant = obs.query_instant("intergrax_tasks_total")
range_result = obs.query_range("intergrax_tasks_total", start=1710000000, end=1710003600, step="15s")
""",
        "notes": "Beta facade over an OTLP-oriented exporter. Inject ``exporter=`` in tests; production wiring may evolve.",
    },
    {
        "slug": "github",
        "category": "issue_tracker",
        "category_enum": "ISSUE_TRACKER",
        "slug_enum": "GITHUB",
        "factory": "create_github_issue_tracker",
        "env": "`INTERGRAX_GITHUB_TOKEN`; optional `INTERGRAX_GITHUB_ORG`, `INTERGRAX_GITHUB_REPO`, `INTERGRAX_GITHUB_URL`",
        "example": """\
tracker = create_github_issue_tracker(token="ghp_...", org="acme", repo="platform")
issue = tracker.get_issue("42")
tracker.add_comment("42", "Agent: root cause identified.")
results = tracker.search_issues("is:open label:agent", limit=20)
""",
        "notes": "httpx REST client opened lazily. ``search_issues`` accepts GitHub search query syntax.",
    },
    {
        "slug": "linear",
        "category": "issue_tracker",
        "category_enum": "ISSUE_TRACKER",
        "slug_enum": "LINEAR",
        "factory": "create_linear_issue_tracker",
        "env": "`INTERGRAX_LINEAR_API_KEY`; optional `INTERGRAX_LINEAR_URL`",
        "example": """\
tracker = create_linear_issue_tracker(api_key="lin_api_...")
issue = tracker.get_issue("ENG-123")
tracker.add_comment("ENG-123", "Automated triage complete.")
results = tracker.search_issues("priority:1 state:open", limit=20)
""",
        "notes": "httpx REST client opened lazily.",
    },
    {
        "slug": "azure_devops",
        "category": "issue_tracker",
        "category_enum": "ISSUE_TRACKER",
        "slug_enum": "AZURE_DEVOPS",
        "factory": "create_azure_devops_issue_tracker",
        "env": "`INTERGRAX_AZURE_DEVOPS_TOKEN`; optional `INTERGRAX_AZURE_DEVOPS_ORG`, `INTERGRAX_AZURE_DEVOPS_REPO`, `INTERGRAX_AZURE_DEVOPS_URL`",
        "example": """\
tracker = create_azure_devops_issue_tracker(token="...", org="acme", repo="Platform")
issue = tracker.get_issue("12345")
tracker.add_comment("12345", "Agent update posted.")
results = tracker.search_issues("[System.State] = 'Active'", limit=20)
""",
        "notes": "REST work-item facade; WIQL passed via ``search_issues``.",
    },
    {
        "slug": "notion",
        "category": "wiki_knowledge",
        "category_enum": "WIKI_KNOWLEDGE",
        "slug_enum": "NOTION",
        "factory": "create_notion_wiki_knowledge",
        "env": "`INTERGRAX_NOTION_API_KEY` (Bearer token); optional `INTERGRAX_NOTION_URL`",
        "example": """\
wiki = create_notion_wiki_knowledge(api_key="secret_...")
page = wiki.get_page("page-uuid")
results = wiki.search_pages("deployment runbook", limit=10)
""",
        "notes": "Notion REST API via httpx. Complements ``confluence`` for mixed knowledge bases.",
    },
    {
        "slug": "sharepoint",
        "category": "wiki_knowledge",
        "category_enum": "WIKI_KNOWLEDGE",
        "slug_enum": "SHAREPOINT",
        "factory": "create_sharepoint_wiki_knowledge",
        "env": "`INTERGRAX_SHAREPOINT_TOKEN`; optional `INTERGRAX_SHAREPOINT_SITE_URL`, `INTERGRAX_SHAREPOINT_URL`",
        "example": """\
wiki = create_sharepoint_wiki_knowledge(token="...", site_url="https://contoso.sharepoint.com/sites/docs")
page = wiki.get_page("page-id")
results = wiki.search_pages("incident response", limit=10)
""",
        "notes": "Microsoft Graph / SharePoint REST via httpx.",
    },
    {
        "slug": "google_workspace",
        "category": "collaboration_suite",
        "category_enum": "COLLABORATION_SUITE",
        "slug_enum": "GOOGLE_WORKSPACE",
        "factory": "create_google_workspace_collaboration_suite",
        "env": "OAuth bearer via `INTERGRAX_GOOGLE_WORKSPACE_TOKEN` or service account; optional `INTERGRAX_GOOGLE_WORKSPACE_URL`",
        "example": """\
suite = create_google_workspace_collaboration_suite(token="ya29....")
user = suite.get_user("user@example.com")
messages = suite.list_messages("user@example.com", folder="inbox", limit=10)
suite.send_mail("user@example.com", subject="Report", body="...", to=["ops@example.com"])
events = suite.list_calendar_events("primary", start="2026-05-01T00:00:00Z", end="2026-05-31T23:59:59Z")
""",
        "notes": "Gmail / Calendar / Directory REST. Google-tenant parity with ``ms365_graph``.",
    },
    {
        "slug": "brave",
        "category": "search_provider",
        "category_enum": "SEARCH_PROVIDER",
        "slug_enum": "BRAVE",
        "factory": "create_brave_search_provider",
        "env": "`INTERGRAX_BRAVE_API_KEY`",
        "example": """\
search = create_brave_search_provider(api_key="BSA...")
hits = search.search("Intergrax agent orchestration", limit=5)
for hit in hits:
    print(hit.rank, hit.title, hit.url)
""",
        "notes": "Brave Web Search API via httpx. Hit normalization in ``_shared/rest_search.py``.",
    },
    {
        "slug": "serpapi",
        "category": "search_provider",
        "category_enum": "SEARCH_PROVIDER",
        "slug_enum": "SERPAPI",
        "factory": "create_serpapi_search_provider",
        "env": "`INTERGRAX_SERPAPI_API_KEY`",
        "example": """\
search = create_serpapi_search_provider(api_key="...")
hits = search.search("enterprise AI agents", limit=5)
""",
        "notes": "SerpAPI JSON API via httpx.",
    },
    {
        "slug": "playwright",
        "category": "browser_automation",
        "category_enum": "BROWSER_AUTOMATION",
        "slug_enum": "PLAYWRIGHT",
        "factory": "create_playwright_browser_automation",
        "env": "Optional overrides: ``headless=True``, ``timeout_ms=30000`` (no required env vars)",
        "example": """\
browser = create_playwright_browser_automation(headless=True, timeout_ms=30000)
page = browser.fetch_page("https://example.com/dashboard", wait_until="networkidle")
print(page.title, page.text[:200])
browser.close()
""",
        "notes": "``playwright`` Chromium launch opened lazily. Use for JS-heavy pages; prefer ``search_provider`` for simple research.",
    },
    {
        "slug": "tavily",
        "category": "search_provider",
        "category_enum": "SEARCH_PROVIDER",
        "slug_enum": "TAVILY",
        "factory": "create_tavily_search_provider",
        "env": "`INTERGRAX_TAVILY_API_KEY`, optional `INTERGRAX_TAVILY_URL`",
        "example": "search = create_tavily_search_provider(api_key=\"tvly-...\")\nhits = search.search(\"agent harness\", limit=5)",
        "notes": "Agent-native research API (Phase M.7). Thin shell → ``_shared/p3/factories``.",
    },
    {
        "slug": "exa",
        "category": "search_provider",
        "category_enum": "SEARCH_PROVIDER",
        "slug_enum": "EXA",
        "factory": "create_exa_search_provider",
        "env": "`INTERGRAX_EXA_API_KEY`",
        "example": "search = create_exa_search_provider(api_key=\"...\")\nhits = search.search(\"RAG patterns\", limit=5)",
        "notes": "Neural / semantic web search (Phase M.7).",
    },
    {
        "slug": "weaviate",
        "category": "vector_store",
        "category_enum": "VECTOR_STORE",
        "slug_enum": "WEAVIATE",
        "factory": "create_weaviate_vector_store",
        "env": "`INTERGRAX_WEAVIATE_URL`, `INTERGRAX_WEAVIATE_API_KEY`, `INTERGRAX_WEAVIATE_COLLECTION`",
        "example": "store = create_weaviate_vector_store(url=\"https://...\", collection=\"docs\")",
        "notes": "Requires ``weaviate-client`` at runtime. Catalog bridge via ``VectorStoreBridge``.",
    },
    {
        "slug": "milvus",
        "category": "vector_store",
        "category_enum": "VECTOR_STORE",
        "slug_enum": "MILVUS",
        "factory": "create_milvus_vector_store",
        "env": "`INTERGRAX_MILVUS_URL`, optional `INTERGRAX_MILVUS_API_KEY`",
        "example": "store = create_milvus_vector_store(url=\"http://localhost:19530\")",
        "notes": "Requires ``pymilvus`` at runtime.",
    },
    {
        "slug": "inmemory",
        "category": "vector_store",
        "category_enum": "VECTOR_STORE",
        "slug_enum": "INMEMORY",
        "factory": "create_inmemory_vector_store",
        "env": "Optional `INTERGRAX_INMEMORY_TENANT_ID` (default ``default``)",
        "example": "store = create_inmemory_vector_store(tenant_id=\"lab\")",
        "notes": "Delegates to ``intergrax.rag.vectorstore.providers.inmemory_vectorstore`` — lab / unit tests.",
    },
    {
        "slug": "vault",
        "category": "secrets_store",
        "category_enum": "SECRETS_STORE",
        "slug_enum": "VAULT",
        "factory": "create_vault_secrets_store",
        "env": "`INTERGRAX_VAULT_ADDR`, `INTERGRAX_VAULT_TOKEN`, `INTERGRAX_VAULT_MOUNT`",
        "example": "secrets = create_vault_secrets_store(addr=\"http://127.0.0.1:8200\", token=\"...\")\nsecrets.put_secret(\"tenant/openai\", \"sk-...\")",
        "notes": "HashiCorp Vault KV v2. Requires ``hvac``. New category ``secrets_store`` (§5.2.4).",
    },
    {
        "slug": "langfuse",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "LANGFUSE",
        "factory": "create_langfuse_observability_backend",
        "env": "`INTERGRAX_LANGFUSE_URL`, `INTERGRAX_LANGFUSE_API_KEY`",
        "example": "obs = create_langfuse_observability_backend(base_url=\"https://cloud.langfuse.com\", api_key=\"...\")",
        "notes": "LLM/agent trace metrics via HTTP (PromQL-shaped facade).",
    },
    {
        "slug": "datadog",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "DATADOG",
        "factory": "create_datadog_observability_backend",
        "env": "`INTERGRAX_DATADOG_API_KEY`, optional `INTERGRAX_DATADOG_URL`",
        "example": "obs = create_datadog_observability_backend(api_key=\"...\")",
        "notes": "Datadog metrics API (instant/range queries).",
    },
    {
        "slug": "clickhouse",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "CLICKHOUSE",
        "factory": "create_clickhouse_observability_backend",
        "env": "`INTERGRAX_CLICKHOUSE_URL`",
        "example": "obs = create_clickhouse_observability_backend(base_url=\"http://localhost:8123\")",
        "notes": "High-volume agent event analytics (HTTP SQL facade).",
    },
    {
        "slug": "sentry",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "SENTRY",
        "factory": "create_sentry_observability_backend",
        "env": "`INTERGRAX_SENTRY_DSN`, `INTERGRAX_SENTRY_ORG`, `INTERGRAX_SENTRY_AUTH_TOKEN`, optional `INTERGRAX_SENTRY_PROJECT`, `INTERGRAX_SENTRY_ENVIRONMENT`",
        "example": """\
obs = create_sentry_observability_backend(
    dsn="https://...@sentry.io/...",
    org="my-org",
    auth_token="sntrys_...",
)
obs.capture_message("agent run failed", level="error")
count = obs.query_instant("is:unresolved").series[0].points[0].value
""",
        "notes": "Error tracking + issue stats. ``sentry-sdk`` for capture; REST API for issue counts. Complements ``otel``/``langfuse``.",
    },
    {
        "slug": "temporal",
        "category": "message_bus",
        "category_enum": "MESSAGE_BUS",
        "slug_enum": "TEMPORAL",
        "factory": "create_temporal_message_bus",
        "env": "`INTERGRAX_TEMPORAL_CONNECTION_STRING`",
        "example": "bus = create_temporal_message_bus(connection_string=\"localhost:7233\")",
        "notes": "Durable workflow enqueue facade (``temporalio`` optional at runtime).",
    },
    {
        "slug": "nats",
        "category": "message_bus",
        "category_enum": "MESSAGE_BUS",
        "slug_enum": "NATS",
        "factory": "create_nats_message_bus",
        "env": "`INTERGRAX_NATS_CONNECTION_STRING` (default ``nats://localhost:4222``)",
        "example": "bus = create_nats_message_bus(connection_string=\"nats://localhost:4222\")",
        "notes": "Lightweight event bus facade.",
    },
    {
        "slug": "neo4j",
        "category": "graph_store",
        "category_enum": "GRAPH_STORE",
        "slug_enum": "NEO4J",
        "factory": "create_neo4j_graph_store",
        "env": "`INTERGRAX_NEO4J_URL`, `INTERGRAX_NEO4J_USER`, `INTERGRAX_NEO4J_PASSWORD`",
        "example": "graph = create_neo4j_graph_store(base_url=\"bolt://localhost:7687\", user=\"neo4j\", password=\"...\")",
        "notes": "Agent memory / tool graphs. Requires ``neo4j`` driver. New category ``graph_store``.",
    },
    {
        "slug": "snowflake",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "SNOWFLAKE",
        "factory": "create_snowflake_relational_store",
        "env": "`INTERGRAX_SNOWFLAKE_DSN` or connection components",
        "example": "store = create_snowflake_relational_store(dsn=\"snowflake://...\")",
        "notes": "SQL facade via ``psycopg``-compatible DSN.",
    },
    {
        "slug": "supabase",
        "category": "relational_store",
        "category_enum": "RELATIONAL_STORE",
        "slug_enum": "SUPABASE",
        "factory": "create_supabase_relational_store",
        "env": "`INTERGRAX_SUPABASE_DSN` (Postgres connection string)",
        "example": "store = create_supabase_relational_store(dsn=\"postgresql://...\")",
        "notes": "Postgres-backed product prototypes.",
    },
    {
        "slug": "minio",
        "category": "object_storage",
        "category_enum": "OBJECT_STORAGE",
        "slug_enum": "MINIO",
        "factory": "create_minio_object_storage",
        "env": "`INTERGRAX_MINIO_ENDPOINT`, `INTERGRAX_MINIO_ACCESS_KEY`, `INTERGRAX_MINIO_SECRET_KEY`, `INTERGRAX_MINIO_BUCKET`",
        "example": "blobs = create_minio_object_storage(endpoint=\"http://localhost:9000\", bucket=\"artifacts\")",
        "notes": "S3-compatible self-hosted storage (boto3).",
    },
    {
        "slug": "filesystem",
        "category": "object_storage",
        "category_enum": "OBJECT_STORAGE",
        "slug_enum": "FILESYSTEM",
        "factory": "create_filesystem_object_storage",
        "env": "`INTERGRAX_FILESYSTEM_ROOT_DIR` (default ``build/artifacts``)",
        "example": "blobs = create_filesystem_object_storage(root_dir=\"build/lab-artifacts\")",
        "notes": "Local artifact store for CI/lab — no cloud SDK.",
    },
    {
        "slug": "discord",
        "category": "notification_channel",
        "category_enum": "NOTIFICATION_CHANNEL",
        "slug_enum": "DISCORD",
        "factory": "create_discord_notification_channel",
        "env": "`INTERGRAX_DISCORD_URL` (webhook URL)",
        "example": "notify = create_discord_notification_channel(base_url=\"https://discord.com/api/webhooks/...\")",
        "notes": "Community ops notifications via webhook POST.",
    },
    {
        "slug": "twilio",
        "category": "notification_channel",
        "category_enum": "NOTIFICATION_CHANNEL",
        "slug_enum": "TWILIO",
        "factory": "create_twilio_notification_channel",
        "env": "`INTERGRAX_TWILIO_ORG` (account SID), `INTERGRAX_TWILIO_USER`/`API_KEY`, `INTERGRAX_TWILIO_PASSWORD`/`TOKEN`, `INTERGRAX_TWILIO_SITE_URL` (from number)",
        "example": "sms = create_twilio_notification_channel(org=\"AC...\", site_url=\"+1...\")",
        "notes": "SMS HITL — set ``metadata['to']`` on ``NotificationMessage``.",
    },
    {
        "slug": "firecrawl",
        "category": "browser_automation",
        "category_enum": "BROWSER_AUTOMATION",
        "slug_enum": "FIRECRAWL",
        "factory": "create_firecrawl_browser_automation",
        "env": "`INTERGRAX_FIRECRAWL_API_KEY`",
        "example": "crawl = create_firecrawl_browser_automation(api_key=\"fc-...\")\npage = crawl.fetch_page(\"https://docs.example.com\")",
        "notes": "Structured crawl API — alternative to raw Playwright.",
    },
    {
        "slug": "selenium",
        "category": "browser_automation",
        "category_enum": "BROWSER_AUTOMATION",
        "slug_enum": "SELENIUM",
        "factory": "create_selenium_browser_automation",
        "env": "`INTERGRAX_SELENIUM_DRIVER_URL` (optional remote grid), `INTERGRAX_SELENIUM_BROWSER`",
        "example": "browser = create_selenium_browser_automation(headless=True)\npage = browser.fetch_page(\"https://legacy.example.com\")",
        "notes": "Legacy browser stacks; requires ``selenium`` package.",
    },
    {
        "slug": "langsmith",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "LANGSMITH",
        "factory": "create_langsmith_observability_backend",
        "env": "`INTERGRAX_LANGSMITH_API_KEY`, `INTERGRAX_LANGSMITH_URL`",
        "example": "obs = create_langsmith_observability_backend(api_key=\"lsv2_...\")\ntraces = obs.query_traces(limit=10)",
        "notes": "LangChain trace export (Phase M.8). Thin shell → ``_shared/p4/factories``.",
    },
    {
        "slug": "helicone",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "HELICONE",
        "factory": "create_helicone_observability_backend",
        "env": "`INTERGRAX_HELICONE_API_KEY`",
        "example": "obs = create_helicone_observability_backend(api_key=\"sk-...\")",
        "notes": "LLM cost/latency proxy observability.",
    },
    {
        "slug": "posthog",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "POSTHOG",
        "factory": "create_posthog_observability_backend",
        "env": "`INTERGRAX_POSTHOG_API_KEY`, `INTERGRAX_POSTHOG_URL`",
        "example": "obs = create_posthog_observability_backend(api_key=\"phc_...\")",
        "notes": "Product analytics + event metrics facade.",
    },
    {
        "slug": "braintrust",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "BRAINTRUST",
        "factory": "create_braintrust_observability_backend",
        "env": "`INTERGRAX_BRAINTRUST_API_KEY`",
        "example": "obs = create_braintrust_observability_backend(api_key=\"...\")",
        "notes": "Evals and regression logs for agent prompts.",
    },
    {
        "slug": "signoz",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "SIGNOZ",
        "factory": "create_signoz_observability_backend",
        "env": "`INTERGRAX_SIGNOZ_URL`",
        "example": "obs = create_signoz_observability_backend(base_url=\"http://localhost:8080\")",
        "notes": "OTEL-native APM (self-hosted).",
    },
    {
        "slug": "honeycomb",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "HONEYCOMB",
        "factory": "create_honeycomb_observability_backend",
        "env": "`INTERGRAX_HONEYCOMB_API_KEY`",
        "example": "obs = create_honeycomb_observability_backend(api_key=\"...\")",
        "notes": "High-cardinality trace/metrics queries.",
    },
    {
        "slug": "arize",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "ARIZE",
        "factory": "create_arize_observability_backend",
        "env": "`INTERGRAX_ARIZE_API_KEY`, `INTERGRAX_ARIZE_URL`",
        "example": "obs = create_arize_observability_backend(api_key=\"...\")",
        "notes": "ML/RAG monitoring and drift.",
    },
    {
        "slug": "phoenix",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "PHOENIX",
        "factory": "create_phoenix_observability_backend",
        "env": "`INTERGRAX_PHOENIX_URL` (default ``http://localhost:6006``)",
        "example": "obs = create_phoenix_observability_backend(base_url=\"http://localhost:6006\")",
        "notes": "Arize Phoenix — local LLM trace UI.",
    },
    {
        "slug": "wandb",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "WANDB",
        "factory": "create_wandb_observability_backend",
        "env": "`INTERGRAX_WANDB_API_KEY`, `INTERGRAX_WANDB_URL`",
        "example": "obs = create_wandb_observability_backend(api_key=\"...\")",
        "notes": "Experiment tracking metrics facade.",
    },
    {
        "slug": "opensearch",
        "category": "observability_backend",
        "category_enum": "OBSERVABILITY_BACKEND",
        "slug_enum": "OPENSEARCH",
        "factory": "create_opensearch_observability_backend",
        "env": "`INTERGRAX_OPENSEARCH_URL`, `INTERGRAX_OPENSEARCH_INDEX`",
        "example": "obs = create_opensearch_observability_backend(base_url=\"http://localhost:9200\")",
        "notes": "Elasticsearch-compatible log/metrics search (reuses ES client).",
    },
    {
        "slug": "pagerduty",
        "category": "notification_channel",
        "category_enum": "NOTIFICATION_CHANNEL",
        "slug_enum": "PAGERDUTY",
        "factory": "create_pagerduty_notification_channel",
        "env": "`INTERGRAX_PAGERDUTY_API_KEY` (routing key)",
        "example": "pd = create_pagerduty_notification_channel(api_key=\"routing-key\")",
        "notes": "On-call escalation via Events API v2.",
    },
    {
        "slug": "opsgenie",
        "category": "notification_channel",
        "category_enum": "NOTIFICATION_CHANNEL",
        "slug_enum": "OPSGENIE",
        "factory": "create_opsgenie_notification_channel",
        "env": "`INTERGRAX_OPSGENIE_API_KEY`",
        "example": "og = create_opsgenie_notification_channel(api_key=\"...\")",
        "notes": "Alertmanager-style HITL escalation.",
    },
    {
        "slug": "gitlab",
        "category": "issue_tracker",
        "category_enum": "ISSUE_TRACKER",
        "slug_enum": "GITLAB",
        "factory": "create_gitlab_issue_tracker",
        "env": "`INTERGRAX_GITLAB_URL`, `INTERGRAX_GITLAB_TOKEN`, `INTERGRAX_GITLAB_REPO` (project id/path)",
        "example": "tracker = create_gitlab_issue_tracker(base_url=\"https://gitlab.com/api/v4\", repo=\"group/project\")",
        "notes": "GitLab REST issue tracker.",
    },
    {
        "slug": "vespa",
        "category": "vector_store",
        "category_enum": "VECTOR_STORE",
        "slug_enum": "VESPA",
        "factory": "create_vespa_vector_store",
        "env": "`INTERGRAX_VESPA_URL`, `INTERGRAX_VESPA_COLLECTION`",
        "example": "store = create_vespa_vector_store(url=\"http://localhost:8080\", collection=\"docs\")",
        "notes": "Vespa vector search catalog bridge.",
    },
]


def profile_field(category: str) -> str:
    if "+" in category:
        return category.split("+")[0].strip()
    return category.strip()


def render(provider: dict[str, str]) -> str:
    slug = provider["slug"]
    pf = profile_field(provider["category"])
    category_folder = pf
    header = COMMON_HEADER.format(
        profile_field=pf,
        slug_enum=provider["slug_enum"],
        category_enum=provider["category_enum"],
        category=category_folder,
        slug=slug,
        factory=provider["factory"],
    )
    return f"""# `{slug}` integration — usage

**Category:** ``{provider['category']}``  
**Catalog factory:** ``{provider['factory']}()``

{header}

## Environment variables

{provider['env']}

## Example

```python
from intergrax.integrations.providers.{category_folder}.{slug}.bundle import {provider['factory']}

{provider['example'].rstrip()}
```

## Notes

{provider['notes']}
"""


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from intergrax.integrations.providers.layout import SLUG_CATEGORY

    for provider in PROVIDERS:
        slug = provider["slug"]
        category = profile_field(provider.get("category") or SLUG_CATEGORY[slug])
        path = PROVIDERS_DIR / category / slug / "USAGE.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(provider), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
