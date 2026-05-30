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
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile({profile_field}=IntegrationSlug.{slug_enum})
backend = profile.resolve(IntegrationCategory.{category_enum})
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.{slug}.bundle import {factory}

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
from intergrax.integrations.providers.slack.bundle import create_slack_interaction_surface
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
]


def profile_field(category: str) -> str:
    if "+" in category:
        return category.split("+")[0].strip()
    return category.strip()


def render(provider: dict[str, str]) -> str:
    slug = provider["slug"]
    pf = profile_field(provider["category"])
    header = COMMON_HEADER.format(
        profile_field=pf,
        slug_enum=provider["slug_enum"],
        category_enum=provider["category_enum"],
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
from intergrax.integrations.providers.{slug}.bundle import {provider['factory']}

{provider['example'].rstrip()}
```

## Notes

{provider['notes']}
"""


def main() -> None:
    for provider in PROVIDERS:
        path = PROVIDERS_DIR / provider["slug"] / "USAGE.md"
        path.write_text(render(provider), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
