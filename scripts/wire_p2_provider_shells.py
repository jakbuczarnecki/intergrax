#!/usr/bin/env python3
"""Generate thin provider shells pointing to _shared.p2.factories."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.layout import SLUG_CATEGORY

PROVIDERS = ROOT / "intergrax" / "integrations" / "providers"
H = "# © Artur Czarnecki. All rights reserved.\n# Intergrax framework – proprietary and confidential.\n\n"

SPECS = [
    ("gcs", "OBJECT_STORAGE", "GCS", "create_gcs_object_storage", "INTERGRAX_GCS"),
    ("dynamodb", "DOCUMENT_STORE", "DYNAMODB", "create_dynamodb_document_store", "INTERGRAX_DYNAMODB"),
    ("sqs", "MESSAGE_BUS", "SQS", "create_sqs_message_bus", "INTERGRAX_SQS"),
    ("service_bus", "MESSAGE_BUS", "SERVICE_BUS", "create_service_bus_message_bus", "INTERGRAX_SERVICE_BUS"),
    ("pubsub", "MESSAGE_BUS", "PUBSUB", "create_pubsub_message_bus", "INTERGRAX_PUBSUB"),
    ("memcached", "KEY_VALUE_CACHE", "MEMCACHED", "create_memcached_key_value_cache", "INTERGRAX_MEMCACHED"),
    ("elasticache", "KEY_VALUE_CACHE", "ELASTICACHE", "create_elasticache_key_value_cache", "INTERGRAX_ELASTICACHE"),
    ("oracle", "RELATIONAL_STORE", "ORACLE", "create_oracle_relational_store", "INTERGRAX_ORACLE"),
    ("mssql", "RELATIONAL_STORE", "MSSQL", "create_mssql_relational_store", "INTERGRAX_MSSQL"),
    ("azure_sql", "RELATIONAL_STORE", "AZURE_SQL", "create_azure_sql_relational_store", "INTERGRAX_AZURE_SQL"),
    ("cloud_sql", "RELATIONAL_STORE", "CLOUD_SQL", "create_cloud_sql_relational_store", "INTERGRAX_CLOUD_SQL"),
    ("email_smtp", "NOTIFICATION_CHANNEL", "EMAIL_SMTP", "create_email_smtp_notification_channel", "INTERGRAX_EMAIL_SMTP"),
    ("otel", "OBSERVABILITY_BACKEND", "OTEL", "create_otel_observability_backend", "INTERGRAX_OTEL"),
    ("github", "ISSUE_TRACKER", "GITHUB", "create_github_issue_tracker", "INTERGRAX_GITHUB"),
    ("linear", "ISSUE_TRACKER", "LINEAR", "create_linear_issue_tracker", "INTERGRAX_LINEAR"),
    ("azure_devops", "ISSUE_TRACKER", "AZURE_DEVOPS", "create_azure_devops_issue_tracker", "INTERGRAX_AZURE_DEVOPS"),
    ("notion", "WIKI_KNOWLEDGE", "NOTION", "create_notion_wiki_knowledge", "INTERGRAX_NOTION"),
    ("sharepoint", "WIKI_KNOWLEDGE", "SHAREPOINT", "create_sharepoint_wiki_knowledge", "INTERGRAX_SHAREPOINT"),
    ("google_workspace", "COLLABORATION_SUITE", "GOOGLE_WORKSPACE", "create_google_workspace_collaboration_suite", "INTERGRAX_GOOGLE_WORKSPACE"),
    ("brave", "SEARCH_PROVIDER", "BRAVE", "create_brave_search_provider", "INTERGRAX_BRAVE"),
    ("serpapi", "SEARCH_PROVIDER", "SERPAPI", "create_serpapi_search_provider", "INTERGRAX_SERPAPI"),
    ("playwright", "BROWSER_AUTOMATION", "PLAYWRIGHT", "create_playwright_browser_automation", "INTERGRAX_PLAYWRIGHT"),
]

SKIP = {"azure_blob"}  # hand-written full package under object_storage/


def _category_folder(cat_enum: str) -> str:
    return IntegrationCategory[cat_enum].value


for slug, cat_enum, enum, factory, env in SPECS:
    if slug in SKIP:
        continue
    category = _category_folder(cat_enum)
    assert SLUG_CATEGORY.get(slug) == category, f"{slug}: layout mismatch"
    pkg = PROVIDERS / category / slug
    pkg.mkdir(parents=True, exist_ok=True)
    import_base = f"intergrax.integrations.providers.{category}.{slug}"
    (pkg / "register.py").write_text(
        H
        + f'"""Register {slug}."""\n\nfrom __future__ import annotations\n\n'
        + "from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus\n"
        + f"from {import_base}.bundle import {factory}\n"
        + "from intergrax.integrations.registry.catalog import register_integration\n"
        + f"def register_{slug}_integration(*, override: bool = False) -> None:\n"
        + "    register_integration(\n"
        + "        IntegrationEntry(\n"
        + f'            slug="{slug}",\n'
        + f"            categories=(IntegrationCategory.{cat_enum},),\n"
        + f"            factory={factory},\n"
        + "            status=IntegrationStatus.BETA,\n"
        + f'            env_prefix="{env}",\n'
        + f'            description="{slug} integration (Phase M.6 P2/P3)",\n'
        + "        ),\n"
        + "        override=override,\n"
        + "    )\n",
        encoding="utf-8",
    )
    (pkg / "bundle.py").write_text(
        H + f"from intergrax.integrations._shared.p2.factories import {factory}\n\n__all__ = [\"{factory}\"]\n",
        encoding="utf-8",
    )
    (pkg / "__init__.py").write_text(
        H
        + f'__all__ = ["{factory}", "register_{slug}_integration"]\n\n'
        + "def __getattr__(name: str):\n"
        + f'    if name == "register_{slug}_integration":\n'
        + f"        from {import_base}.register import register_{slug}_integration\n"
        + f"        return register_{slug}_integration\n"
        + f'    if name == "{factory}":\n'
        + f"        from {import_base}.bundle import {factory}\n"
        + f"        return {factory}\n"
        + "    raise AttributeError(name)\n",
        encoding="utf-8",
    )
print(f"generated {len(SPECS) - len(SKIP)} provider shells")
