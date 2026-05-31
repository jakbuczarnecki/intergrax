#!/usr/bin/env python3
"""Generate thin provider shells pointing to _shared.p3.factories."""
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
    ("tavily", "SEARCH_PROVIDER", "TAVILY", "create_tavily_search_provider", "INTERGRAX_TAVILY"),
    ("exa", "SEARCH_PROVIDER", "EXA", "create_exa_search_provider", "INTERGRAX_EXA"),
    ("weaviate", "VECTOR_STORE", "WEAVIATE", "create_weaviate_vector_store", "INTERGRAX_WEAVIATE"),
    ("milvus", "VECTOR_STORE", "MILVUS", "create_milvus_vector_store", "INTERGRAX_MILVUS"),
    ("inmemory", "VECTOR_STORE", "INMEMORY", "create_inmemory_vector_store", "INTERGRAX_INMEMORY"),
    ("vault", "SECRETS_STORE", "VAULT", "create_vault_secrets_store", "INTERGRAX_VAULT"),
    ("langfuse", "OBSERVABILITY_BACKEND", "LANGFUSE", "create_langfuse_observability_backend", "INTERGRAX_LANGFUSE"),
    ("datadog", "OBSERVABILITY_BACKEND", "DATADOG", "create_datadog_observability_backend", "INTERGRAX_DATADOG"),
    ("clickhouse", "OBSERVABILITY_BACKEND", "CLICKHOUSE", "create_clickhouse_observability_backend", "INTERGRAX_CLICKHOUSE"),
    ("temporal", "MESSAGE_BUS", "TEMPORAL", "create_temporal_message_bus", "INTERGRAX_TEMPORAL"),
    ("nats", "MESSAGE_BUS", "NATS", "create_nats_message_bus", "INTERGRAX_NATS"),
    ("neo4j", "GRAPH_STORE", "NEO4J", "create_neo4j_graph_store", "INTERGRAX_NEO4J"),
    ("snowflake", "RELATIONAL_STORE", "SNOWFLAKE", "create_snowflake_relational_store", "INTERGRAX_SNOWFLAKE"),
    ("supabase", "RELATIONAL_STORE", "SUPABASE", "create_supabase_relational_store", "INTERGRAX_SUPABASE"),
    ("minio", "OBJECT_STORAGE", "MINIO", "create_minio_object_storage", "INTERGRAX_MINIO"),
    ("filesystem", "OBJECT_STORAGE", "FILESYSTEM", "create_filesystem_object_storage", "INTERGRAX_FILESYSTEM"),
    ("discord", "NOTIFICATION_CHANNEL", "DISCORD", "create_discord_notification_channel", "INTERGRAX_DISCORD"),
    ("twilio", "NOTIFICATION_CHANNEL", "TWILIO", "create_twilio_notification_channel", "INTERGRAX_TWILIO"),
    ("firecrawl", "BROWSER_AUTOMATION", "FIRECRAWL", "create_firecrawl_browser_automation", "INTERGRAX_FIRECRAWL"),
    ("selenium", "BROWSER_AUTOMATION", "SELENIUM", "create_selenium_browser_automation", "INTERGRAX_SELENIUM"),
]


def _category_folder(cat_enum: str) -> str:
    return IntegrationCategory[cat_enum].value


for slug, cat_enum, enum, factory, env in SPECS:
    category = _category_folder(cat_enum)
    assert SLUG_CATEGORY.get(slug) == category, f"{slug}: layout mismatch {SLUG_CATEGORY.get(slug)} != {category}"
    pkg = PROVIDERS / category / slug
    pkg.mkdir(parents=True, exist_ok=True)
    import_base = f"intergrax.integrations.providers.{category}.{slug}"
    (pkg / "register.py").write_text(
        H
        + f'"""Register {slug}."""\n\nfrom __future__ import annotations\n\n'
        + "from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus\n"
        + f"from {import_base}.bundle import {factory}\n"
        + "from intergrax.integrations.registry.catalog import register_integration\n"
        + "from intergrax.integrations.registry.slugs import IntegrationSlug\n\n"
        + f"def register_{slug}_integration(*, override: bool = False) -> None:\n"
        + "    register_integration(\n"
        + "        IntegrationEntry(\n"
        + f"            slug=IntegrationSlug.{enum}.value,\n"
        + f"            categories=(IntegrationCategory.{cat_enum},),\n"
        + f"            factory={factory},\n"
        + "            status=IntegrationStatus.BETA,\n"
        + f'            env_prefix="{env}",\n'
        + f'            description="{slug} integration (Phase M.7)",\n'
        + "        ),\n"
        + "        override=override,\n"
        + "    )\n",
        encoding="utf-8",
    )
    (pkg / "bundle.py").write_text(
        H + f"from intergrax.integrations._shared.p3.factories import {factory}\n\n__all__ = [\"{factory}\"]\n",
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
print(f"generated {len(SPECS)} provider shells")
