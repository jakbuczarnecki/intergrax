#!/usr/bin/env python3
"""
Generate thin provider shells pointing to _shared.p3.factories.

Legacy shell generator for unmigrated providers. When ``integration.py`` exists
in a provider package, all canonical files are preserved (contract-aware mode).
Do not use this script to regenerate migrated contract-based providers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.layout import SLUG_CATEGORY
from scripts.maintenance._provider_shell_contract import write_provider_file_if_allowed

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
    ("sentry", "OBSERVABILITY_BACKEND", "SENTRY", "create_sentry_observability_backend", "INTERGRAX_SENTRY"),
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


def generate_provider_shell(
    slug: str,
    cat_enum: str,
    *,
    factory: str,
    env: str,
    providers_root: Path = PROVIDERS,
) -> dict[str, bool]:
    """Generate or preserve legacy shell files for one provider slug."""
    category = _category_folder(cat_enum)
    assert SLUG_CATEGORY.get(slug) == category, f"{slug}: layout mismatch {SLUG_CATEGORY.get(slug)} != {category}"
    pkg = providers_root / category / slug
    pkg.mkdir(parents=True, exist_ok=True)
    import_base = f"intergrax.integrations.providers.{category}.{slug}"
    written: dict[str, bool] = {}
    written["register.py"] = write_provider_file_if_allowed(
        pkg,
        "register.py",
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
        + f'            description="{slug} integration (Phase M.7)",\n'
        + "        ),\n"
        + "        override=override,\n"
        + "    )\n",
    )
    written["bundle.py"] = write_provider_file_if_allowed(
        pkg,
        "bundle.py",
        H + f"from intergrax.integrations._shared.p3.factories import {factory}\n\n__all__ = [\"{factory}\"]\n",
    )
    written["__init__.py"] = write_provider_file_if_allowed(
        pkg,
        "__init__.py",
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
    )
    return written


def main() -> None:
    generated = 0
    for slug, cat_enum, _enum, factory, env in SPECS:
        generate_provider_shell(slug, cat_enum, factory=factory, env=env)
        generated += 1
    print(f"generated {generated} provider shells")


if __name__ == "__main__":
    main()
