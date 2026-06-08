#!/usr/bin/env python3
"""Generate thin M.7 P7 provider shells pointing to _shared.p8.factories."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROVIDERS = ROOT / "intergrax" / "integrations" / "providers"
H = "# © Artur Czarnecki. All rights reserved.\n# Intergrax framework – proprietary and confidential.\n\n"

SPECS = [
    ("perplexity", "search_provider", "SEARCH_PROVIDER", "create_perplexity_search_provider", "INTERGRAX_PERPLEXITY"),
    ("arxiv", "search_provider", "SEARCH_PROVIDER", "create_arxiv_search_provider", "INTERGRAX_ARXIV"),
    ("semantic_scholar", "search_provider", "SEARCH_PROVIDER", "create_semantic_scholar_search_provider", "INTERGRAX_SEMANTIC_SCHOLAR"),
    ("llamaparse", "document_parser", "DOCUMENT_PARSER", "create_llamaparse_document_parser", "INTERGRAX_LLAMAPARSE"),
    ("lancedb", "vector_store", "VECTOR_STORE", "create_lancedb_vector_store", "INTERGRAX_LANCEDB"),
    ("telegram", "notification_channel", "NOTIFICATION_CHANNEL, INTERACTION_SURFACE", "create_telegram_catalog_factory", "INTERGRAX_TELEGRAM", True),
    ("browserbase", "browser_automation", "BROWSER_AUTOMATION", "create_browserbase_browser_automation", "INTERGRAX_BROWSERBASE"),
    ("google_drive", "object_storage", "OBJECT_STORAGE", "create_google_drive_object_storage", "INTERGRAX_GOOGLE_DRIVE"),
    ("n8n", "workflow_orchestrator", "WORKFLOW_ORCHESTRATOR", "create_n8n_workflow_orchestrator", "INTERGRAX_N8N"),
    ("wikipedia", "wiki_knowledge", "WIKI_KNOWLEDGE", "create_wikipedia_wiki_knowledge", "INTERGRAX_WIKIPEDIA"),
    ("clerk", "identity_provider", "IDENTITY_PROVIDER", "create_clerk_identity_provider", "INTERGRAX_CLERK"),
    ("upstash_redis", "key_value_cache", "KEY_VALUE_CACHE", "create_upstash_redis_key_value_cache", "INTERGRAX_UPSTASH_REDIS"),
    ("upstash_qstash", "message_bus", "MESSAGE_BUS", "create_upstash_qstash_message_bus", "INTERGRAX_UPSTASH_QSTASH"),
    ("okta", "identity_provider", "IDENTITY_PROVIDER", "create_okta_identity_provider", "INTERGRAX_OKTA"),
    ("bigquery", "relational_store", "RELATIONAL_STORE", "create_bigquery_relational_store", "INTERGRAX_BIGQUERY"),
    ("motherduck", "relational_store", "RELATIONAL_STORE", "create_motherduck_relational_store", "INTERGRAX_MOTHERDUCK"),
    ("airbyte", "workflow_orchestrator", "WORKFLOW_ORCHESTRATOR", "create_airbyte_workflow_orchestrator", "INTERGRAX_AIRBYTE"),
    ("apify", "browser_automation", "BROWSER_AUTOMATION", "create_apify_browser_automation", "INTERGRAX_APIFY"),
]

for spec in SPECS:
    if len(spec) == 6:
        slug, category, cat_enum, factory, env, dual = spec
    else:
        slug, category, cat_enum, factory, env = spec
        dual = False

    pkg = PROVIDERS / category / slug
    pkg.mkdir(parents=True, exist_ok=True)
    import_base = f"intergrax.integrations.providers.{category}.{slug}"

    if dual:
        categories_line = (
            "    categories=(IntegrationCategory.NOTIFICATION_CHANNEL, IntegrationCategory.INTERACTION_SURFACE,),\n"
        )
        phase = "Phase M.7 P7 — dual notification + interaction"
    else:
        enum_name = cat_enum.split(",")[0].strip()
        categories_line = f"    categories=(IntegrationCategory.{enum_name},),\n"
        phase = "Phase M.7 P7"

    (pkg / "manifest.py").write_text(
        H
        + f'"""Catalog manifest for ``{slug}`` integration."""\n\nfrom __future__ import annotations\n\n'
        + "from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus\n"
        + "from intergrax.integrations.core.manifest import IntegrationManifest\n\n"
        + "MANIFEST = IntegrationManifest(\n"
        + f'    slug="{slug}",\n'
        + categories_line
        + "    status=IntegrationStatus.STABLE,\n"
        + f"    env_prefix='{env}',\n"
        + f"    description='{slug} integration ({phase})',\n"
        + ")\n",
        encoding="utf-8",
    )
    (pkg / "register.py").write_text(
        H
        + f'"""Register {slug} in the integration catalog."""\n\nfrom __future__ import annotations\n\n'
        + f"from {import_base}.bundle import {factory}\n"
        + f"from {import_base}.manifest import MANIFEST\n"
        + "from intergrax.integrations.registry.plugin_register import register_from_manifest\n\n\n"
        + f"def register_{slug}_integration(*, override: bool = False) -> None:\n"
        + f"    register_from_manifest(MANIFEST, {factory}, override=override)\n",
        encoding="utf-8",
    )
    (pkg / "bundle.py").write_text(
        H + f"from intergrax.integrations._shared.p8.factories import {factory}\n\n__all__ = [\"{factory}\"]\n",
        encoding="utf-8",
    )
    (pkg / "__init__.py").write_text(
        H
        + f'from {import_base}.bundle import {factory}\n'
        + f"from {import_base}.register import register_{slug}_integration\n\n"
        + f'__all__ = ["{factory}", "register_{slug}_integration"]\n',
        encoding="utf-8",
    )
    profile_field = category
    (pkg / "USAGE.md").write_text(
        H
        + f"# `{slug}` integration — usage\n\n"
        + f"**Category:** `{profile_field}`  \n"
        + f"**Catalog factory:** ``{factory}()``  \n"
        + f"**Env prefix:** ``{env}_*``\n\n"
        + f"```python\nfrom {import_base}.bundle import {factory}\n\n"
        + f"backend = {factory}()\n```\n",
        encoding="utf-8",
    )

print(f"generated {len(SPECS)} M.7 P7 provider shells")
