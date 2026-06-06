#!/usr/bin/env python3
"""Generate thin M.6 P4 provider shells pointing to _shared.p5.factories."""
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
    ("pgvector", "VECTOR_STORE", "create_pgvector_vector_store", "INTERGRAX_PGVECTOR"),
    ("duckdb", "RELATIONAL_STORE", "create_duckdb_relational_store", "INTERGRAX_DUCKDB"),
    ("influxdb", "OBSERVABILITY_BACKEND", "create_influxdb_observability_backend", "INTERGRAX_INFLUXDB"),
    ("timescaledb", "RELATIONAL_STORE", "create_timescaledb_relational_store", "INTERGRAX_TIMESCALEDB"),
    ("grafana", "OBSERVABILITY_BACKEND", "create_grafana_observability_backend", "INTERGRAX_GRAFANA"),
    ("loki", "OBSERVABILITY_BACKEND", "create_loki_observability_backend", "INTERGRAX_LOKI"),
    ("tempo", "OBSERVABILITY_BACKEND", "create_tempo_observability_backend", "INTERGRAX_TEMPO"),
    ("aws_secrets_manager", "SECRETS_STORE", "create_aws_secrets_manager_secrets_store", "INTERGRAX_AWS_SECRETS_MANAGER"),
    ("azure_key_vault", "SECRETS_STORE", "create_azure_key_vault_secrets_store", "INTERGRAX_AZURE_KEY_VAULT"),
    ("gcp_secret_manager", "SECRETS_STORE", "create_gcp_secret_manager_secrets_store", "INTERGRAX_GCP_SECRET_MANAGER"),
    ("doppler", "SECRETS_STORE", "create_doppler_secrets_store", "INTERGRAX_DOPPLER"),
    ("unleash", "FEATURE_FLAG", "create_unleash_feature_flag", "INTERGRAX_UNLEASH"),
    ("launchdarkly", "FEATURE_FLAG", "create_launchdarkly_feature_flag", "INTERGRAX_LAUNCHDARKLY"),
    ("github_actions", "CI_CD", "create_github_actions_ci_cd", "INTERGRAX_GITHUB_ACTIONS"),
    ("redpanda", "MESSAGE_BUS", "create_redpanda_message_bus", "INTERGRAX_REDPANDA"),
    ("cloudflare_r2", "OBJECT_STORAGE", "create_cloudflare_r2_object_storage", "INTERGRAX_CLOUDFLARE_R2"),
    ("memgraph", "GRAPH_STORE", "create_memgraph_graph_store", "INTERGRAX_MEMGRAPH"),
    ("falkordb", "GRAPH_STORE", "create_falkordb_graph_store", "INTERGRAX_FALKORDB"),
    ("incident_io", "NOTIFICATION_CHANNEL", "create_incident_io_notification_channel", "INTERGRAX_INCIDENT_IO"),
    ("kubernetes", "CLOUD_PLATFORM", "create_kubernetes_cloud_platform", "INTERGRAX_KUBERNETES"),
    ("servicenow", "ISSUE_TRACKER", "create_servicenow_issue_tracker", "INTERGRAX_SERVICENOW"),
    ("bitbucket", "ISSUE_TRACKER", "create_bitbucket_issue_tracker", "INTERGRAX_BITBUCKET"),
    ("asana", "ISSUE_TRACKER", "create_asana_issue_tracker", "INTERGRAX_ASANA"),
    ("sendgrid", "NOTIFICATION_CHANNEL", "create_sendgrid_notification_channel", "INTERGRAX_SENDGRID"),
    ("mailgun", "INTERACTION_SURFACE", "create_mailgun_interaction_surface", "INTERGRAX_MAILGUN"),
    ("mlflow", "OBSERVABILITY_BACKEND", "create_mlflow_observability_backend", "INTERGRAX_MLFLOW"),
    ("huggingface_hub", "OBJECT_STORAGE", "create_huggingface_hub_object_storage", "INTERGRAX_HUGGINGFACE_HUB"),
    ("ollama", "INTERACTION_SURFACE", "create_ollama_interaction_surface", "INTERGRAX_OLLAMA"),
]


def _category_folder(cat_enum: str) -> str:
    return IntegrationCategory[cat_enum].value


for slug, cat_enum, factory, env in SPECS:
    category = _category_folder(cat_enum)
    assert SLUG_CATEGORY.get(slug) == category, f"{slug}: layout mismatch"
    pkg = PROVIDERS / category / slug
    pkg.mkdir(parents=True, exist_ok=True)
    import_base = f"intergrax.integrations.providers.{category}.{slug}"
    (pkg / "manifest.py").write_text(
        H
        + f'"""Catalog manifest for ``{slug}`` integration."""\n\nfrom __future__ import annotations\n\n'
        + "from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus\n"
        + "from intergrax.integrations.core.manifest import IntegrationManifest\n\n"
        + "MANIFEST = IntegrationManifest(\n"
        + f'    slug="{slug}",\n'
        + f"    categories=(IntegrationCategory.{cat_enum},),\n"
        + "    status=IntegrationStatus.BETA,\n"
        + f"    env_prefix='{env}',\n"
        + f"    description='{slug} integration (Phase M.6 P4)',\n"
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
        H + f"from intergrax.integrations._shared.p5.factories import {factory}\n\n__all__ = [\"{factory}\"]\n",
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
    profile_field = IntegrationCategory[cat_enum].value
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

print(f"generated {len(SPECS)} M.6 P4 provider shells")
