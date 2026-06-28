#!/usr/bin/env python3
"""
Generate thin M.6 P6 provider shells pointing to _shared.p7.factories.

Legacy shell generator for unmigrated providers. When ``integration.py`` exists
in a provider package, canonical files are preserved (contract-aware mode).
Do not use this script to regenerate migrated contract-based providers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.maintenance._provider_shell_contract import write_provider_file_if_allowed

PROVIDERS = ROOT / "intergrax" / "integrations" / "providers"
H = "# © Artur Czarnecki. All rights reserved.\n# Intergrax framework – proprietary and confidential.\n\n"

SPECS = [
    ("trivy", "security_scanner", "SECURITY_SCANNER", "create_trivy_security_scanner", "INTERGRAX_TRIVY"),
    ("snyk", "security_scanner", "SECURITY_SCANNER", "create_snyk_security_scanner", "INTERGRAX_SNYK"),
    ("semgrep", "security_scanner", "SECURITY_SCANNER", "create_semgrep_security_scanner", "INTERGRAX_SEMGREP"),
    ("infisical", "secrets_store", "SECRETS_STORE", "create_infisical_secrets_store", "INTERGRAX_INFISICAL"),
    ("e2b", "sandbox_host", "SANDBOX_HOST", "create_e2b_sandbox_host", "INTERGRAX_E2B"),
    ("modal", "sandbox_host", "SANDBOX_HOST", "create_modal_sandbox_host", "INTERGRAX_MODAL"),
    ("daytona", "sandbox_host", "SANDBOX_HOST", "create_daytona_sandbox_host", "INTERGRAX_DAYTONA"),
    ("auth0", "identity_provider", "IDENTITY_PROVIDER", "create_auth0_identity_provider", "INTERGRAX_AUTH0"),
    ("keycloak", "identity_provider", "IDENTITY_PROVIDER", "create_keycloak_identity_provider", "INTERGRAX_KEYCLOAK"),
    ("workos", "identity_provider", "IDENTITY_PROVIDER", "create_workos_identity_provider", "INTERGRAX_WORKOS"),
    ("argocd", "ci_cd", "CI_CD", "create_argocd_ci_cd", "INTERGRAX_ARGOCD"),
    ("buildkite", "ci_cd", "CI_CD", "create_buildkite_ci_cd", "INTERGRAX_BUILDKITE"),
    ("jenkins", "ci_cd", "CI_CD", "create_jenkins_ci_cd", "INTERGRAX_JENKINS"),
    ("elevenlabs", "speech_provider", "SPEECH_PROVIDER", "create_elevenlabs_speech_provider", "INTERGRAX_ELEVENLABS"),
    ("deepgram", "speech_provider", "SPEECH_PROVIDER", "create_deepgram_speech_provider", "INTERGRAX_DEEPGRAM"),
    ("newrelic", "observability_backend", "OBSERVABILITY_BACKEND", "create_newrelic_observability_backend", "INTERGRAX_NEWRELIC"),
    ("splunk", "observability_backend", "OBSERVABILITY_BACKEND", "create_splunk_observability_backend", "INTERGRAX_SPLUNK"),
    ("zendesk", "issue_tracker", "ISSUE_TRACKER", "create_zendesk_issue_tracker", "INTERGRAX_ZENDESK"),
    ("statsig", "feature_flag", "FEATURE_FLAG", "create_statsig_feature_flag", "INTERGRAX_STATSIG"),
    ("prefect", "workflow_orchestrator", "WORKFLOW_ORCHESTRATOR", "create_prefect_workflow_orchestrator", "INTERGRAX_PREFECT"),
    ("airflow", "workflow_orchestrator", "WORKFLOW_ORCHESTRATOR", "create_airflow_workflow_orchestrator", "INTERGRAX_AIRFLOW"),
    ("typesense", "vector_store", "VECTOR_STORE", "create_typesense_vector_store", "INTERGRAX_TYPESENSE"),
    ("neon", "relational_store", "RELATIONAL_STORE", "create_neon_relational_store", "INTERGRAX_NEON"),
    ("pulsar", "message_bus", "MESSAGE_BUS", "create_pulsar_message_bus", "INTERGRAX_PULSAR"),
    ("algolia", "search_provider", "SEARCH_PROVIDER", "create_algolia_search_provider", "INTERGRAX_ALGOLIA"),
    ("confluent", "message_bus", "MESSAGE_BUS", "create_confluent_message_bus", "INTERGRAX_CONFLUENT"),
    ("backblaze_b2", "object_storage", "OBJECT_STORAGE", "create_backblaze_b2_object_storage", "INTERGRAX_BACKBLAZE_B2"),
    ("triton", "vision_serving", "VISION_SERVING", "create_triton_vision_serving", "INTERGRAX_TRITON"),
    ("replicate", "ml_inference_host", "ML_INFERENCE_HOST", "create_replicate_ml_inference_host", "INTERGRAX_REPLICATE"),
    ("stripe", "billing_meter", "BILLING_METER", "create_stripe_billing_meter", "INTERGRAX_STRIPE"),
    ("salesforce", "crm", "CRM", "create_salesforce_crm", "INTERGRAX_SALESFORCE"),
    ("hubspot", "crm", "CRM", "create_hubspot_crm", "INTERGRAX_HUBSPOT"),
]


def generate_provider_shell(
    slug: str,
    category: str,
    cat_enum: str,
    *,
    factory: str,
    env: str,
    providers_root: Path = PROVIDERS,
) -> dict[str, bool]:
    """Generate or preserve legacy shell files for one provider slug."""
    pkg = providers_root / category / slug
    pkg.mkdir(parents=True, exist_ok=True)
    import_base = f"intergrax.integrations.providers.{category}.{slug}"
    written: dict[str, bool] = {}
    written["manifest.py"] = write_provider_file_if_allowed(
        pkg,
        "manifest.py",
        H
        + f'"""Catalog manifest for ``{slug}`` integration."""\n\nfrom __future__ import annotations\n\n'
        + "from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus\n"
        + "from intergrax.integrations.core.manifest import IntegrationManifest\n\n"
        + "MANIFEST = IntegrationManifest(\n"
        + f'    slug="{slug}",\n'
        + f"    categories=(IntegrationCategory.{cat_enum},),\n"
        + "    status=IntegrationStatus.STABLE,\n"
        + f"    env_prefix='{env}',\n"
        + f"    description='{slug} integration (Phase M.6 P6)',\n"
        + ")\n",
    )
    written["register.py"] = write_provider_file_if_allowed(
        pkg,
        "register.py",
        H
        + f'"""Register {slug} in the integration catalog."""\n\nfrom __future__ import annotations\n\n'
        + f"from {import_base}.bundle import {factory}\n"
        + f"from {import_base}.manifest import MANIFEST\n"
        + "from intergrax.integrations.registry.plugin_register import register_from_manifest\n\n\n"
        + f"def register_{slug}_integration(*, override: bool = False) -> None:\n"
        + f"    register_from_manifest(MANIFEST, {factory}, override=override)\n",
    )
    written["bundle.py"] = write_provider_file_if_allowed(
        pkg,
        "bundle.py",
        H + f"from intergrax.integrations._shared.p7.factories import {factory}\n\n__all__ = [\"{factory}\"]\n",
    )
    written["__init__.py"] = write_provider_file_if_allowed(
        pkg,
        "__init__.py",
        H
        + f'from {import_base}.bundle import {factory}\n'
        + f"from {import_base}.register import register_{slug}_integration\n\n"
        + f'__all__ = ["{factory}", "register_{slug}_integration"]\n',
    )
    profile_field = category
    written["USAGE.md"] = write_provider_file_if_allowed(
        pkg,
        "USAGE.md",
        H
        + f"# `{slug}` integration — usage\n\n"
        + f"**Category:** `{profile_field}`  \n"
        + f"**Catalog factory:** ``{factory}()``  \n"
        + f"**Env prefix:** ``{env}_*``\n\n"
        + f"```python\nfrom {import_base}.bundle import {factory}\n\n"
        + f"backend = {factory}()\n```\n",
    )
    return written


def main() -> None:
    generated = 0
    for slug, category, cat_enum, factory, env in SPECS:
        generate_provider_shell(slug, category, cat_enum, factory=factory, env=env)
        generated += 1
    print(f"generated {generated} M.6 P6 provider shells")


if __name__ == "__main__":
    main()
