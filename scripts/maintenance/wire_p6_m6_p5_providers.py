#!/usr/bin/env python3
"""
Generate thin M.6 P5 provider shells pointing to _shared.p6.factories.

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
    ("gitlab_ci", "ci_cd", "CI_CD", "create_gitlab_ci_ci_cd", "INTERGRAX_GITLAB_CI"),
    ("circleci", "ci_cd", "CI_CD", "create_circleci_ci_cd", "INTERGRAX_CIRCLECI"),
    ("azure_pipelines", "ci_cd", "CI_CD", "create_azure_pipelines_ci_cd", "INTERGRAX_AZURE_PIPELINES"),
    ("mailpit", "notification_channel", "NOTIFICATION_CHANNEL", "create_mailpit_notification_channel", "INTERGRAX_MAILPIT"),
    ("localstack", "cloud_platform", "CLOUD_PLATFORM", "create_localstack_cloud_platform", "INTERGRAX_LOCALSTACK"),
    ("codecov", "ci_cd", "CI_CD", "create_codecov_ci_cd", "INTERGRAX_CODECOV"),
    ("grafana_oncall", "notification_channel", "NOTIFICATION_CHANNEL", "create_grafana_oncall_notification_channel", "INTERGRAX_GRAFANA_ONCALL"),
    (
        "opentelemetry_collector",
        "observability_backend",
        "OBSERVABILITY_BACKEND",
        "create_opentelemetry_collector_observability_backend",
        "INTERGRAX_OPENTELEMETRY_COLLECTOR",
    ),
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
        + f"    description='{slug} integration (Phase M.6 P5)',\n"
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
        H + f"from intergrax.integrations._shared.p6.factories import {factory}\n\n__all__ = [\"{factory}\"]\n",
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
    print(f"generated {generated} M.6 P5 provider shells")


if __name__ == "__main__":
    main()
