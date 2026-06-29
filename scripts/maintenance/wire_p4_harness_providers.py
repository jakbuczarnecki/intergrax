#!/usr/bin/env python3
"""
Generate thin provider shells pointing to _shared.p4.factories.

Legacy shell generator for unmigrated providers. When ``integration.py`` exists
in a provider package, canonical files are preserved (contract-aware mode).
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
    ("langsmith", "OBSERVABILITY_BACKEND", "LANGSMITH", "create_langsmith_observability_backend", "INTERGRAX_LANGSMITH"),
    ("helicone", "OBSERVABILITY_BACKEND", "HELICONE", "create_helicone_observability_backend", "INTERGRAX_HELICONE"),
    ("posthog", "OBSERVABILITY_BACKEND", "POSTHOG", "create_posthog_observability_backend", "INTERGRAX_POSTHOG"),
    ("braintrust", "OBSERVABILITY_BACKEND", "BRAINTRUST", "create_braintrust_observability_backend", "INTERGRAX_BRAINTRUST"),
    ("signoz", "OBSERVABILITY_BACKEND", "SIGNOZ", "create_signoz_observability_backend", "INTERGRAX_SIGNOZ"),
    ("honeycomb", "OBSERVABILITY_BACKEND", "HONEYCOMB", "create_honeycomb_observability_backend", "INTERGRAX_HONEYCOMB"),
    ("arize", "OBSERVABILITY_BACKEND", "ARIZE", "create_arize_observability_backend", "INTERGRAX_ARIZE"),
    ("phoenix", "OBSERVABILITY_BACKEND", "PHOENIX", "create_phoenix_observability_backend", "INTERGRAX_PHOENIX"),
    ("wandb", "OBSERVABILITY_BACKEND", "WANDB", "create_wandb_observability_backend", "INTERGRAX_WANDB"),
    ("opensearch", "OBSERVABILITY_BACKEND", "OPENSEARCH", "create_opensearch_observability_backend", "INTERGRAX_OPENSEARCH"),
    ("pagerduty", "NOTIFICATION_CHANNEL", "PAGERDUTY", "create_pagerduty_notification_channel", "INTERGRAX_PAGERDUTY"),
    ("opsgenie", "NOTIFICATION_CHANNEL", "OPSGENIE", "create_opsgenie_notification_channel", "INTERGRAX_OPSGENIE"),
    ("gitlab", "ISSUE_TRACKER", "GITLAB", "create_gitlab_issue_tracker", "INTERGRAX_GITLAB"),
    ("vespa", "VECTOR_STORE", "VESPA", "create_vespa_vector_store", "INTERGRAX_VESPA"),
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
        + f'            description="{slug} integration (Phase M.8 harness)",\n'
        + "        ),\n"
        + "        override=override,\n"
        + "    )\n",
    )
    written["bundle.py"] = write_provider_file_if_allowed(
        pkg,
        "bundle.py",
        H + f"from intergrax.integrations._shared.p4.factories import {factory}\n\n__all__ = [\"{factory}\"]\n",
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
