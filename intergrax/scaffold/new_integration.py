# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from intergrax.scaffold.integration_templates import (
    generate_bundle_py,
    generate_integration_py,
    generate_register_py,
    generate_usage_md,
    validate_category,
)


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "new-integration",
        help="Create intergrax/integrations/providers/<category>/<slug>/ INTEGRATIONS-2E scaffold",
    )
    parser.add_argument("slug", help="Integration slug (e.g. acme_kv)")
    parser.add_argument(
        "--category",
        required=True,
        help="IntegrationCategory folder name (e.g. key_value_cache)",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true")


def run_new_integration(args: argparse.Namespace) -> int:
    slug = args.slug.strip().lower().replace("-", "_")
    category = args.category.strip().lower().replace("-", "_")
    if not slug or not category:
        print("error: slug and category are required", flush=True)
        return 1

    category_error = validate_category(category)
    if category_error:
        print(category_error, flush=True)
        return 1

    root = args.root.resolve()
    provider_dir = root / "intergrax" / "integrations" / "providers" / category / slug
    if provider_dir.exists() and not args.force:
        print(f"error: {provider_dir} already exists (use --force)", flush=True)
        return 1
    provider_dir.mkdir(parents=True, exist_ok=True)

    manifest_const = slug.upper()

    (provider_dir / "manifest.py").write_text(
        textwrap.dedent(
            f'''\
            # © Artur Czarnecki. All rights reserved.

            from __future__ import annotations

            from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
            from intergrax.integrations.core.manifest import IntegrationManifest

            MANIFEST = IntegrationManifest(
                slug="{slug}",
                categories=(IntegrationCategory.{category.upper()},),
                status=IntegrationStatus.BETA,
                env_prefix="INTERGRAX_{manifest_const}",
                description="TODO: describe this integration provider",
            )
            '''
        ),
        encoding="utf-8",
    )
    (provider_dir / "integration.py").write_text(generate_integration_py(slug, category), encoding="utf-8")
    (provider_dir / "bundle.py").write_text(generate_bundle_py(slug, category), encoding="utf-8")
    (provider_dir / "register.py").write_text(generate_register_py(slug, category), encoding="utf-8")
    (provider_dir / "USAGE.md").write_text(generate_usage_md(slug, category), encoding="utf-8")

    print(f"Created integration scaffold under {provider_dir}")
    return 0
