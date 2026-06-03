# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "new-integration",
        help="Create intergrax/integrations/providers/<category>/<slug>/ IntegrationPlugin scaffold",
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
    root = args.root.resolve()
    provider_dir = root / "intergrax" / "integrations" / "providers" / category / slug
    if provider_dir.exists() and not args.force:
        print(f"error: {provider_dir} already exists (use --force)", flush=True)
        return 1
    provider_dir.mkdir(parents=True, exist_ok=True)

    class_name = "".join(part.title() for part in slug.split("_")) + "IntegrationPlugin"
    manifest_const = slug.upper()
    factory_name = f"create_{slug}_{category}"

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
    (provider_dir / "adapter.py").write_text(
        textwrap.dedent(
            f'''\
            # © Artur Czarnecki. All rights reserved.

            from __future__ import annotations

            from typing import Any


            class {class_name.replace("IntegrationPlugin", "Adapter")}:
                """Minimal adapter stub — replace with a real integration contract."""

                def ping(self) -> str:
                    return "ok"
            '''
        ),
        encoding="utf-8",
    )
    (provider_dir / "plugin.py").write_text(
        textwrap.dedent(
            f'''\
            # © Artur Czarnecki. All rights reserved.

            from __future__ import annotations

            from typing import Any

            from intergrax.integrations.core.manifest import IntegrationManifest
            from intergrax.integrations.core.plugin import IntegrationPlugin
            from intergrax.integrations.providers.{category}.{slug}.adapter import {class_name.replace("IntegrationPlugin", "Adapter")}
            from intergrax.integrations.providers.{category}.{slug}.manifest import MANIFEST


            class {class_name}:
                @classmethod
                def integration_manifest(cls) -> IntegrationManifest:
                    return MANIFEST

                @classmethod
                def create_integration(cls, **kwargs: Any) -> Any:
                    _ = kwargs
                    return {class_name.replace("IntegrationPlugin", "Adapter")()}
            '''
        ),
        encoding="utf-8",
    )
    (provider_dir / "bundle.py").write_text(
        textwrap.dedent(
            f'''\
            # © Artur Czarnecki. All rights reserved.

            from __future__ import annotations

            from typing import Any

            from intergrax.integrations.providers.{category}.{slug}.plugin import {class_name}


            def {factory_name}(**kwargs: Any) -> Any:
                return {class_name}.create_integration(**kwargs)
            '''
        ),
        encoding="utf-8",
    )
    (provider_dir / "register.py").write_text(
        textwrap.dedent(
            f'''\
            # © Artur Czarnecki. All rights reserved.

            from intergrax.integrations.providers.{category}.{slug}.plugin import {class_name}
            from intergrax.integrations.registry.plugin_register import register_integration_plugin


            def register_{slug}_integration(*, override: bool = False) -> None:
                register_integration_plugin({class_name}, override=override)
            '''
        ),
        encoding="utf-8",
    )
    (provider_dir / "USAGE.md").write_text(
        textwrap.dedent(
            f'''\
            # `{slug}` integration — usage

            **Category:** ``{category}``

            ```python
            from intergrax.integrations.contracts.base import IntegrationCategory
            from intergrax.integrations.registry.bootstrap import register_default_integrations
            from intergrax.integrations.registry.profile import IntegrationProfile

            register_default_integrations()
            profile = IntegrationProfile({category}="{slug}")
            backend = profile.resolve(IntegrationCategory.{category.upper()})
            ```

            External plugin registration:

            ```python
            from intergrax.integrations.registry.plugin_register import register_integration_plugin
            from intergrax.integrations.providers.{category}.{slug}.plugin import {class_name}

            register_integration_plugin({class_name})
            ```
            '''
        ),
        encoding="utf-8",
    )
    print(f"Created integration scaffold under {provider_dir}")
    return 0
