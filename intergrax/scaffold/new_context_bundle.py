# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "new-context-bundle",
        help="Create intergrax/context/providers/<bundle_id>/ ContextPlugin scaffold",
    )
    parser.add_argument("bundle_id", help="Context bundle id (e.g. acme_context)")
    parser.add_argument(
        "--provider-id",
        help="Primary provider id (default: <bundle_id>.source)",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true")


def _class_prefix(bundle_id: str) -> str:
    return "".join(part.title() for part in bundle_id.replace(".", "_").split("_") if part)


def run_new_context_bundle(args: argparse.Namespace) -> int:
    bundle_id = args.bundle_id.strip().lower().replace("-", "_")
    if not bundle_id:
        print("error: bundle_id is required", flush=True)
        return 1
    class_prefix = _class_prefix(bundle_id)
    if not class_prefix:
        print("error: bundle_id is required", flush=True)
        return 1
    provider_id = (args.provider_id or f"{bundle_id}.source").strip()
    if not provider_id:
        print("error: provider_id is required", flush=True)
        return 1
    root = args.root.resolve()
    provider_dir = root / "intergrax" / "context" / "providers" / bundle_id
    if provider_dir.exists() and not args.force:
        print(f"error: {provider_dir} already exists (use --force)", flush=True)
        return 1
    provider_dir.mkdir(parents=True, exist_ok=True)

    plugin_class = f"{class_prefix}ContextPlugin"
    provider_class = f"{class_prefix}SourceProvider"
    helper_name = f"register_{bundle_id.replace('.', '_')}_context_bundle"
    module_path = f"intergrax.context.providers.{bundle_id}.plugin"

    (provider_dir / "plugin.py").write_text(
        textwrap.dedent(
            f'''\
            # © Artur Czarnecki. All rights reserved.

            from __future__ import annotations

            from intergrax.context.contracts import (
                ContextAssemblyRequest,
                ContextFragment,
                ContextFragmentSource,
                ContextProviderContext,
            )
            from intergrax.context.plugin import ContextPlugin
            from intergrax.context.registry import ContextPluginRegistry


            class {provider_class}:
                @property
                def provider_id(self) -> str:
                    return "{provider_id}"

                @property
                def supported_sources(self) -> frozenset[ContextFragmentSource]:
                    return frozenset({{ContextFragmentSource.CUSTOM}})

                async def collect(
                    self,
                    request: ContextAssemblyRequest,
                    ctx: ContextProviderContext,
                ) -> list[ContextFragment]:
                    _ = request, ctx
                    return [
                        ContextFragment(
                            fragment_id="{bundle_id}-placeholder-1",
                            source=ContextFragmentSource.CUSTOM,
                            source_id="{bundle_id}",
                            content="TODO: replace this placeholder context fragment",
                            token_estimate=8,
                            relevance_score=0.5,
                            freshness_score=0.5,
                            confidence_score=0.5,
                            mandatory=False,
                        )
                    ]


            class {plugin_class}:
                @classmethod
                def plugin_id(cls) -> str:
                    return "{bundle_id}"

                @classmethod
                def plugin_version(cls) -> str:
                    return "0.1.0"

                @classmethod
                def plugin_description(cls) -> str:
                    return "TODO: describe this context plugin"

                @classmethod
                def register(cls, registry: ContextPluginRegistry) -> None:
                    registry.add_provider({provider_class}())


            PLUGIN_TYPE: type[ContextPlugin] = {plugin_class}
            '''
        ),
        encoding="utf-8",
    )
    (provider_dir / "bundle.py").write_text(
        textwrap.dedent(
            f'''\
            # © Artur Czarnecki. All rights reserved.

            from intergrax.context.plugin import register_context_plugin
            from intergrax.context.providers.{bundle_id}.plugin import {plugin_class}


            def {helper_name}(*, override: bool = False) -> None:
                register_context_plugin({plugin_class}, override=override)
            '''
        ),
        encoding="utf-8",
    )
    (provider_dir / "USAGE.md").write_text(
        (
            f"# Context plugin `{bundle_id}`\n\n"
            "Trusted in-process Python plugin. Generating this skeleton does not enable it.\n\n"
            "**installed ≠ enabled.** Installation or catalog registration is not activation.\n\n"
            "## Local registration (canonical)\n\n"
            "```python\n"
            "from intergrax.context.plugin import register_context_plugin\n"
            f"from {module_path} import {plugin_class}\n\n"
            f"register_context_plugin({plugin_class})\n"
            "```\n\n"
            f"Helper: `{helper_name}()` in `bundle.py`.\n\n"
            "Register before `bootstrap_context_catalog()`, or pass "
            f"`context_plugins=[{plugin_class}]` to bootstrap. "
            "Manual host composition remains valid.\n\n"
            "## Enablement (`ContextProfile`)\n\n"
            "Catalog presence does not materialize providers. Enable the plugin id:\n\n"
            "```python\n"
            "from intergrax.applications.contracts.environment_profile import ContextProfile\n\n"
            "context_profile = ContextProfile(\n"
            f'    context_plugin_ids=["{bundle_id}"]\n'
            ")\n"
            "```\n\n"
            "## Entry-point delivery (wheel)\n\n"
            "This scaffold does not build or install a wheel. Wheel is delivery/discovery only.\n\n"
            "```toml\n"
            '[project.entry-points."intergrax.context"]\n'
            f'{bundle_id} = "{module_path}:{plugin_class}"\n'
            "```\n\n"
            "For an external package, replace the module path with your package.\n\n"
            "**EP mode:** the plugin must be installed, **discovery enabled**, and listed in "
            "`context_plugin_ids`. Enable discovery with `discover_entry_points=True` on "
            "`bootstrap_context_catalog`, `INTERGRAX_DISCOVER_PLUGINS=true`, or application "
            "wiring that calls `bootstrap_application_context_catalog(discover_entry_points=True)`.\n\n"
            f"- Plugin: `{plugin_class}` (`plugin_id` = `{bundle_id}`)\n"
            f"- Provider: `{provider_class}` (`provider_id` = `{provider_id}`)\n"
            "- Placeholder `supported_sources`: `ContextFragmentSource.CUSTOM`\n"
        ),
        encoding="utf-8",
    )
    print(f"Created context bundle scaffold under {provider_dir}")
    return 0
