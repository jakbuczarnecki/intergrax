# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "new-tool-bundle",
        help="Create intergrax/tools/providers/<bundle_id>/ ToolPlugin scaffold",
    )
    parser.add_argument("bundle_id", help="Tool bundle id (e.g. acme_ops)")
    parser.add_argument(
        "--tool-id",
        help="Primary tool id (default: <bundle_id>.ping)",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true")


def run_new_tool_bundle(args: argparse.Namespace) -> int:
    bundle_id = args.bundle_id.strip().lower().replace("-", "_")
    if not bundle_id:
        print("error: bundle_id is required", flush=True)
        return 1
    tool_id = (args.tool_id or f"{bundle_id}.ping").strip()
    root = args.root.resolve()
    provider_dir = root / "intergrax" / "tools" / "providers" / bundle_id
    if provider_dir.exists() and not args.force:
        print(f"error: {provider_dir} already exists (use --force)", flush=True)
        return 1
    provider_dir.mkdir(parents=True, exist_ok=True)

    class_name = "".join(part.title() for part in bundle_id.split("_")) + "ToolPlugin"
    const = tool_id.upper().replace(".", "_")

    (provider_dir / "plugin.py").write_text(
        textwrap.dedent(
            f'''\
            # © Artur Czarnecki. All rights reserved.

            from __future__ import annotations

            from pydantic import BaseModel, Field

            from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
            from intergrax.tools.core.handler import ServiceToolHandler
            from intergrax.tools.core.manifest import ToolBundleManifest
            from intergrax.tools.registry.catalog import ToolBundleStatus
            from intergrax.tools.registry.runtime import ToolRegistry
            from intergrax.tools.registry.wiring import ToolWiringContext

            {const}_ID = "{tool_id}"


            class {class_name.replace("ToolPlugin", "Input")}(BaseModel):
                message: str = Field(default="pong")


            class {class_name.replace("ToolPlugin", "Output")}(BaseModel):
                message: str


            def _contract() -> ToolContract:
                return ToolContract(
                    tool_id={const}_ID,
                    name="{tool_id}",
                    description="TODO: describe this tool",
                    input_schema={class_name.replace("ToolPlugin", "Input")},
                    output_schema={class_name.replace("ToolPlugin", "Output")},
                    error_mapping={{}},
                    side_effects=False,
                    risk_level=ToolRiskLevel.LOW,
                    tags=("{bundle_id}",),
                )


            def _service(_ctx: ToolWiringContext, request: {class_name.replace("ToolPlugin", "Input")}) -> {class_name.replace("ToolPlugin", "Output")}:
                return {class_name.replace("ToolPlugin", "Output")}(message=request.message)


            class {class_name.replace("ToolPlugin", "Handler")}(ServiceToolHandler[{class_name.replace("ToolPlugin", "Input")}, {class_name.replace("ToolPlugin", "Output")}]):
                _service = _service


            class {class_name}:
                @classmethod
                def tool_bundle_manifest(cls) -> ToolBundleManifest:
                    return ToolBundleManifest(
                        bundle_id="{bundle_id}",
                        tool_ids=({const}_ID,),
                        status=ToolBundleStatus.BETA,
                        description="TODO: describe this tool bundle",
                    )

                @classmethod
                def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
                    registry.register(_contract(), {class_name.replace("ToolPlugin", "Handler")}(ctx))
            '''
        ),
        encoding="utf-8",
    )
    (provider_dir / "bundle.py").write_text(
        textwrap.dedent(
            f'''\
            # © Artur Czarnecki. All rights reserved.

            from intergrax.tools.providers.{bundle_id}.plugin import {class_name}
            from intergrax.tools.registry.plugin_register import register_tool_plugin


            def register_{bundle_id}_tool_bundle(*, override: bool = False) -> None:
                register_tool_plugin({class_name}, override=override)
            '''
        ),
        encoding="utf-8",
    )
    (provider_dir / "USAGE.md").write_text(
        f"# Tool bundle `{bundle_id}`\n\n"
        f"Register with `register_tool_plugin({class_name})` or setuptools entry point `intergrax.tools`.\n",
        encoding="utf-8",
    )
    print(f"Created tool bundle scaffold under {provider_dir}")
    return 0
