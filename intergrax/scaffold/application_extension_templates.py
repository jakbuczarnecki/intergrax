# © Artur Czarnecki. All rights reserved.

"""Scaffold templates for host-embedded Platform Plugin extensions (PLUGIN-8)."""

from __future__ import annotations

from textwrap import dedent

from intergrax.scaffold.application_names import ScaffoldApplicationNames


def extensions_readme(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    return dedent(
        f"""\
        # Host-embedded extensions ({pkg})

        Add local Platform Plugin implementations here when you do **not** want to
        package them as external wheels.

        - Implement the public domain contract (e.g. `ToolPlugin`).
        - Register explicitly from `host/tool_wiring.py` (or your composition root).
        - Pass dependencies via the domain wiring context (`ToolWiringContext` for tools).
        - Production hosts still require production-qualified evidence.

        Reference: `examples/platform_plugins/local_embedded_tool_extension/` and
        [`EXTENSION_AUTHOR_GUIDE.md`](../../docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md).
        """
    )


def local_prefix_echo_plugin_py(names: ScaffoldApplicationNames) -> str:
    return dedent(
        '''\
        # © Artur Czarnecki. All rights reserved.

        """Example host-embedded ToolPlugin — remove or replace for your application."""

        from __future__ import annotations

        from pydantic import BaseModel, Field

        from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
        from intergrax.tools.core.handler import ServiceToolHandler
        from intergrax.tools.core.manifest import ToolBundleManifest
        from intergrax.tools.registry.catalog import ToolBundleStatus
        from intergrax.tools.registry.runtime import ToolRegistry
        from intergrax.tools.registry.wiring import ToolWiringContext

        LOCAL_PREFIX_ECHO_TOOL_ID = "local_prefix_echo.ping"


        class LocalPrefixEchoInput(BaseModel):
            message: str = Field(default="pong")


        class LocalPrefixEchoOutput(BaseModel):
            message: str


        def _local_prefix_echo_contract() -> ToolContract:
            return ToolContract(
                tool_id=LOCAL_PREFIX_ECHO_TOOL_ID,
                name="local_prefix_echo.ping",
                description="Scaffold example host-embedded echo tool.",
                input_schema=LocalPrefixEchoInput,
                output_schema=LocalPrefixEchoOutput,
                error_mapping={},
                side_effects=False,
                risk_level=ToolRiskLevel.LOW,
                tags=("example", "host-embedded"),
            )


        def _prefix_echo_service(
            ctx: ToolWiringContext,
            request: LocalPrefixEchoInput,
        ) -> LocalPrefixEchoOutput:
            prefix = str(ctx.extras.get("echo_prefix", ""))
            body = request.message
            if prefix:
                body = f"{prefix}:{body}"
            return LocalPrefixEchoOutput(message=body)


        class LocalPrefixEchoHandler(
            ServiceToolHandler[LocalPrefixEchoInput, LocalPrefixEchoOutput],
        ):
            _service = _prefix_echo_service


        class LocalPrefixEchoToolPlugin:
            @classmethod
            def tool_bundle_manifest(cls) -> ToolBundleManifest:
                return ToolBundleManifest(
                    bundle_id="local_prefix_echo",
                    tool_ids=(LOCAL_PREFIX_ECHO_TOOL_ID,),
                    status=ToolBundleStatus.BETA,
                    description="Scaffold example host-embedded tool bundle.",
                )

            @classmethod
            def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
                registry.register(_local_prefix_echo_contract(), LocalPrefixEchoHandler(ctx))
        '''
    )


def tool_wiring_local_extension_block(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    return dedent(
        f"""\
        from intergrax.tools.registry.plugin_register import register_tool_plugin
        from {pkg}.extensions.local_prefix_echo_plugin import LocalPrefixEchoToolPlugin

        register_tool_plugin(LocalPrefixEchoToolPlugin)
        """
    )
