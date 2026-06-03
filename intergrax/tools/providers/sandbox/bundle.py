# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.sandbox.contracts import SandboxExecInput, SandboxExecOutput
from intergrax.tools.providers.sandbox.handlers import SandboxExecHandler
from intergrax.tools.providers.sandbox.service import SANDBOX_EXEC_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

SANDBOX_BUNDLE_ID = "sandbox"
SANDBOX_TOOL_IDS: tuple[str, ...] = (SANDBOX_EXEC_TOOL_ID,)


def sandbox_exec_contract() -> ToolContract:
    return ToolContract(
        tool_id=SANDBOX_EXEC_TOOL_ID,
        name=SANDBOX_EXEC_TOOL_ID,
        description=(
            "Execute an allowlisted operation inside an isolated runtime sandbox session "
            "(echo, read_file, write_file, list_files)."
        ),
        description_short="Run sandbox operation.",
        input_schema=SandboxExecInput,
        output_schema=SandboxExecOutput,
        error_mapping={},
        side_effects=True,
        category="sandbox",
        risk_level=ToolRiskLevel.HIGH,
        tags=("sandbox", "execution"),
    )


def register_sandbox_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(sandbox_exec_contract(), SandboxExecHandler(ctx))


SANDBOX_EXEC_TOOL_CONTRACT = sandbox_exec_contract()
