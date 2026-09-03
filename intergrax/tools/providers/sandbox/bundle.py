# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolIsolationRequirement, ToolRiskLevel
from intergrax.tools.providers.sandbox.contracts import (
    BrowserRunInput,
    BrowserRunOutput,
    CodeExecInput,
    SandboxExecInput,
    SandboxExecOutput,
    SandboxListOperationsInput,
    SandboxListOperationsOutput,
    ScriptRunInput,
)
from intergrax.tools.providers.sandbox.extended_handlers import (
    BrowserRunHandler,
    CodeExecHandler,
    SandboxListOperationsHandler,
    ScriptRunHandler,
)
from intergrax.tools.providers.sandbox.extended_service import (
    BROWSER_RUN_TOOL_ID,
    CODE_EXEC_TOOL_ID,
    SANDBOX_LIST_OPERATIONS_TOOL_ID,
    SCRIPT_RUN_TOOL_ID,
)
from intergrax.tools.providers.sandbox.handlers import SandboxExecHandler
from intergrax.tools.providers.sandbox.service import SANDBOX_EXEC_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

SANDBOX_BUNDLE_ID = "sandbox"
SANDBOX_TOOL_IDS: tuple[str, ...] = (
    SANDBOX_EXEC_TOOL_ID,
    CODE_EXEC_TOOL_ID,
    SCRIPT_RUN_TOOL_ID,
    BROWSER_RUN_TOOL_ID,
    SANDBOX_LIST_OPERATIONS_TOOL_ID,
)

_SANDBOX_ISOLATION = ToolIsolationRequirement.SANDBOX


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
        isolation_requirement=_SANDBOX_ISOLATION,
    )


def register_sandbox_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(sandbox_exec_contract(), SandboxExecHandler(ctx))
    registry.register(
        ToolContract(
            tool_id=CODE_EXEC_TOOL_ID,
            name=CODE_EXEC_TOOL_ID,
            description="Execute Python code inside an isolated sandbox session (run_python operation).",
            description_short="Run Python in sandbox.",
            input_schema=CodeExecInput,
            output_schema=SandboxExecOutput,
            error_mapping={},
            side_effects=True,
            category="sandbox",
            risk_level=ToolRiskLevel.HIGH,
            tags=("sandbox", "code", "execution"),
            isolation_requirement=_SANDBOX_ISOLATION,
        ),
        CodeExecHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=SCRIPT_RUN_TOOL_ID,
            name=SCRIPT_RUN_TOOL_ID,
            description="Run an allowlisted script file inside the sandbox root directory.",
            description_short="Run sandbox script.",
            input_schema=ScriptRunInput,
            output_schema=SandboxExecOutput,
            error_mapping={},
            side_effects=True,
            category="sandbox",
            risk_level=ToolRiskLevel.HIGH,
            tags=("sandbox", "script", "execution"),
            isolation_requirement=_SANDBOX_ISOLATION,
        ),
        ScriptRunHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=BROWSER_RUN_TOOL_ID,
            name=BROWSER_RUN_TOOL_ID,
            description="Fetch a web page via BrowserAutomation integration or sandbox browser_fetch fallback.",
            description_short="Run browser fetch.",
            input_schema=BrowserRunInput,
            output_schema=BrowserRunOutput,
            error_mapping={},
            side_effects=True,
            category="sandbox",
            risk_level=ToolRiskLevel.HIGH,
            tags=("sandbox", "browser", "execution"),
            isolation_requirement=_SANDBOX_ISOLATION,
        ),
        BrowserRunHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=SANDBOX_LIST_OPERATIONS_TOOL_ID,
            name=SANDBOX_LIST_OPERATIONS_TOOL_ID,
            description="List allowlisted operations for the active sandbox session.",
            description_short="List sandbox operations.",
            input_schema=SandboxListOperationsInput,
            output_schema=SandboxListOperationsOutput,
            error_mapping={},
            side_effects=False,
            category="sandbox",
            risk_level=ToolRiskLevel.LOW,
            tags=("sandbox", "introspection"),
        ),
        SandboxListOperationsHandler(ctx),
    )


SANDBOX_EXEC_TOOL_CONTRACT = sandbox_exec_contract()
