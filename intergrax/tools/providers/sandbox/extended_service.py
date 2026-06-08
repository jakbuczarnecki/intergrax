# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import sys

from intergrax.tools.providers.sandbox._session import resolve_sandbox_session, run_sandbox_operation
from intergrax.tools.providers.sandbox.contracts import (
    BrowserRunInput,
    BrowserRunOutput,
    CodeExecInput,
    SandboxExecOutput,
    SandboxListOperationsInput,
    SandboxListOperationsOutput,
    ScriptRunInput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

CODE_EXEC_TOOL_ID = "code.exec"
SCRIPT_RUN_TOOL_ID = "script.run"
BROWSER_RUN_TOOL_ID = "browser.run"
SANDBOX_LIST_OPERATIONS_TOOL_ID = "sandbox.list_operations"


def code_exec(ctx: ToolWiringContext, params: CodeExecInput) -> SandboxExecOutput:
    return run_sandbox_operation(
        ctx,
        "run_python",
        {
            "code": params.code,
            "language": params.language,
            "timeout_s": params.timeout_s,
        },
    )


def script_run(ctx: ToolWiringContext, params: ScriptRunInput) -> SandboxExecOutput:
    interpreter = params.interpreter.strip() or sys.executable
    return run_sandbox_operation(
        ctx,
        "run_script",
        {
            "path": params.path,
            "args": list(params.args),
            "interpreter": interpreter,
            "timeout_s": params.timeout_s,
        },
    )


def browser_run(ctx: ToolWiringContext, params: BrowserRunInput) -> BrowserRunOutput:
    automation = ctx.browser_automation
    if automation is not None:
        try:
            page = automation.fetch_page(params.url.strip(), wait_until=params.wait_until)
            content = page.text or page.html or ""
            if len(content) > params.max_chars:
                content = content[: params.max_chars]
            return BrowserRunOutput(
                success=True,
                url=params.url.strip(),
                title=page.title or "",
                content=content,
            )
        except Exception as exc:  # noqa: BLE001 — tool boundary
            return BrowserRunOutput(success=False, url=params.url.strip(), error=str(exc))

    result = run_sandbox_operation(
        ctx,
        "browser_fetch",
        {
            "url": params.url.strip(),
            "max_chars": params.max_chars,
            "timeout_s": params.timeout_s,
        },
    )
    if not result.success:
        return BrowserRunOutput(success=False, url=params.url.strip(), error=result.error, session_id=result.session_id)
    output = result.output
    return BrowserRunOutput(
        success=True,
        url=str(output.get("url") or params.url.strip()),
        title="",
        content=str(output.get("content") or ""),
        session_id=result.session_id,
    )


def sandbox_list_operations(
    ctx: ToolWiringContext,
    params: SandboxListOperationsInput,
) -> SandboxListOperationsOutput:
    _ = params
    session = resolve_sandbox_session(ctx)
    if session is None:
        raise RuntimeError("sandbox_session_not_configured")
    manifest = session.manifest()
    return SandboxListOperationsOutput(
        session_id=manifest.session_id,
        operations=list(manifest.allowed_operations),
    )
