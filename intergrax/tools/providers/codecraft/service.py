# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""codecraft.* catalog tool services (ECC-1+)."""

from __future__ import annotations

import time
from uuid import uuid4

from intergrax.codecraft.contracts import CodeCraftRunInput, CraftResult, StaticGateResult
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.codecraft.static_gate import StaticCodeGate
from intergrax.runtime.codecraft.ephemeral_registry import get_ephemeral_registry_store
from intergrax.runtime.codecraft.orchestrator import CodeCraftOrchestrator, resolve_codecraft_profile
from intergrax.runtime.codecraft.trace import CodeCraftTraceEmitter
from intergrax.tools.providers.codecraft.contracts import (
    CodeCraftDisposeToolInput,
    CodeCraftDisposeToolOutput,
    CodeCraftGetStateToolInput,
    CodeCraftGetStateToolOutput,
    CodeCraftIterateToolInput,
    CodeCraftIterateToolOutput,
    CodeCraftListEphemeralToolsInput,
    CodeCraftListEphemeralToolsOutput,
    CodeCraftPromoteToolInput,
    CodeCraftPromoteToolOutput,
    CodeCraftRunToolInput,
    CodeCraftRunToolOutput,
    CodeCraftStartToolInput,
    CodeCraftStartToolOutput,
)
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from intergrax.tools.providers.sandbox.extended_service import code_exec
from intergrax.tools.providers.sandbox._session import resolve_sandbox_session
from intergrax.tools.registry.wiring import ToolWiringContext

CODECRAFT_RUN_TOOL_ID = "codecraft.run"
CODECRAFT_START_TOOL_ID = "codecraft.start"
CODECRAFT_ITERATE_TOOL_ID = "codecraft.iterate"
CODECRAFT_GET_STATE_TOOL_ID = "codecraft.get_state"
CODECRAFT_DISPOSE_TOOL_ID = "codecraft.dispose"
CODECRAFT_PROMOTE_TOOL_ID = "codecraft.promote"
CODECRAFT_LIST_EPHEMERAL_TOOLS_TOOL_ID = "codecraft.list_ephemeral_tools"

CODECRAFT_TOOL_IDS: tuple[str, ...] = (
    CODECRAFT_RUN_TOOL_ID,
    CODECRAFT_START_TOOL_ID,
    CODECRAFT_ITERATE_TOOL_ID,
    CODECRAFT_GET_STATE_TOOL_ID,
    CODECRAFT_DISPOSE_TOOL_ID,
    CODECRAFT_PROMOTE_TOOL_ID,
    CODECRAFT_LIST_EPHEMERAL_TOOLS_TOOL_ID,
)


def _orchestrator(ctx: ToolWiringContext, run_id: str | None) -> CodeCraftOrchestrator:
    return CodeCraftOrchestrator(ctx, run_id=run_id or f"craft_run_{uuid4().hex[:12]}")


def codecraft_start(ctx: ToolWiringContext, params: CodeCraftStartToolInput) -> CodeCraftStartToolOutput:
    orch = _orchestrator(ctx, params.run_id)
    session, deny = orch.start(
        goal=params.goal,
        task_id=params.task_id,
        tenant_id=params.tenant_id,
        agent_id=params.agent_id,
        constraints=params.constraints,
        craft_id=params.craft_id,
        initial_code=params.initial_code,
        language=params.language,
    )
    if deny is not None:
        return CodeCraftStartToolOutput(error=deny.error, trace_event_count=orch.trace_event_count)
    return CodeCraftStartToolOutput(session=session, trace_event_count=orch.trace_event_count)


def codecraft_iterate(ctx: ToolWiringContext, params: CodeCraftIterateToolInput) -> CodeCraftIterateToolOutput:
    orch = _orchestrator(ctx, params.run_id)
    session, result = orch.iterate(
        craft_id=params.craft_id,
        task_id=params.task_id,
        tenant_id=params.tenant_id,
        agent_id=params.agent_id,
        patch_diagnostics=params.patch_diagnostics,
        hitl_approved=params.hitl_approved,
        timeout_s=params.timeout_s,
    )
    return CodeCraftIterateToolOutput(
        session=session,
        result=result,
        trace_event_count=orch.trace_event_count,
    )


def codecraft_get_state(ctx: ToolWiringContext, params: CodeCraftGetStateToolInput) -> CodeCraftGetStateToolOutput:
    orch = _orchestrator(ctx, params.run_id)
    session = orch.get_state(params.craft_id)
    return CodeCraftGetStateToolOutput(session=session, found=session is not None)


def codecraft_dispose(ctx: ToolWiringContext, params: CodeCraftDisposeToolInput) -> CodeCraftDisposeToolOutput:
    orch = _orchestrator(ctx, params.run_id)
    disposed = orch.dispose(
        params.craft_id,
        tenant_id=params.tenant_id,
        task_id=params.task_id,
        agent_id=params.agent_id,
    )
    return CodeCraftDisposeToolOutput(
        disposed=disposed is not None,
        craft_id=params.craft_id,
        trace_event_count=orch.trace_event_count,
    )


def codecraft_promote(ctx: ToolWiringContext, params: CodeCraftPromoteToolInput) -> CodeCraftPromoteToolOutput:
    orch = _orchestrator(ctx, params.run_id)
    result = orch.promote(params.craft_id)
    return CodeCraftPromoteToolOutput(result=result)


def codecraft_list_ephemeral_tools(
    ctx: ToolWiringContext,
    params: CodeCraftListEphemeralToolsInput,
) -> CodeCraftListEphemeralToolsOutput:
    registry = get_ephemeral_registry_store(ctx).for_craft(params.craft_id)
    return CodeCraftListEphemeralToolsOutput(
        craft_id=params.craft_id,
        tool_ids=list(registry.list_tools()),
    )


def codecraft_run(ctx: ToolWiringContext, params: CodeCraftRunToolInput) -> CodeCraftRunToolOutput:
    profile = resolve_codecraft_profile(ctx)
    craft_id = params.craft_id or f"craft_{uuid4().hex[:12]}"
    run_id = params.run_id or craft_id
    emitter = CodeCraftTraceEmitter(run_id=run_id)

    if profile is None:
        gate = StaticGateResult(passed=False, rule_ids=["profile_missing"], message="codecraft_profile_missing")
        return CodeCraftRunToolOutput(
            result=CraftResult(
                craft_id=craft_id,
                success=False,
                mode="disabled",
                static_gate=gate,
                error="codecraft_profile_missing",
                verdict="abort",
            ),
        )

    emitter.session_opened(
        craft_id=craft_id,
        mode=profile.mode,
        tenant_id=params.tenant_id,
        task_id=params.task_id,
        agent_id=params.agent_id,
    )

    if profile.mode == "disabled":
        gate = StaticGateResult(passed=False, rule_ids=["mode_disabled"], message="codecraft mode disabled")
        emitter.static_gate(
            craft_id=craft_id,
            mode=profile.mode,
            passed=False,
            rule_ids=("mode_disabled",),
            tenant_id=params.tenant_id,
            task_id=params.task_id,
            agent_id=params.agent_id,
        )
        emitter.disposed(
            craft_id=craft_id,
            mode=profile.mode,
            tenant_id=params.tenant_id,
            task_id=params.task_id,
            agent_id=params.agent_id,
        )
        return CodeCraftRunToolOutput(
            result=CraftResult(
                craft_id=craft_id,
                success=False,
                mode=profile.mode,
                static_gate=gate,
                error="codecraft_mode_disabled",
                verdict="abort",
            ),
            trace_event_count=len(emitter.events),
        )

    gate = StaticCodeGate(profile).scan(params.code, language=params.language)
    emitter.static_gate(
        craft_id=craft_id,
        mode=profile.mode,
        passed=gate.passed,
        rule_ids=tuple(gate.rule_ids),
        tenant_id=params.tenant_id,
        task_id=params.task_id,
        agent_id=params.agent_id,
    )

    if not gate.passed:
        emitter.disposed(
            craft_id=craft_id,
            mode=profile.mode,
            tenant_id=params.tenant_id,
            task_id=params.task_id,
            agent_id=params.agent_id,
        )
        return CodeCraftRunToolOutput(
            result=CraftResult(
                craft_id=craft_id,
                success=False,
                mode=profile.mode,
                static_gate=gate,
                error=gate.message or "static_gate_failed",
                verdict="revise",
            ),
            trace_event_count=len(emitter.events),
        )

    if profile.mode in ("dry_run", "assist_only"):
        structured: dict[str, str] = {}
        if profile.mode == "assist_only":
            structured["code"] = params.code
        emitter.disposed(
            craft_id=craft_id,
            mode=profile.mode,
            tenant_id=params.tenant_id,
            task_id=params.task_id,
            agent_id=params.agent_id,
        )
        return CodeCraftRunToolOutput(
            result=CraftResult(
                craft_id=craft_id,
                success=True,
                mode=profile.mode,
                static_gate=gate,
                structured_output=structured,
                verdict="promote" if profile.mode == "assist_only" else "continue",
            ),
            trace_event_count=len(emitter.events),
        )

    session = resolve_sandbox_session(ctx)
    if session is None:
        emitter.disposed(
            craft_id=craft_id,
            mode=profile.mode,
            tenant_id=params.tenant_id,
            task_id=params.task_id,
            agent_id=params.agent_id,
        )
        return CodeCraftRunToolOutput(
            result=CraftResult(
                craft_id=craft_id,
                success=False,
                mode=profile.mode,
                static_gate=gate,
                error="sandbox_session_not_configured",
                verdict="abort",
            ),
            trace_event_count=len(emitter.events),
        )

    started = time.perf_counter()
    exec_out = code_exec(
        ctx,
        CodeExecInput(code=params.code, language=params.language, timeout_s=params.timeout_s),
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    stdout = str((exec_out.output or {}).get("stdout") or "")
    stderr = str((exec_out.output or {}).get("stderr") or exec_out.error or "")
    exit_code_raw = (exec_out.output or {}).get("exit_code")
    exit_code = int(exit_code_raw) if exit_code_raw is not None else None

    emitter.exec_completed(
        craft_id=craft_id,
        mode=profile.mode,
        sandbox_session_id=exec_out.session_id,
        exit_code=exit_code,
        success=exec_out.success,
        tenant_id=params.tenant_id,
        task_id=params.task_id,
        agent_id=params.agent_id,
        duration_ms=duration_ms,
    )
    emitter.disposed(
        craft_id=craft_id,
        mode=profile.mode,
        tenant_id=params.tenant_id,
        task_id=params.task_id,
        agent_id=params.agent_id,
    )

    return CodeCraftRunToolOutput(
        result=CraftResult(
            craft_id=craft_id,
            success=exec_out.success,
            mode=profile.mode,
            static_gate=gate,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            sandbox_session_id=exec_out.session_id,
            error="" if exec_out.success else stderr,
            structured_output={"stdout": stdout} if exec_out.success else {},
            verdict="promote" if exec_out.success else "revise",
        ),
        trace_event_count=len(emitter.events),
    )


def codecraft_run_internal(ctx: ToolWiringContext, params: CodeCraftRunInput) -> CraftResult:
    tool_out = codecraft_run(
        ctx,
        CodeCraftRunToolInput(
            code=params.code,
            goal=params.goal,
            language=params.language,
            timeout_s=params.timeout_s,
            craft_id=params.craft_id,
        ),
    )
    return tool_out.result
