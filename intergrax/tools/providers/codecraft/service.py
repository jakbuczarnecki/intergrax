# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""codecraft.run service — static gate + sandbox delegate (ECC-1)."""

from __future__ import annotations

import time
from uuid import uuid4

from intergrax.codecraft.contracts import CodeCraftRunInput, CraftResult, StaticGateResult
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.codecraft.static_gate import StaticCodeGate
from intergrax.runtime.codecraft.trace import CodeCraftTraceEmitter
from intergrax.tools.providers.codecraft.contracts import CodeCraftRunToolInput, CodeCraftRunToolOutput
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from intergrax.tools.providers.sandbox.extended_service import code_exec
from intergrax.tools.providers.sandbox._session import resolve_sandbox_session
from intergrax.tools.registry.wiring import ToolWiringContext

CODECRAFT_RUN_TOOL_ID = "codecraft.run"


def resolve_codecraft_profile(ctx: ToolWiringContext) -> CodeCraftProfile | None:
    raw = ctx.extras.get("codecraft_profile")
    if raw is None:
        return None
    if isinstance(raw, CodeCraftProfile):
        return raw
    if isinstance(raw, dict):
        return CodeCraftProfile.model_validate(raw)
    return None


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
