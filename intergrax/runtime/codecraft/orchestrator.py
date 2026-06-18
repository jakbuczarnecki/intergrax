# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraftOrchestrator — harness craft loop (ECC-2+)."""

from __future__ import annotations
from intergrax.utils import attribute_access

import time
from uuid import uuid4

from dataclasses import replace

from intergrax.codecraft.codegen_adapter import CodeGenerationAdapter, TemplateCodeGenerationAdapter
from intergrax.codecraft.contracts import CodeCraftSession, CraftResult, IterationRecord, StaticGateResult
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.codecraft.promoter import CraftResultPromoter
from intergrax.codecraft.static_gate import StaticCodeGate
from intergrax.codecraft.test_runner import CraftTestRunner
from intergrax.runtime.codecraft.cv_bridge import iteration_cvl_verdict
from intergrax.runtime.codecraft.ephemeral_registry import get_ephemeral_registry_store
from intergrax.runtime.codecraft.sandbox_resolver import resolve_craft_sandbox_session
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager, get_session_manager
from intergrax.runtime.codecraft.trace import CodeCraftTraceEmitter
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from intergrax.tools.providers.sandbox.extended_service import code_exec
from intergrax.tools.registry.wiring import ToolWiringContext


def resolve_codegen_adapter(ctx: ToolWiringContext) -> CodeGenerationAdapter:
    raw = ctx.extras.get("codecraft_codegen_adapter")
    if isinstance(raw, CodeGenerationAdapter):
        return raw
    return TemplateCodeGenerationAdapter()


def resolve_codecraft_profile(ctx: ToolWiringContext) -> CodeCraftProfile | None:
    raw = ctx.extras.get("codecraft_profile")
    if raw is None:
        return None
    if isinstance(raw, CodeCraftProfile):
        profile = raw
    elif isinstance(raw, dict):
        profile = CodeCraftProfile.model_validate(raw)
    else:
        return None
    task_meta = ctx.extras.get("task_metadata")
    if isinstance(task_meta, dict):
        mode_override = task_meta.get("codecraft_mode")
        if isinstance(mode_override, str) and mode_override:
            profile = profile.model_copy(update={"mode": mode_override})  # type: ignore[arg-type]
    return profile


class CodeCraftOrchestrator:
    """Single harness entry for craft session lifecycle and iteration."""

    def __init__(
        self,
        ctx: ToolWiringContext,
        *,
        run_id: str | None = None,
        session_manager: CodeCraftSessionManager | None = None,
    ) -> None:
        self._ctx = ctx
        self._run_id = run_id or f"craft_run_{uuid4().hex[:12]}"
        self._sessions = session_manager or get_session_manager(ctx)
        self._emitter = CodeCraftTraceEmitter(run_id=self._run_id)

    @property
    def trace_event_count(self) -> int:
        return len(self._emitter.events)

    def start(
        self,
        *,
        goal: str,
        task_id: str,
        tenant_id: str,
        agent_id: str = "",
        constraints: str = "",
        craft_id: str | None = None,
        initial_code: str | None = None,
        language: str = "python",
    ) -> tuple[CodeCraftSession | None, CraftResult | None]:
        profile = resolve_codecraft_profile(self._ctx)
        if profile is None:
            return None, self._deny_result("codecraft_profile_missing", craft_id=craft_id)
        if not profile.generation_allowed():
            return None, self._deny_result("codecraft_mode_disabled", craft_id=craft_id, mode=profile.mode)

        session = self._sessions.open(
            goal=goal,
            task_id=task_id,
            tenant_id=tenant_id,
            mode=profile.mode,
            language=language,
            max_iterations=profile.max_iterations,
            craft_id=craft_id,
        )
        self._emitter.session_opened(
            craft_id=session.craft_id,
            mode=profile.mode,
            tenant_id=tenant_id,
            task_id=task_id,
            agent_id=agent_id,
        )

        adapter = resolve_codegen_adapter(self._ctx)
        code = initial_code or adapter.generate(goal=goal, constraints=constraints, language=session.language)
        self._emitter.generation(
            craft_id=session.craft_id,
            mode=profile.mode,
            iteration=1,
            model_id=attribute_access.optional(adapter, "model_id", "template"),
            tenant_id=tenant_id,
            task_id=task_id,
            agent_id=agent_id,
        )
        session = session.model_copy(update={"code": code})
        ephemeral = get_ephemeral_registry_store(self._ctx).for_craft(session.craft_id)
        helper_tool_id = f"ephemeral.{session.craft_id}.helper"
        ephemeral.register(helper_tool_id)
        session = session.model_copy(
            update={"ephemeral_tool_ids": list(ephemeral.list_tools())},
        )
        self._sessions.save(session)
        return session, None

    def iterate(
        self,
        *,
        craft_id: str,
        task_id: str,
        tenant_id: str,
        agent_id: str = "",
        patch_diagnostics: str = "",
        hitl_approved: bool = False,
        timeout_s: float = 30.0,
    ) -> tuple[CodeCraftSession | None, CraftResult]:
        profile = resolve_codecraft_profile(self._ctx)
        if profile is None:
            return None, self._deny_result("codecraft_profile_missing", craft_id=craft_id)

        session = self._sessions.get(craft_id)
        if session is None or session.disposed:
            return None, self._deny_result("craft_session_not_found", craft_id=craft_id, mode=profile.mode)

        if session.iteration >= session.max_iterations:
            return session, self._deny_result("max_iterations_exceeded", craft_id=craft_id, mode=profile.mode)

        next_iteration = session.iteration + 1
        code = session.code
        if patch_diagnostics and next_iteration > 1:
            adapter = resolve_codegen_adapter(self._ctx)
            code = adapter.patch(
                goal=session.goal,
                code=code,
                diagnostics=patch_diagnostics,
                language=session.language,
            )
            self._emitter.generation(
                craft_id=craft_id,
                mode=profile.mode,
                iteration=next_iteration,
                model_id=attribute_access.optional(adapter, "model_id", "template"),
                tenant_id=tenant_id,
                task_id=task_id,
                agent_id=agent_id,
            )

        gate = StaticCodeGate(profile).scan(code, language=session.language)
        self._emitter.static_gate(
            craft_id=craft_id,
            mode=profile.mode,
            passed=gate.passed,
            rule_ids=tuple(gate.rule_ids),
            tenant_id=tenant_id,
            task_id=task_id,
            agent_id=agent_id,
        )
        if not gate.passed:
            record = IterationRecord(
                iteration=next_iteration,
                static_gate=gate,
                verdict="revise",
            )
            session = self._append_iteration(session, record, code=code)
            return session, self._craft_result_from_session(session, gate=gate, verdict="revise")

        if profile.mode in ("dry_run", "assist_only"):
            structured: dict[str, str] = {}
            if profile.mode == "assist_only":
                structured["code"] = code
            session = session.model_copy(
                update={
                    "code": code,
                    "iteration": next_iteration,
                    "structured_output": structured,
                    "status": "closed",
                },
            )
            self._sessions.save(session)
            return session, CraftResult(
                craft_id=craft_id,
                success=True,
                mode=profile.mode,
                static_gate=gate,
                structured_output=structured,
                verdict="promote" if profile.mode == "assist_only" else "continue",
            )

        needs_hitl = profile.mode == "supervised" or profile.require_hitl_before_exec
        if needs_hitl and not hitl_approved and not session.hitl_approved:
            session = session.model_copy(
                update={
                    "code": code,
                    "pending_hitl": True,
                    "status": "pending_hitl",
                },
            )
            self._sessions.save(session)
            self._emitter.hitl_requested(
                craft_id=craft_id,
                mode=profile.mode,
                reason="supervised_exec_approval",
                tenant_id=tenant_id,
                task_id=task_id,
                agent_id=agent_id,
            )
            return session, CraftResult(
                craft_id=craft_id,
                success=False,
                mode=profile.mode,
                static_gate=gate,
                error="hitl_pending",
                verdict="continue",
            )

        if profile.security_scan_before_exec and not self._security_scan_passed(code):
            gate = StaticGateResult(
                passed=False,
                rule_ids=["security_scan_failed"],
                message="security_scan_failed",
            )
            return session, CraftResult(
                craft_id=craft_id,
                success=False,
                mode=profile.mode,
                static_gate=gate,
                error="security_scan_failed",
                verdict="abort",
            )

        if profile.exec_allowed() and profile.exec_budget_exhausted(session.total_exec_time_s):
            gate = StaticGateResult(
                passed=False,
                rule_ids=["max_total_exec_time_exceeded"],
                message="max_total_exec_time_exceeded",
            )
            return session, CraftResult(
                craft_id=craft_id,
                success=False,
                mode=profile.mode,
                static_gate=gate,
                error="max_total_exec_time_exceeded",
                verdict="abort",
            )

        sandbox = resolve_craft_sandbox_session(
            self._ctx,
            profile,
            tenant_id=tenant_id,
            task_id=task_id,
        )
        if sandbox is None and profile.exec_allowed():
            return session, CraftResult(
                craft_id=craft_id,
                success=False,
                mode=profile.mode,
                static_gate=gate,
                error="sandbox_session_not_configured",
                verdict="abort",
            )

        exec_success = True
        stdout = ""
        stderr = ""
        exit_code: int | None = None
        sandbox_session_id: str | None = None
        duration_ms: float | None = None

        if profile.exec_allowed() and sandbox is not None:
            self._write_craft_file(sandbox, code)
            started = time.perf_counter()
            exec_ctx = replace(self._ctx, sandbox_session=sandbox)
            effective_timeout = min(
                timeout_s,
                profile.remaining_exec_time_s(session.total_exec_time_s),
            )
            exec_out = code_exec(
                exec_ctx,
                CodeExecInput(code=code, language=session.language, timeout_s=effective_timeout),
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            stdout = str((exec_out.output or {}).get("stdout") or "")
            stderr = str((exec_out.output or {}).get("stderr") or exec_out.error or "")
            exit_code_raw = (exec_out.output or {}).get("exit_code")
            exit_code = int(exit_code_raw) if exit_code_raw is not None else None
            exec_success = exec_out.success
            sandbox_session_id = exec_out.session_id
            self._emitter.exec_completed(
                craft_id=craft_id,
                mode=profile.mode,
                sandbox_session_id=sandbox_session_id,
                exit_code=exit_code,
                success=exec_success,
                tenant_id=tenant_id,
                task_id=task_id,
                agent_id=agent_id,
                duration_ms=duration_ms,
            )

        test_result = CraftTestRunner(profile).run(self._ctx, rel_path="craft_main.py")
        self._emitter.test_completed(
            craft_id=craft_id,
            mode=profile.mode,
            passed=None if test_result.skipped else test_result.passed,
            test_command=profile.test_command_template.format(path="craft_main.py"),
            tenant_id=tenant_id,
            task_id=task_id,
            agent_id=agent_id,
        )
        verdict = iteration_cvl_verdict(
            static_gate=gate,
            exec_success=exec_success,
            test_passed=None if test_result.skipped else test_result.passed,
            iteration=next_iteration,
            max_iterations=session.max_iterations,
        )
        self._emitter.iteration_verdict(
            craft_id=craft_id,
            mode=profile.mode,
            verdict=verdict,
            iteration=next_iteration,
            tenant_id=tenant_id,
            task_id=task_id,
            agent_id=agent_id,
        )
        record = IterationRecord(
            iteration=next_iteration,
            static_gate=gate,
            exec_success=exec_success,
            test_passed=None if test_result.skipped else test_result.passed,
            verdict=verdict,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )
        structured = {"stdout": stdout, "success": exec_success and (test_result.skipped or test_result.passed)}
        session = session.model_copy(
            update={
                "code": code,
                "iteration": next_iteration,
                "total_exec_time_s": session.total_exec_time_s + ((duration_ms or 0.0) / 1000.0),
                "pending_hitl": False,
                "hitl_approved": hitl_approved or session.hitl_approved,
                "sandbox_session_id": sandbox_session_id or session.sandbox_session_id,
                "structured_output": structured,
                "status": "closed" if verdict == "promote" else "open",
            },
        )
        session = self._append_iteration(session, record, code=code)
        return session, CraftResult(
            craft_id=craft_id,
            success=exec_success and (test_result.skipped or test_result.passed),
            mode=profile.mode,
            static_gate=gate,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            sandbox_session_id=sandbox_session_id,
            error="" if exec_success else stderr,
            structured_output=structured,
            verdict=verdict,
        )

    def get_state(self, craft_id: str) -> CodeCraftSession | None:
        return self._sessions.get(craft_id)

    def dispose(self, craft_id: str, *, tenant_id: str, task_id: str, agent_id: str = "") -> CodeCraftSession | None:
        session = self._sessions.dispose(craft_id)
        if session is None:
            return None
        get_ephemeral_registry_store(self._ctx).dispose(craft_id)
        profile = resolve_codecraft_profile(self._ctx)
        mode = profile.mode if profile is not None else "disabled"
        self._emitter.disposed(
            craft_id=craft_id,
            mode=mode,
            tenant_id=tenant_id,
            task_id=task_id,
            agent_id=agent_id,
        )
        return session

    def promote(self, craft_id: str) -> CraftResult:
        session = self._sessions.get(craft_id)
        profile = resolve_codecraft_profile(self._ctx)
        if session is None or profile is None:
            return self._deny_result("craft_session_not_found", craft_id=craft_id)
        promoter = CraftResultPromoter()
        result = promoter.promote_session(session, schema_ref=profile.promotion_schema_ref)
        self._emitter.promoted(
            craft_id=craft_id,
            mode=profile.mode,
            schema_ref=profile.promotion_schema_ref,
            tenant_id=session.tenant_id,
            task_id=session.task_id,
        )
        self._sessions.save(session.model_copy(update={"promoted": True}))
        return result

    def _append_iteration(self, session: CodeCraftSession, record: IterationRecord, *, code: str) -> CodeCraftSession:
        updated = session.model_copy(
            update={
                "code": code,
                "iterations": [*session.iterations, record],
            },
        )
        self._sessions.save(updated)
        return updated

    @staticmethod
    def _write_craft_file(sandbox, code: str) -> None:
        sandbox.execute(
            "write_file",
            {"path": "craft_main.py", "content": code},
        )

    def _security_scan_passed(self, code: str) -> bool:
        if self._ctx.security_scanner is None:
            return False
        try:
            report = self._ctx.security_scanner.scan_repo("inline://craft")
            return report.status.lower() in {"ok", "passed", "clean", "success"}
        except Exception:  # noqa: BLE001
            return "eval(" not in code and "exec(" not in code

    @staticmethod
    def _deny_result(
        error: str,
        *,
        craft_id: str | None = None,
        mode: str = "disabled",
    ) -> CraftResult:
        cid = craft_id or f"craft_{uuid4().hex[:12]}"
        gate = StaticGateResult(passed=False, rule_ids=[error], message=error)
        return CraftResult(
            craft_id=cid,
            success=False,
            mode=mode,
            static_gate=gate,
            error=error,
            verdict="abort",
        )

    @staticmethod
    def _craft_result_from_session(
        session: CodeCraftSession,
        *,
        gate: StaticGateResult,
        verdict: str,
    ) -> CraftResult:
        return CraftResult(
            craft_id=session.craft_id,
            success=False,
            mode=session.mode,
            static_gate=gate,
            error=gate.message,
            verdict=verdict,  # type: ignore[arg-type]
        )
