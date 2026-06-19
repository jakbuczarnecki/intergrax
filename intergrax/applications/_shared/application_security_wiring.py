# © Artur Czarnecki. All rights reserved.

"""Register per-application V-SEC hooks into Nexus middleware (Phase H-APP.2.7, V-REM-SEC)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.applications.contracts.environment_profile import ApplicationSecurityProfile
from intergrax.runtime.architecture.prompt_security import (
    PromptDefenseProfile,
    PromptInjectionRule,
    PromptRiskLevel,
    inspect_prompt_for_injection,
)
from intergrax.runtime.architecture.tenant_security import (
    SecurityAuditEvent,
    TenantIsolationCheck,
    verify_tenant_security,
)
from intergrax.runtime.architecture.tool_security import (
    ToolInvocationPolicy,
    ToolInvocationRequest,
    evaluate_tool_invocation_security,
)
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.applications._shared.security_runtime_bridge import (
    SecurityWiringOptions,
)
from intergrax.runtime.security.defense_plugin import PluginSecurityDefenseMiddleware
from intergrax.runtime.security.defense_registry import resolve_security_defense_plugins
from intergrax.runtime.security.encryption_middleware import EncryptionEnforcementMiddleware
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def default_prompt_defense_profile() -> PromptDefenseProfile:
    return PromptDefenseProfile(
        profile_id="harness.default",
        version="1",
        rules=[
            PromptInjectionRule(
                rule_id="ignore_instructions",
                pattern="ignore previous instructions",
                risk_level=PromptRiskLevel.HIGH,
                block=True,
            ),
        ],
    )


def default_tool_invocation_policy() -> ToolInvocationPolicy:
    return ToolInvocationPolicy(
        allowed_tool_ids=[],
        blocked_argument_tokens=["ignore previous instructions", "system override"],
        require_explicit_capability_match=False,
    )


class PromptDefenseMiddleware(RuntimeMiddleware):
    """Block prompts matching configured injection patterns."""

    priority = 50
    name = "PromptDefenseMiddleware"

    def __init__(self, profile: PromptDefenseProfile) -> None:
        self._profile = profile

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if point != HookPoint.BEFORE_CONTEXT_BUILD:
            return HookResult()
        prompt = str(ctx.runtime_state.get("prompt", ""))
        if not prompt:
            return HookResult()
        result = inspect_prompt_for_injection(prompt=prompt, profile=self._profile)
        if result.blocked:
            return HookResult(
                action=HookAction.BLOCK,
                reason=f"Prompt blocked: {', '.join(result.reasons)}",
            )
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()


class ToolInjectionDefenseMiddleware(RuntimeMiddleware):
    """Evaluate tool invocation requests against injection policy."""

    priority = 55
    name = "ToolInjectionDefenseMiddleware"

    def __init__(self, policy: ToolInvocationPolicy) -> None:
        self._policy = policy

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if point != HookPoint.BEFORE_TOOL_CALL:
            return HookResult()
        tool_id = str(ctx.runtime_state.get("tool_id", ""))
        if not tool_id:
            return HookResult()
        arguments = _stringify_argument_map(ctx.runtime_state.get("arguments"))
        capability_ids = _string_list(ctx.runtime_state.get("capability_ids"))
        allowed_tool_ids = _string_list(ctx.runtime_state.get("allowed_tool_ids"))
        policy = self._policy
        if allowed_tool_ids:
            policy = policy.model_copy(update={"allowed_tool_ids": allowed_tool_ids})
        decision = evaluate_tool_invocation_security(
            request=ToolInvocationRequest(
                tool_id=tool_id,
                arguments=arguments,
                capability_ids=capability_ids,
            ),
            policy=policy,
        )
        if not decision.allowed:
            return HookResult(
                action=HookAction.BLOCK,
                reason="; ".join(decision.reasons) or "tool invocation blocked",
            )
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()


class TenantSecurityMiddleware(RuntimeMiddleware):
    """Verify tenant isolation and audit trail at task intake."""

    priority = 45
    name = "TenantSecurityMiddleware"

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if point != HookPoint.BEFORE_TASK_INTAKE:
            return HookResult()
        request_tenant_id = str(ctx.runtime_state.get("tenant_id", ""))
        resource_tenant_id = str(ctx.runtime_state.get("resource_tenant_id", request_tenant_id))
        actor_id = str(ctx.runtime_state.get("user_id", "unknown"))
        if not request_tenant_id:
            return HookResult(
                action=HookAction.BLOCK,
                reason="Missing tenant_id on task intake",
            )
        check = TenantIsolationCheck(
            request_tenant_id=request_tenant_id,
            resource_tenant_id=resource_tenant_id,
            passed=request_tenant_id == resource_tenant_id,
            reason="" if request_tenant_id == resource_tenant_id else "tenant mismatch",
        )
        audit_event = SecurityAuditEvent(
            event_id=f"{ctx.run_id}:intake",
            tenant_id=request_tenant_id,
            actor_id=actor_id,
            action="task_intake",
            occurred_at=datetime.now(UTC),
        )
        report = verify_tenant_security(checks=[check], audit_events=[audit_event])
        if not report.passed:
            return HookResult(
                action=HookAction.BLOCK,
                reason="; ".join(report.reasons) or "tenant security verification failed",
            )
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()


def _attach_middleware(nexus: NexusLoop, middleware: RuntimeMiddleware) -> None:
    pipeline = nexus._middleware  # noqa: SLF001 — Tier-3 composition hook
    if isinstance(pipeline, MiddlewarePipeline):
        pipeline._middleware = sorted(  # noqa: SLF001
            [middleware, *pipeline._middleware],
            key=lambda item: item.priority,
        )


def register_application_security_hooks(
    nexus: NexusLoop,
    profile: ApplicationSecurityProfile,
    *,
    options: SecurityWiringOptions | None = None,
) -> None:
    """Attach security middleware when V-SEC toggles are enabled."""
    resolved = options
    if resolved is None:
        from intergrax.applications._shared.security_runtime_bridge import (
            resolve_security_wiring_options,
        )

        resolved = resolve_security_wiring_options(profile)
    if resolved.encryption_enforcement_enabled:
        from intergrax.runtime.security.encryption_transform import HarnessEnvelopeEncryptor

        encryptor = HarnessEnvelopeEncryptor() if resolved.secrets_store_configured else None
        _attach_middleware(
            nexus,
            EncryptionEnforcementMiddleware(
                enforcement_enabled=True,
                secrets_store_configured=resolved.secrets_store_configured,
                encryptor=encryptor,
                event_bus=nexus.event_bus,
            ),
        )
    if profile.prompt_defense_enabled:
        _attach_middleware(nexus, PromptDefenseMiddleware(default_prompt_defense_profile()))
    if profile.tool_injection_defense_enabled:
        _attach_middleware(nexus, ToolInjectionDefenseMiddleware(default_tool_invocation_policy()))
    if profile.tenant_security_verify_enabled:
        _attach_middleware(nexus, TenantSecurityMiddleware())
    for plugin in resolve_security_defense_plugins(
        resolved.defense_plugin_ids,
        resolved.defense_bundle_ids,
    ):
        _attach_middleware(nexus, PluginSecurityDefenseMiddleware(plugin, event_bus=nexus.event_bus))


def _stringify_argument_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in raw.items():
        result[str(key)] = str(value)
    return result


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item) for item in raw]
