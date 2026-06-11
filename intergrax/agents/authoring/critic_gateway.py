# © Artur Czarnecki. All rights reserved.

"""ACP critic gateway — Tier-2 calls CVL via harness hooks only (ACP-CLOSE-PAT-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.critic.contracts import CriticAction, CriticVerdict
from intergrax.runtime.critic.critic_wiring import (
    CriticGraphHooks,
    validate_uaep_step_with_critic_detail,
)


@dataclass(frozen=True, slots=True)
class ReflectionCriticOutcome:
    """Normalized CVL result for reflection pattern authors."""

    passed: bool
    action: CriticAction
    summary: str
    verdict: CriticVerdict


def resolve_critic_hooks(step_ctx: AgentStepContext) -> CriticGraphHooks | None:
    raw = step_ctx.metadata.get(AcpRunContextKey.CRITIC_HOOKS)
    if isinstance(raw, CriticGraphHooks):
        return raw
    return None


def _resolve_tenant_id(step_ctx: AgentStepContext) -> str:
    tenant = step_ctx.metadata.get(AcpRunContextKey.TENANT_ID)
    if isinstance(tenant, str) and tenant:
        return tenant
    org = step_ctx.metadata.get(AcpRunContextKey.ORGANIZATIONAL)
    if isinstance(org, dict):
        nested = org.get("tenant_id")
        if isinstance(nested, str) and nested:
            return nested
    return "default"


def verify_reflection_draft(
    step_ctx: AgentStepContext,
    *,
    contract: AgentContract,
    draft: str,
    step_id: str = "reflection_critique",
) -> ReflectionCriticOutcome | None:
    """
    Run CVL partial verification on a reflection draft.

    Returns ``None`` when host did not wire critic hooks — authors may fall back to
    domain critique in ``act`` / ``evaluate``.
    """
    hooks = resolve_critic_hooks(step_ctx)
    if hooks is None or not draft.strip():
        return None

    execution = AgentExecutionResult(
        agent_id=contract.id,
        run_id=step_ctx.run_id,
        status=AgentExecutionStatus.COMPLETED,
        summary=draft,
        structured_data={"draft": draft, "phase": "critique"},
    )
    _validation, verdict = validate_uaep_step_with_critic_detail(
        execution,
        contract=contract,
        hooks=hooks,
        run_id=step_ctx.run_id,
        tenant_id=_resolve_tenant_id(step_ctx),
        step_id=step_id,
        extra_context={"reflection_draft": draft},
    )
    summary = "; ".join(verdict.failure_reasons) if verdict.failure_reasons else draft[:240]
    return ReflectionCriticOutcome(
        passed=verdict.passed,
        action=verdict.recommended_action,
        summary=summary,
        verdict=verdict,
    )


def critic_hooks_attached(step_ctx: AgentStepContext) -> bool:
    return resolve_critic_hooks(step_ctx) is not None
