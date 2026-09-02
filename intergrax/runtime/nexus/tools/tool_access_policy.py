# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from intergrax.runtime.modality.modality_profile import ModalityProfile, filter_tool_ids_by_modality_profile
from intergrax.runtime.tools.scope_policy import ToolScopePolicy
from intergrax.runtime.nexus.tools.tool_runtime import (
    ToolInvocationPlan,
    plan_includes_rag,
    plan_includes_websearch,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.tools.unified.constants import (
    RAG_RETRIEVE_TOOL_ID,
    RAG_TOOL_ALIASES,
    WEBSEARCH_QUERY_TOOL_ID,
    WEBSEARCH_TOOL_ALIASES,
)


class TraceEmittingRuntimeState(Protocol):
    def trace_event(
        self,
        *,
        component: TraceComponent,
        step: str,
        message: str,
        level: TraceLevel,
    ) -> None: ...


class ToolAccessPolicy:
    """
    Enforces agent contract tool permissions before ToolRuntime.invoke (§10.8, Phase B.6).
    """

    @staticmethod
    def apply(
        plan: ToolInvocationPlan,
        *,
        allowed_tools: Optional[Sequence[str]],
        state: Optional[TraceEmittingRuntimeState] = None,
    ) -> ToolInvocationPlan:
        normalized = plan.normalized()
        if allowed_tools is None:
            return normalized

        requested_rag = plan_includes_rag(normalized.tool_ids)
        requested_websearch = plan_includes_websearch(normalized.tool_ids)
        requested_tools = normalized.use_tools

        filtered_ids = tuple(
            tool_id
            for tool_id in normalized.tool_ids
            if ToolAccessPolicy.is_tool_allowed(tool_id, allowed_tools)
        )
        filtered_has_rag = plan_includes_rag(filtered_ids)
        filtered_has_websearch = plan_includes_websearch(filtered_ids)
        use_tools = requested_tools and len(set(allowed_tools)) > 0

        if requested_rag and not filtered_has_rag:
            ToolAccessPolicy._emit_denied(
                state,
                message=f"RAG denied: {RAG_RETRIEVE_TOOL_ID!r} not in allowed_tools.",
            )
        if requested_websearch and not filtered_has_websearch:
            ToolAccessPolicy._emit_denied(
                state,
                message=f"Websearch denied: {WEBSEARCH_QUERY_TOOL_ID!r} not in allowed_tools.",
            )
        if requested_tools and not use_tools:
            ToolAccessPolicy._emit_denied(
                state,
                message="Tool invocation denied: agent allowed_tools is empty.",
            )

        filtered_inputs = {
            tool_id: dict(inputs)
            for tool_id, inputs in normalized.tool_inputs.items()
            if tool_id in filtered_ids
        }
        return ToolInvocationPlan(
            tool_ids=filtered_ids,
            use_tools=use_tools,
            tool_inputs=filtered_inputs,
        )

    @staticmethod
    def apply_scope_policy(
        plan: ToolInvocationPlan,
        *,
        scope_policy: ToolScopePolicy,
        agent_id: str,
        state: Optional[TraceEmittingRuntimeState] = None,
    ) -> ToolInvocationPlan:
        """Narrow plan using dynamic ToolScopePolicy for the requesting agent."""
        normalized = plan.normalized()
        requested_rag = plan_includes_rag(normalized.tool_ids)
        requested_websearch = plan_includes_websearch(normalized.tool_ids)

        filtered_ids = tuple(
            tool_id
            for tool_id in normalized.tool_ids
            if ToolAccessPolicy._is_allowed_by_scope_policy(
                scope_policy,
                agent_id=agent_id,
                tool_id=tool_id,
            )
        )
        filtered_has_rag = plan_includes_rag(filtered_ids)
        filtered_has_websearch = plan_includes_websearch(filtered_ids)

        if requested_rag and not filtered_has_rag:
            ToolAccessPolicy._emit_denied(
                state,
                message=f"RAG denied by scope policy: {RAG_RETRIEVE_TOOL_ID!r}.",
            )
        if requested_websearch and not filtered_has_websearch:
            ToolAccessPolicy._emit_denied(
                state,
                message=f"Websearch denied by scope policy: {WEBSEARCH_QUERY_TOOL_ID!r}.",
            )

        filtered_inputs = {
            tool_id: dict(inputs)
            for tool_id, inputs in normalized.tool_inputs.items()
            if tool_id in filtered_ids
        }
        return ToolInvocationPlan(
            tool_ids=filtered_ids,
            use_tools=normalized.use_tools,
            tool_inputs=filtered_inputs,
        )

    @staticmethod
    def apply_modality_profile(
        plan: ToolInvocationPlan,
        *,
        profile: ModalityProfile,
    ) -> ToolInvocationPlan:
        """Intersect tool plan with modality plane policy (Phase W-ML.6)."""
        normalized = plan.normalized()
        filtered = filter_tool_ids_by_modality_profile(normalized.tool_ids, profile)
        filtered_inputs = {
            tool_id: dict(inputs)
            for tool_id, inputs in normalized.tool_inputs.items()
            if tool_id in filtered
        }
        return ToolInvocationPlan(
            tool_ids=filtered,
            use_tools=normalized.use_tools,
            tool_inputs=filtered_inputs,
        )

    @staticmethod
    def _allows_rag(allowed: set[str]) -> bool:
        return bool(allowed.intersection(RAG_TOOL_ALIASES))

    @staticmethod
    def _allows_websearch(allowed: set[str]) -> bool:
        return bool(allowed.intersection(WEBSEARCH_TOOL_ALIASES))

    @staticmethod
    def _is_allowed_by_scope_policy(
        scope_policy: ToolScopePolicy,
        *,
        agent_id: str,
        tool_id: str,
    ) -> bool:
        if tool_id in RAG_TOOL_ALIASES:
            return scope_policy.is_allowed(agent_id=agent_id, tool_id=RAG_RETRIEVE_TOOL_ID)
        if tool_id in WEBSEARCH_TOOL_ALIASES:
            return scope_policy.is_allowed(agent_id=agent_id, tool_id=WEBSEARCH_QUERY_TOOL_ID)
        return scope_policy.is_allowed(agent_id=agent_id, tool_id=tool_id)

    @staticmethod
    def is_tool_allowed(tool_name: str, allowed_tools: Optional[Sequence[str]]) -> bool:
        if allowed_tools is None:
            return True
        if not allowed_tools:
            return False
        allowed = set(allowed_tools)
        if tool_name in allowed:
            return True
        if tool_name in RAG_TOOL_ALIASES:
            return ToolAccessPolicy._allows_rag(allowed)
        if tool_name in WEBSEARCH_TOOL_ALIASES:
            return ToolAccessPolicy._allows_websearch(allowed)
        return tool_name in allowed

    @staticmethod
    def _emit_denied(
        state: Optional[TraceEmittingRuntimeState], *, message: str
    ) -> None:
        if state is None:
            return
        state.trace_event(
            component=TraceComponent.POLICY,
            step="ToolAccessPolicy",
            message=message,
            level=TraceLevel.WARNING,
        )
