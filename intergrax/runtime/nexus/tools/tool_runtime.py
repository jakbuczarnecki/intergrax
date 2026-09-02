# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

from intergrax.tools.unified.constants import (
    RAG_RETRIEVE_TOOL_ID,
    RAG_TOOL_ALIASES,
    WEBSEARCH_QUERY_TOOL_ID,
    WEBSEARCH_TOOL_ALIASES,
)

if TYPE_CHECKING:
    from intergrax.contracts.tool_request import ToolRequest, ToolResponse
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def plan_includes_rag(tool_ids: Sequence[str]) -> bool:
    return any(
        tool_id in RAG_TOOL_ALIASES or tool_id.startswith("rag.")
        for tool_id in tool_ids
    )


def plan_includes_websearch(tool_ids: Sequence[str]) -> bool:
    return any(
        tool_id in WEBSEARCH_TOOL_ALIASES or tool_id.startswith("websearch.")
        for tool_id in tool_ids
    )


@dataclass(frozen=True, slots=True)
class ToolInvocationPlan:
    """
    Runtime-neutral plan for capability invocation.

    Canonical: ``tool_ids`` lists catalog tools (e.g. ``rag.retrieve``).
    ``use_tools`` enables the bounded tool planner loop when configured.
    """

    tool_ids: tuple[str, ...] = ()
    use_tools: bool = False
    tool_inputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def normalized(self) -> ToolInvocationPlan:
        deduped = tuple(dict.fromkeys(self.tool_ids))
        if deduped == self.tool_ids:
            return self
        return ToolInvocationPlan(
            tool_ids=deduped,
            use_tools=self.use_tools,
            tool_inputs=dict(self.tool_inputs),
        )

    @classmethod
    def from_tool_ids(cls, tool_ids: Sequence[str], *, use_tools: bool = False) -> ToolInvocationPlan:
        return cls(tool_ids=tuple(tool_ids), use_tools=use_tools).normalized()


def capability_payload_to_tool_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """
    Resolve catalog tool ids from a capability-plan payload (Phase LEG-1).

    Prefers explicit ``tool_ids``; otherwise maps legacy boolean flags to catalog ids.
    """
    raw_ids = payload.get("tool_ids")
    if isinstance(raw_ids, (list, tuple)) and raw_ids:
        return tuple(dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip()))

    ids: list[str] = []
    if bool(payload.get("use_rag", False)):
        ids.append(RAG_RETRIEVE_TOOL_ID)
    if bool(payload.get("use_websearch", False)):
        ids.append(WEBSEARCH_QUERY_TOOL_ID)
    return tuple(dict.fromkeys(ids))


def _tool_inputs_from_payload(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("tool_inputs")
    if not isinstance(raw, Mapping):
        return {}
    parsed: dict[str, dict[str, Any]] = {}
    for tool_id, tool_input in raw.items():
        if isinstance(tool_input, Mapping):
            parsed[str(tool_id)] = dict(tool_input)
    return parsed


def tool_invocation_plan_from_capability_payload(payload: Mapping[str, Any]) -> ToolInvocationPlan:
    """Build a normalized plan from gateway/capability payload."""
    plan = ToolInvocationPlan.from_tool_ids(
        capability_payload_to_tool_ids(payload),
        use_tools=bool(payload.get("use_tools", False)),
    )
    tool_inputs = _tool_inputs_from_payload(payload)
    if not tool_inputs:
        return plan
    return ToolInvocationPlan(
        tool_ids=plan.tool_ids,
        use_tools=plan.use_tools,
        tool_inputs=tool_inputs,
    )


@dataclass(frozen=True, slots=True)
class ToolRuntimeResult:
    used_rag: bool
    used_websearch: bool
    used_tools: bool
    tool_trace_count: int
    tool_ids: tuple[str, ...] = ()


@runtime_checkable
class ToolPlanLike(Protocol):
    use_rag: bool
    use_websearch: bool
    use_tools: bool

    @property
    def tool_ids(self) -> Sequence[str]:
        ...


class ToolRuntime:
    """
    Tier-1 runtime primitive for invoking Nexus capability steps.

    Agents declare tool needs via contract; runtime executes catalog tools and
    context injection (RAG / websearch / tools planner).
    """

    @staticmethod
    def plan_from_like(source: ToolPlanLike) -> ToolInvocationPlan:
        tool_ids = list(source.tool_ids)
        if source.use_rag and RAG_RETRIEVE_TOOL_ID not in tool_ids:
            tool_ids.append(RAG_RETRIEVE_TOOL_ID)
        if source.use_websearch and WEBSEARCH_QUERY_TOOL_ID not in tool_ids:
            tool_ids.append(WEBSEARCH_QUERY_TOOL_ID)
        return ToolInvocationPlan.from_tool_ids(
            tool_ids,
            use_tools=bool(source.use_tools),
        )

    @staticmethod
    async def invoke(
        *,
        state: "RuntimeState",
        plan: ToolInvocationPlan,
        trace_step: str = "ToolRuntime",
        allowed_tools: Optional[Sequence[str]] = None,
    ) -> ToolRuntimeResult:
        from intergrax.runtime.nexus.tools.plan_context_invocation import (
            run_rag_context,
            run_tools_context,
            run_websearch_context,
        )
        from intergrax.runtime.nexus.context.memory_context_invocation import (
            run_longterm_memory_context,
            run_session_semantic_recall_context,
        )
        from intergrax.runtime.nexus.context.runtime_state_handle_bridge import (
            merge_provider_metadata_into_request,
        )
        from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
        from intergrax.runtime.policy.tool_policy_resolution import resolve_allowed_tools_from_config
        from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel

        cfg = state.context.config
        incoming_plan = plan.normalized()
        effective_allowed = resolve_allowed_tools_from_config(cfg, explicit=allowed_tools)
        has_authoritative_scope = (
            allowed_tools is not None
            or effective_allowed is not None
            or bool(incoming_plan.tool_ids)
        )
        authoritative_empty_scope = (
            has_authoritative_scope
            and effective_allowed is not None
            and len(effective_allowed) == 0
        )
        plan = ToolAccessPolicy.apply(
            incoming_plan,
            allowed_tools=effective_allowed,
            state=state,
        )
        scope_policy = cfg.tool_scope_policy
        if scope_policy is not None:
            plan = ToolAccessPolicy.apply_scope_policy(
                plan,
                scope_policy=scope_policy,
                agent_id=state.request.agent_id,
                state=state,
            )
        modality_profile = cfg.modality_profile
        if modality_profile is not None:
            plan = ToolAccessPolicy.apply_modality_profile(plan, profile=modality_profile)

        await run_longterm_memory_context(state)
        await run_session_semantic_recall_context(state)
        merge_provider_metadata_into_request(state)

        if plan_includes_rag(plan.tool_ids):
            if cfg.enable_rag:
                await run_rag_context(state)
            else:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=trace_step,
                    message="Plan requested RAG but enable_rag is false; skipping RagStep.",
                    level=TraceLevel.WARNING,
                )

        if plan_includes_websearch(plan.tool_ids):
            if cfg.enable_websearch:
                await run_websearch_context(state)
            else:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=trace_step,
                    message="Plan requested websearch but enable_websearch is false; skipping.",
                    level=TraceLevel.WARNING,
                )

        if plan.use_tools or (incoming_plan.use_tools and authoritative_empty_scope):
            if cfg.tool_planner and cfg.tool_invoker and cfg.tools_mode != "off":
                from intergrax.runtime.nexus.tools.catalog_dispatch import catalog_tool_ids

                planner_constraints = catalog_tool_ids(plan.tool_ids)
                previous_constraints = state.tool_planner_allowed_tool_ids
                if has_authoritative_scope:
                    state.tool_planner_allowed_tool_ids = planner_constraints
                try:
                    await run_tools_context(state)
                finally:
                    state.tool_planner_allowed_tool_ids = previous_constraints
            else:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=trace_step,
                    message="Plan requested tools but tools are off or not configured; skipping.",
                    level=TraceLevel.WARNING,
                )

        from intergrax.runtime.nexus.tools.catalog_dispatch import (
            catalog_tool_ids,
            invoke_catalog_tool_ids,
        )

        direct_catalog_ids = catalog_tool_ids(plan.tool_ids)
        if direct_catalog_ids:
            invoke_catalog_tool_ids(
                state=state,
                tool_ids=direct_catalog_ids,
                tool_inputs=plan.tool_inputs,
                trace_step=trace_step,
            )

        return ToolRuntimeResult(
            used_rag=state.used_rag,
            used_websearch=state.used_websearch,
            used_tools=state.used_tools,
            tool_trace_count=len(state.tool_traces or []),
            tool_ids=plan.tool_ids,
        )

    @staticmethod
    async def invoke_request(
        *,
        state: "RuntimeState",
        request: ToolRequest,
        allowed_tools: Optional[Sequence[str]] = None,
        trace_step: str = "ToolRuntime",
    ) -> ToolResponse:
        """§42.12 gateway entry — prefer over direct ``invoke`` from agent code."""
        from intergrax.runtime.nexus.tools.tool_gateway import RuntimeToolGateway
        from intergrax.runtime.policy.tool_policy_resolution import resolve_allowed_tools_from_config

        cfg = state.context.config
        effective_allowed = resolve_allowed_tools_from_config(cfg, explicit=allowed_tools)
        gateway = RuntimeToolGateway.for_state(
            state,
            allowed_tools=effective_allowed,
            trace_step=trace_step,
        )
        return await gateway.invoke(request)
