# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, runtime_checkable

from intergrax.runtime.nexus.planning.engine_plan_models import EnginePlan

from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID

if TYPE_CHECKING:
    from intergrax.contracts.tool_request import ToolRequest, ToolResponse
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


@dataclass(frozen=True, slots=True)
class ToolInvocationPlan:
    """
    Runtime-neutral plan for capability invocation.

    Canonical: ``tool_ids`` lists catalog tools (e.g. ``rag.retrieve``).
    Legacy booleans (``use_rag``, ``use_websearch``, ``use_tools``) remain as
    compatibility shims — normalized via :meth:`normalized`.
    """

    tool_ids: tuple[str, ...] = ()
    use_rag: bool = False
    use_websearch: bool = False
    use_tools: bool = False

    def normalized(self) -> ToolInvocationPlan:
        ids = list(self.tool_ids)
        use_rag = self.use_rag
        use_websearch = self.use_websearch

        if use_rag and RAG_RETRIEVE_TOOL_ID not in ids:
            ids.append(RAG_RETRIEVE_TOOL_ID)
        if use_websearch and WEBSEARCH_QUERY_TOOL_ID not in ids:
            ids.append(WEBSEARCH_QUERY_TOOL_ID)

        if RAG_RETRIEVE_TOOL_ID in ids:
            use_rag = True
        if WEBSEARCH_QUERY_TOOL_ID in ids:
            use_websearch = True

        deduped = tuple(dict.fromkeys(ids))
        return ToolInvocationPlan(
            tool_ids=deduped,
            use_rag=use_rag,
            use_websearch=use_websearch,
            use_tools=self.use_tools,
        )

    @classmethod
    def from_legacy(
        cls,
        *,
        use_rag: bool = False,
        use_websearch: bool = False,
        use_tools: bool = False,
        tool_ids: Sequence[str] = (),
    ) -> ToolInvocationPlan:
        if (use_rag or use_websearch) and not tool_ids:
            warnings.warn(
                "ToolInvocationPlan.from_legacy(use_rag/use_websearch) is deprecated; "
                "pass explicit tool_ids (e.g. rag.retrieve, websearch.query)",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            use_rag=use_rag,
            use_websearch=use_websearch,
            use_tools=use_tools,
            tool_ids=tuple(tool_ids),
        ).normalized()

    @classmethod
    def from_tool_ids(cls, tool_ids: Sequence[str], *, use_tools: bool = False) -> ToolInvocationPlan:
        return cls(tool_ids=tuple(tool_ids), use_tools=use_tools).normalized()

    def uses_legacy_booleans_only(self) -> bool:
        """True when plan was expressed via deprecated flags without explicit tool_ids."""
        return (self.use_rag or self.use_websearch) and not self.tool_ids


def capability_payload_to_tool_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """
    Resolve catalog tool ids from a capability-plan payload (Phase LEG-1).

    Prefers explicit ``tool_ids``; otherwise maps deprecated boolean flags.
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


def tool_invocation_plan_from_capability_payload(payload: Mapping[str, Any]) -> ToolInvocationPlan:
    """Build a normalized plan from gateway/capability payload without ``from_legacy``."""
    return ToolInvocationPlan.from_tool_ids(
        capability_payload_to_tool_ids(payload),
        use_tools=bool(payload.get("use_tools", False)),
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
    compatibility pipeline steps (RAG / websearch / tools planner).
    """

    @staticmethod
    def plan_from_like(source: ToolPlanLike) -> ToolInvocationPlan:
        if isinstance(source, EnginePlan):
            tool_ids = source.resolved_tool_ids()
        else:
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
        from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
        from intergrax.runtime.policy.tool_policy_resolution import resolve_allowed_tools_from_config
        from intergrax.runtime.nexus.runtime_steps.rag_step import RagStep
        from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
        from intergrax.runtime.nexus.runtime_steps.websearch_step import WebsearchStep
        from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel

        cfg = state.context.config
        effective_allowed = resolve_allowed_tools_from_config(cfg, explicit=allowed_tools)
        raw_plan = plan
        plan = ToolAccessPolicy.apply(
            plan.normalized(),
            allowed_tools=effective_allowed,
            state=state,
        )
        modality_profile = cfg.modality_profile
        if modality_profile is not None:
            plan = ToolAccessPolicy.apply_modality_profile(plan, profile=modality_profile)

        if raw_plan.uses_legacy_booleans_only():
            state.trace_event(
                component=TraceComponent.PIPELINE,
                step=trace_step,
                message=(
                    "Deprecated ToolInvocationPlan booleans (use_rag/use_websearch); "
                    "prefer tool_ids e.g. ['rag.retrieve', 'websearch.query']."
                ),
                level=TraceLevel.WARNING,
            )

        if plan.use_rag:
            if cfg.enable_rag:
                await RagStep().run(state)
            else:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=trace_step,
                    message="Plan requested RAG but enable_rag is false; skipping RagStep.",
                    level=TraceLevel.WARNING,
                )

        if plan.use_websearch:
            if cfg.enable_websearch:
                await WebsearchStep().run(state)
            else:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=trace_step,
                    message="Plan requested websearch but enable_websearch is false; skipping.",
                    level=TraceLevel.WARNING,
                )

        if plan.use_tools:
            if cfg.tool_planner and cfg.tool_invoker and cfg.tools_mode != "off":
                await ToolsStep().run(state)
            else:
                state.trace_event(
                    component=TraceComponent.PIPELINE,
                    step=trace_step,
                    message="Plan requested tools but tools are off or not configured; skipping.",
                    level=TraceLevel.WARNING,
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
