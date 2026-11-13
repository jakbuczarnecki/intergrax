# © Artur Czarnecki. All rights reserved.

"""Shared real-path harness for the PDC-D2 qualification tests (fakes only for the LLM adapter)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.provider_descriptor import build_provider_descriptor
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    build_context_assembly_runtime_dependencies,
)


class FakeAdapter(BaseLLMAdapter):
    """External-provider stand-in; the model-facing messages are what would be sent to it."""

    provider = "fake"
    model = "fake-pdc-d2"

    def __init__(self, window: int = 8192) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:  # type: ignore[no-untyped-def]
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")

    def count_messages_tokens(self, messages: object) -> int:
        total = 0
        for message in messages:  # type: ignore[attr-defined]
            total += max(1, len(message.content or "") // 4)
        return total


class FragmentProvider:
    """Context provider returning a fixed fragment list (real engine path, no mocks)."""

    def __init__(
        self,
        provider_id: str,
        sources: Iterable[ContextFragmentSource],
        fragments: Callable[[], list[ContextFragment]],
    ) -> None:
        self.provider_id = provider_id
        self._sources = frozenset(sources)
        self._fragments = fragments

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return self._sources

    @property
    def descriptor(self):  # type: ignore[no-untyped-def]
        return build_provider_descriptor(
            self.provider_id,
            provider_version="1.0.0",
            supported_sources=self._sources,
            origin="test",
        )

    async def collect(
        self,
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        _ = request, ctx
        return self._fragments()


def make_fragment(
    fragment_id: str,
    source: ContextFragmentSource,
    content: str,
    *,
    mandatory: bool = False,
    relevance: float = 0.9,
    token_estimate: int | None = None,
    metadata: dict[str, object] | None = None,
    **extra: object,
) -> ContextFragment:
    return ContextFragment(
        fragment_id=fragment_id,
        source=source,
        source_id=fragment_id,
        content=content,
        token_estimate=token_estimate if token_estimate is not None else max(1, len(content) // 4),
        relevance_score=relevance,
        freshness_score=relevance,
        confidence_score=relevance,
        mandatory=mandatory,
        metadata=dict(metadata or {}),
        **extra,  # type: ignore[arg-type]
    )


def assembly_request(budget_tokens: int = 4000) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-pdc-d2",
        run_id="run-pdc-d2",
        task_id="task-pdc-d2",
        tenant_id="tenant-a",
        assembly_scope="graph_node",
        objective="pdc-d2 qualification",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=budget_tokens),
        assembly_options=TaskContextAssemblyOptions(),
        step_kind="model_call",
    )


def provider_context(adapter: BaseLLMAdapter, user_text: str) -> ContextProviderContext:
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=RuntimeConfig(llm_adapter=adapter, production_mode=False),
        messages=[ChatMessage(role="user", content=user_text)],
        max_output_tokens=64,
    )
    return ContextProviderContext(engine_id="pdc-d2", runtime=runtime)


def message_text(messages: Iterable[ChatMessage]) -> str:
    return "\n".join(message.content or "" for message in messages)


def tool_linkage_violations(messages: list[ChatMessage]) -> list[str]:
    """Return every orphaned tool call or tool result in a model-facing message list."""
    issues: list[str] = []
    call_ids: set[str] = set()
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                call_ids.add(str(call.get("id")))
    result_ids = {m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id}
    for call_id in sorted(call_ids - result_ids):
        issues.append(f"call without result: {call_id}")
    for result_id in sorted(result_ids - call_ids):  # type: ignore[arg-type]
        issues.append(f"result without call: {result_id}")
    return issues
