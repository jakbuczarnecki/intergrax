# © Artur Czarnecki. All rights reserved.

"""Failover strict tool argument conformance — DS-E2E-13B correction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    StrictToolArgumentConformanceError,
    ToolDispatchRequirements,
)
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    build_atomic_planner_round_tool_definition,
)
from testing_support.atomic_planner_round_transport import poc_business_tool_schemas

pytestmark = pytest.mark.unit


class _StrictToolsAdapter(LLMAdapter):
    provider = "strict-primary"
    model = "strict-primary"

    def __init__(self, *, fail: bool = False, label: str = "strict-primary") -> None:
        super().__init__()
        self._fail = fail
        self._label = label
        self.model = label
        self.call_config = MagicMock()
        self.call_config.retry_on_status = (429, 500, 502, 503, 504)
        self.dispatched = False

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_strict_tool_argument_conformance(self) -> bool:
        return True

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        raise NotImplementedError

    def generate_with_tools(
        self,
        messages,
        tools_schema,
        *,
        temperature=None,
        max_tokens=None,
        tool_choice=None,
        run_id=None,
        tool_dispatch_requirements=None,
    ) -> LLMAdapterResponse:
        self.dispatched = True
        if self._fail:
            exc = RuntimeError("rate limited")
            exc.status_code = 429  # type: ignore[attr-defined]
            raise exc
        return LLMAdapterResponse(
            content=f"ok-{self._label}",
            usage=LLMTokenUsage(input_tokens=1, output_tokens=1),
            model=self.model,
            provider=str(self.provider),
        )


class _StrictlessToolsAdapter(LLMAdapter):
    provider = "strictless"
    model = "strictless"

    def __init__(self, *, label: str = "strictless") -> None:
        super().__init__()
        self._label = label
        self.model = label
        self.call_config = MagicMock()
        self.dispatched = False

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        return True

    def supports_strict_tool_argument_conformance(self) -> bool:
        return False

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        raise NotImplementedError

    def generate_with_tools(
        self,
        messages,
        tools_schema,
        *,
        temperature=None,
        max_tokens=None,
        tool_choice=None,
        run_id=None,
        tool_dispatch_requirements=None,
    ) -> LLMAdapterResponse:
        self.dispatched = True
        return LLMAdapterResponse(
            content=f"ok-{self._label}",
            usage=LLMTokenUsage(input_tokens=1, output_tokens=1),
            model=self.model,
            provider=str(self.provider),
        )


def _strict_round_schema() -> tuple[list[dict[str, object]], list[ToolDispatchRequirements]]:
    definition = build_atomic_planner_round_tool_definition(poc_business_tool_schemas())
    return [dict(definition.wire_schema)], [definition.dispatch_requirements]


def test_failover_supports_strict_reflects_primary_only() -> None:
    primary = _StrictlessToolsAdapter(label="primary")
    secondary = _StrictToolsAdapter(label="secondary")
    adapter = FailoverLLMAdapter([primary, secondary])
    assert adapter.supports_strict_tool_argument_conformance() is False


def test_failover_unsupported_primary_fails_closed_before_dispatch() -> None:
    primary = _StrictlessToolsAdapter(label="primary")
    secondary = _StrictToolsAdapter(label="secondary")
    adapter = FailoverLLMAdapter([primary, secondary])
    with pytest.raises(StrictToolArgumentConformanceError, match="does not support"):
        wire_schema, dispatch_requirements = _strict_round_schema()
        adapter.generate_with_tools(
            [ChatMessage(role="user", content="plan")],
            wire_schema,
            tool_dispatch_requirements=dispatch_requirements,
        )
    assert primary.dispatched is False
    assert secondary.dispatched is False


def test_failover_supported_primary_does_not_fallback_to_unsupported() -> None:
    primary = _StrictToolsAdapter(fail=True, label="primary")
    secondary = _StrictlessToolsAdapter(label="secondary")
    adapter = FailoverLLMAdapter([primary, secondary], profile_ids=("primary", "secondary"))
    wire_schema, dispatch_requirements = _strict_round_schema()
    with pytest.raises(RuntimeError, match="rate limited"):
        adapter.generate_with_tools(
            [ChatMessage(role="user", content="plan")],
            wire_schema,
            tool_dispatch_requirements=dispatch_requirements,
        )
    assert primary.dispatched is True
    assert secondary.dispatched is False
    assert len(adapter.routing_attempts) == 1


def test_failover_both_supported_allows_fallback() -> None:
    primary = _StrictToolsAdapter(fail=True, label="primary")
    secondary = _StrictToolsAdapter(label="secondary")
    adapter = FailoverLLMAdapter([primary, secondary], profile_ids=("primary", "secondary"))
    wire_schema, dispatch_requirements = _strict_round_schema()
    response = adapter.generate_with_tools(
        [ChatMessage(role="user", content="plan")],
        wire_schema,
        tool_dispatch_requirements=dispatch_requirements,
    )
    assert response.content == "ok-secondary"
    assert len(adapter.routing_attempts) == 1


def test_failover_both_unsupported_fails_closed() -> None:
    primary = _StrictlessToolsAdapter(label="primary")
    secondary = _StrictlessToolsAdapter(label="secondary")
    adapter = FailoverLLMAdapter([primary, secondary])
    with pytest.raises(StrictToolArgumentConformanceError, match="does not support"):
        wire_schema, dispatch_requirements = _strict_round_schema()
        adapter.generate_with_tools(
            [ChatMessage(role="user", content="plan")],
            wire_schema,
            tool_dispatch_requirements=dispatch_requirements,
        )
