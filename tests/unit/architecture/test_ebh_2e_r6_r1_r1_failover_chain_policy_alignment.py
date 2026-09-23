# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R1-R1 — failover chain policy alignment after strict filtering."""

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    build_atomic_planner_round_tool_definition,
)
from testing_support.atomic_planner_round_transport import poc_business_tool_schemas

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _HttpStatusError(RuntimeError):
    status_code: int

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _StrictPolicyToolsAdapter(BaseLLMAdapter):
    provider = "strict-policy"
    model = "strict-policy"

    def __init__(
        self,
        *,
        label: str,
        fail: bool = False,
        status_code: int = 429,
    ) -> None:
        super().__init__()
        self._label = label
        self.model = label
        self._fail = fail
        self._status_code = status_code
        self.dispatched = False

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_strict_tool_argument_conformance(self) -> bool:
        return True

    def generate_messages(self, messages, **kwargs):
        raise NotImplementedError

    def generate_with_tools(
        self,
        messages,
        tools,
        *,
        temperature=None,
        max_tokens=None,
        tool_choice=None,
        run_id=None,
    ) -> LLMAdapterResponse:
        del messages, tools, temperature, max_tokens, tool_choice, run_id
        self.dispatched = True
        if self._fail:
            raise _HttpStatusError("provider error", status_code=self._status_code)
        return LLMAdapterResponse(
            content=f"ok-{self._label}",
            usage=LLMTokenUsage(input_tokens=1, output_tokens=1),
            model=self.model,
            provider=str(self.provider),
        )


class _StrictlessPolicyToolsAdapter(BaseLLMAdapter):
    provider = "strictless-policy"
    model = "strictless-policy"

    def __init__(self, *, label: str) -> None:
        super().__init__()
        self._label = label
        self.model = label
        self.dispatched = False

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_strict_tool_argument_conformance(self) -> bool:
        return False

    def generate_messages(self, messages, **kwargs):
        raise NotImplementedError

    def generate_with_tools(self, messages, tools, **kwargs) -> LLMAdapterResponse:
        del messages, tools, kwargs
        self.dispatched = True
        return LLMAdapterResponse(
            content=f"ok-{self._label}",
            usage=LLMTokenUsage(input_tokens=1, output_tokens=1),
            model=self.model,
            provider=str(self.provider),
        )


def _strict_round_definition():
    return build_atomic_planner_round_tool_definition(poc_business_tool_schemas())


def test_ebh_2e_r6_r1_r1_strict_filter_preserves_per_entry_failover_retry_policy() -> None:
    """A/B/C chain: filtered B must not shift C onto B's retry policy (409 case)."""
    config_a = LLMCallConfig(retry_on_status=(418,))
    config_b = LLMCallConfig(retry_on_status=())
    config_c = LLMCallConfig(retry_on_status=(409,))
    config_d = LLMCallConfig()

    adapter_a = _StrictPolicyToolsAdapter(label="a", fail=True, status_code=418)
    adapter_b = _StrictlessPolicyToolsAdapter(label="b")
    adapter_c = _StrictPolicyToolsAdapter(label="c", fail=True, status_code=409)
    adapter_d = _StrictPolicyToolsAdapter(label="d")

    failover = FailoverLLMAdapter(
        [adapter_a, adapter_b, adapter_c, adapter_d],
        profile_ids=("profile-a", "profile-b", "profile-c", "profile-d"),
        adapter_failover_retry_configs=(config_a, config_b, config_c, config_d),
    )
    response = failover.generate_with_tools(
        [ChatMessage(role="user", content="plan")],
        [_strict_round_definition()],
    )
    assert response.content == "ok-d"
    assert adapter_b.dispatched is False
    assert len(failover.routing_attempts) == 2
    assert failover.routing_attempts[0].profile_id == "profile-a"
    assert failover.routing_attempts[1].profile_id == "profile-c"


def test_ebh_2e_r6_r1_r1_misaligned_policy_would_stop_at_c() -> None:
    """Control: if C used default (non-409) policy, failover would not reach D."""
    config_a = LLMCallConfig(retry_on_status=(418,))
    config_b = LLMCallConfig(retry_on_status=())
    config_c_wrong = LLMCallConfig(retry_on_status=())
    config_d = LLMCallConfig()

    adapter_a = _StrictPolicyToolsAdapter(label="a", fail=True, status_code=418)
    adapter_b = _StrictlessPolicyToolsAdapter(label="b")
    adapter_c = _StrictPolicyToolsAdapter(label="c", fail=True, status_code=409)
    adapter_d = _StrictPolicyToolsAdapter(label="d")

    failover = FailoverLLMAdapter(
        [adapter_a, adapter_b, adapter_c, adapter_d],
        profile_ids=("profile-a", "profile-b", "profile-c", "profile-d"),
        adapter_failover_retry_configs=(config_a, config_b, config_c_wrong, config_d),
    )
    with pytest.raises(_HttpStatusError, match="provider error"):
        failover.generate_with_tools(
            [ChatMessage(role="user", content="plan")],
            [_strict_round_definition()],
        )
    assert adapter_d.dispatched is False
    assert failover.routing_attempts[-1].profile_id == "profile-c"
