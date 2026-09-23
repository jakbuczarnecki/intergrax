# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R1-R4 — pluggable failover policy and authoritative eligibility."""

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.failover_policy import (
    FailoverCandidateEligibility,
    FailoverCandidateEligibilityContext,
    FailoverCandidateNotAuthorizedError,
    FailoverDecision,
    FailoverPolicy,
    FailoverProgressionContext,
    FailoverRoutingAuthorisationContext,
)
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.llm_adapters.registry.failover_policy import (
    PlatformDefaultFailoverPolicy,
    RoutingAllowlistFailoverEligibilityPolicy,
    assert_failover_chain_authorized,
)
from intergrax.llm_adapters.registry.profile import create_adapter_with_failover
from intergrax.llm_adapters.routing.evaluator import profile_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _HttpStatusError(RuntimeError):
    status_code: int

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _StubAdapter(BaseLLMAdapter):
    provider = LLMProvider.OPENAI
    model: str = "gpt-4o"

    def __init__(self, *, model: str, fail: bool = False, status_code: int = 429) -> None:
        super().__init__()
        self.model = model
        self._fail = fail
        self._status_code = status_code
        self.dispatched = False

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        del messages, kwargs
        self.dispatched = True
        if self._fail:
            raise _HttpStatusError("provider error", status_code=self._status_code)
        return LLMAdapterResponse(
            content=f"ok-{self.model}",
            usage=LLMTokenUsage(input_tokens=1, output_tokens=1),
            model=self.model,
            provider="openai",
        )


class ExternalFailoverPolicy:
    """Structural external progression policy — always STOP after first failure."""

    def decide_after_failure(self, context: FailoverProgressionContext) -> FailoverDecision:
        del context
        return FailoverDecision.STOP


class ExternalDenyAllEligibilityPolicy:
    """Structural external eligibility policy — deny every candidate when authorised."""

    def evaluate(self, context: FailoverCandidateEligibilityContext) -> FailoverCandidateEligibility:
        if context.routing_authorisation is None:
            return FailoverCandidateEligibility.ALLOWED
        return FailoverCandidateEligibility.DENIED


def _profile(provider: LLMProvider, model: str) -> LLMProfile:
    return LLMProfile(provider=provider, model=model)


def test_e1_candidate_in_allowed_set_is_allowed() -> None:
    primary = _profile(LLMProvider.OPENAI, "gpt-4o")
    fallback = _profile(LLMProvider.GROQ, "llama-backup")
    auth = FailoverRoutingAuthorisationContext(allowed_profiles=(primary, fallback))
    policy = RoutingAllowlistFailoverEligibilityPolicy()
    verdict = policy.evaluate(
        FailoverCandidateEligibilityContext(
            candidate=fallback,
            candidate_index=1,
            primary_selected=primary,
            routing_authorisation=auth,
        )
    )
    assert verdict is FailoverCandidateEligibility.ALLOWED


def test_e2_candidate_outside_allowed_set_is_denied() -> None:
    primary = _profile(LLMProvider.OPENAI, "gpt-4o")
    fallback = _profile(LLMProvider.GROQ, "llama-backup")
    auth = FailoverRoutingAuthorisationContext(allowed_profiles=(primary,))
    policy = RoutingAllowlistFailoverEligibilityPolicy()
    verdict = policy.evaluate(
        FailoverCandidateEligibilityContext(
            candidate=fallback,
            candidate_index=1,
            primary_selected=primary,
            routing_authorisation=auth,
        )
    )
    assert verdict is FailoverCandidateEligibility.DENIED


def test_e3_primary_selected_profile_remains_valid() -> None:
    primary = _profile(LLMProvider.OPENAI, "gpt-4o")
    auth = FailoverRoutingAuthorisationContext(allowed_profiles=(primary,))
    assert_failover_chain_authorized((primary,), routing_authorisation=auth)


def test_e4_partial_chain_with_unauthorized_fallback_raises() -> None:
    primary = _profile(LLMProvider.OPENAI, "gpt-4o")
    allowed_fallback = _profile(LLMProvider.GROQ, "allowed")
    denied_fallback = _profile(LLMProvider.CLAUDE, "denied")
    auth = FailoverRoutingAuthorisationContext(
        allowed_profiles=(primary, allowed_fallback),
    )
    with pytest.raises(FailoverCandidateNotAuthorizedError, match="denied"):
        assert_failover_chain_authorized(
            (primary, denied_fallback, allowed_fallback),
            routing_authorisation=auth,
        )


def test_e5_external_eligibility_policy_is_pluggable() -> None:
    primary = _profile(LLMProvider.OPENAI, "gpt-4o")
    auth = FailoverRoutingAuthorisationContext(allowed_profiles=(primary,))
    with pytest.raises(FailoverCandidateNotAuthorizedError):
        assert_failover_chain_authorized(
            (primary,),
            routing_authorisation=auth,
            eligibility_policy=ExternalDenyAllEligibilityPolicy(),
        )


def test_routing_failover_unauthorized_fallback_must_not_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routing selects A; B is fallback but not policy-authorized → B must not run."""
    profile_a = _profile(LLMProvider.OPENAI, "gpt-4o")
    profile_b = _profile(LLMProvider.GROQ, "backup")
    profile_a = profile_a.model_copy(
        update={"fallback_profiles": (profile_b,)},
    )
    auth = FailoverRoutingAuthorisationContext(allowed_profiles=(profile_a,))

    def _fake_create(profile: LLMProfile, **kwargs: object) -> LLMAdapter:
        del kwargs
        return _StubAdapter(model=profile.model or "default")

    monkeypatch.setattr(
        "intergrax.llm_adapters.registry.profile.create_adapter",
        _fake_create,
    )
    with pytest.raises(FailoverCandidateNotAuthorizedError, match="groq:backup"):
        create_adapter_with_failover(
            profile_a,
            routing_authorisation=auth,
        )


def test_external_failover_policy_stops_on_retryable_error() -> None:
    primary = _StubAdapter(model="primary", fail=True, status_code=429)
    secondary = _StubAdapter(model="secondary")
    adapter = FailoverLLMAdapter(
        [primary, secondary],
        failover_policy=ExternalFailoverPolicy(),
    )
    with pytest.raises(_HttpStatusError):
        adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert secondary.dispatched is False


def test_default_failover_policy_regression_primary_429_advances() -> None:
    primary = _StubAdapter(model="primary", fail=True, status_code=429)
    secondary = _StubAdapter(model="secondary")
    adapter = FailoverLLMAdapter(
        [primary, secondary],
        failover_policy=PlatformDefaultFailoverPolicy(),
    )
    response = adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert response.content == "ok-secondary"


def test_default_failover_policy_regression_non_retriable_stops() -> None:
    primary = _StubAdapter(model="primary", fail=True, status_code=400)
    secondary = _StubAdapter(model="secondary")
    adapter = FailoverLLMAdapter(
        [primary, secondary],
        failover_policy=PlatformDefaultFailoverPolicy(),
    )
    with pytest.raises(_HttpStatusError):
        adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert secondary.dispatched is False


def test_default_failover_policy_regression_last_candidate_propagates() -> None:
    primary = _StubAdapter(model="primary", fail=True, status_code=429)
    adapter = FailoverLLMAdapter(
        [primary],
        failover_policy=PlatformDefaultFailoverPolicy(),
    )
    with pytest.raises(_HttpStatusError):
        adapter.generate_messages([ChatMessage(role="user", content="hi")])


def test_eligibility_error_names_profile_identity() -> None:
    primary = _profile(LLMProvider.OPENAI, "gpt-4o")
    rogue = _profile(LLMProvider.GROQ, "rogue")
    auth = FailoverRoutingAuthorisationContext(allowed_profiles=(primary,))
    with pytest.raises(FailoverCandidateNotAuthorizedError) as exc:
        assert_failover_chain_authorized(
            (primary, rogue),
            routing_authorisation=auth,
        )
    assert profile_identity(rogue) in str(exc.value)
