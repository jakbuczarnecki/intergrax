# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R1-R4 — pluggable failover policy and authoritative eligibility."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FAILOVER_POLICY_CONTRACT = (
    _REPO_ROOT / "intergrax/llm_adapters/contracts/failover_policy.py"
)
_FAILOVER_ADAPTER = _REPO_ROOT / "intergrax/llm_adapters/registry/failover_adapter.py"
_PROFILE_FACTORY = _REPO_ROOT / "intergrax/llm_adapters/registry/profile.py"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _forbidden_import_prefixes(modules: set[str], prefixes: tuple[str, ...]) -> list[str]:
    return sorted(
        module
        for module in modules
        if any(module.startswith(prefix) or f".{prefix}" in module for prefix in prefixes)
        or any(module == prefix for prefix in prefixes)
    )


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


def test_r4_r1_failover_contract_import_purity() -> None:
    modules = _imported_modules(_FAILOVER_POLICY_CONTRACT)
    forbidden = _forbidden_import_prefixes(
        modules,
        (
            "intergrax.llm_adapters._shared",
            "intergrax.llm_adapters.registry",
            "intergrax.llm_adapters.routing.evaluator",
            "intergrax.runtime",
            "applications",
        ),
    )
    assert forbidden == []


def test_r4_r1_failover_executor_does_not_select_default_policy() -> None:
    modules = _imported_modules(_FAILOVER_ADAPTER)
    assert "intergrax.llm_adapters.registry.failover_policy" not in modules
    source = _FAILOVER_ADAPTER.read_text(encoding="utf-8")
    assert "default_failover_policy" not in source


def test_r4_r1_composition_selects_default_failover_policy() -> None:
    source = _PROFILE_FACTORY.read_text(encoding="utf-8")
    assert "default_failover_policy()" in source
    assert "failover_policy or default_failover_policy()" in source


def test_r4_r1_canonical_llm_call_config_single_type() -> None:
    from intergrax.llm_adapters._shared.call_config import LLMCallConfig as SharedConfig
    from intergrax.llm_adapters.contracts.call_config import LLMCallConfig as CanonicalConfig

    assert SharedConfig is CanonicalConfig
    contract_path = _REPO_ROOT / "intergrax/llm_adapters/contracts/call_config.py"
    tree = ast.parse(contract_path.read_text(encoding="utf-8"))
    class_defs = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert class_defs.count("LLMCallConfig") == 1


def test_r4_r1_failover_adapter_requires_explicit_policy() -> None:
    params = inspect.signature(FailoverLLMAdapter.__init__).parameters
    assert "failover_policy" in params
    assert params["failover_policy"].default is inspect.Parameter.empty


def test_r4_r1_create_adapter_with_failover_default_advances_on_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_a = _profile(LLMProvider.OPENAI, "gpt-4o")
    profile_b = _profile(LLMProvider.GROQ, "backup")
    profile_a = profile_a.model_copy(update={"fallback_profiles": (profile_b,)})

    def _fake_create(profile: LLMProfile, **kwargs: object) -> LLMAdapter:
        del kwargs
        fail = profile.provider is LLMProvider.OPENAI
        return _StubAdapter(model=profile.model or "default", fail=fail, status_code=429)

    monkeypatch.setattr(
        "intergrax.llm_adapters.registry.profile.create_adapter",
        _fake_create,
    )
    adapter = create_adapter_with_failover(profile_a)
    response = adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert response.content == "ok-backup"
