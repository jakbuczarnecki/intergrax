# © Artur Czarnecki. All rights reserved.

"""Research host explicit LLM composition (NPSC-3B-R3V-R7)."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from research_application.host.environment_profile import (
    ResearchHostConfigurationError,
    build_research_environment_profile,
    require_research_orchestration_llm_profile,
    resolve_research_llm_profile,
)
from research_application.host.factory import create_research_backend_app
from research_application.host.settings import ResearchBackendSettings
from research_application.tests.research_ac3_projection import build_research_test_registry_projection

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_research_explicit_provider_propagates_to_environment_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    settings = ResearchBackendSettings(use_nexus_loop=True, llm_provider="groq")
    env = build_research_environment_profile(settings)
    assert env.llm_profile is not None
    assert env.llm_profile.provider == LLMProvider.GROQ


def test_research_explicit_model_propagates_to_environment_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    settings = ResearchBackendSettings(
        use_nexus_loop=True,
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",
    )
    env = build_research_environment_profile(settings)
    assert env.llm_profile is not None
    assert env.llm_profile.model == "llama-3.3-70b-versatile"


def test_research_profile_without_explicit_provider_is_not_implicit_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RESEARCH_LLM_MODEL", raising=False)
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("INTERGRAX_LLM_MODEL", raising=False)
    settings = ResearchBackendSettings(use_nexus_loop=True)
    profile = resolve_research_llm_profile(settings)
    assert profile is None
    env = build_research_environment_profile(settings)
    assert env.llm_profile is None


def test_research_factory_requires_explicit_llm_before_runtime_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    settings = ResearchBackendSettings(use_nexus_loop=True)
    with pytest.raises(ResearchHostConfigurationError, match="explicit LLM selection"):
        create_research_backend_app(
            registry_projection=build_research_test_registry_projection(settings),
            settings=settings,
        )


def test_research_profile_validation_is_not_optional_for_engine_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    settings = ResearchBackendSettings(use_nexus_loop=True)
    env = build_research_environment_profile(settings)
    with pytest.raises(ResearchHostConfigurationError, match="explicit LLM selection"):
        require_research_orchestration_llm_profile(settings, env)


def test_research_llm_profile_is_declarative_not_materialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    settings = ResearchBackendSettings(use_nexus_loop=True, llm_provider="ollama")
    profile = resolve_research_llm_profile(settings)
    assert isinstance(profile, LLMProfile)
    assert profile.provider == LLMProvider.OLLAMA
