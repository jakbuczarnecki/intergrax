"""Red-first NT-PLG probes that avoid optional langchain_ollama import failure."""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = pytest.mark.unit


def _stub_optional_langchain_embeddings_if_missing() -> None:
    """Avoid optional vendor extras blocking collection; not the FRZ assertion target."""
    for name, attr in (
        ("langchain_ollama", "OllamaEmbeddings"),
        ("langchain_openai", "OpenAIEmbeddings"),
    ):
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        setattr(stub, attr, type(attr, (), {}))
        sys.modules[name] = stub


def test_nt_plg_01_validate_evaluation_wiring_requires_registry_when_online_enabled() -> None:
    _stub_optional_langchain_embeddings_if_missing()
    from intergrax.applications._shared.evaluation_assembly_resolver import validate_evaluation_wiring
    from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
        EvaluationProfile,
    )

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.registry")
    env.evaluation_profile = EvaluationProfile(
        shadow_eval_enabled=True,
        online_registry_enabled=True,
    )
    wiring = wire_application_evaluation(env)
    wiring = type(wiring)(
        profile=wiring.profile,
        options=wiring.options,
        registry=None,
        governance_bridge=wiring.governance_bridge,
        domain_fragments=wiring.domain_fragments,
    )
    result = validate_evaluation_wiring(wiring, env)
    assert not result.valid
    assert any("online_registry_enabled requires evaluation registry" in error for error in result.errors)


def test_nt_plg_02_validate_evaluation_wiring_rejects_baseline_without_trend() -> None:
    _stub_optional_langchain_embeddings_if_missing()
    from intergrax.applications._shared.evaluation_assembly_resolver import (
        EvaluationAssemblyError,
        assert_evaluation_assembly_valid,
    )
    from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
        EvaluationProfile,
    )

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.baseline")
    env.evaluation_profile = EvaluationProfile(
        require_baseline_for_release=True,
        trend_comparison_enabled=False,
        online_registry_enabled=True,
    )
    wiring = wire_application_evaluation(env)
    with pytest.raises(EvaluationAssemblyError):
        assert_evaluation_assembly_valid(wiring, env)


def test_nt_plg_07_evaluation_governance_wiring_fragment_guard() -> None:
    """NT-PLG-07: host wiring must publish evaluation_governance domain fragments."""
    _stub_optional_langchain_embeddings_if_missing()
    from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="eval.plg07")
    wiring = wire_application_evaluation(env)
    fragment = wiring.domain_fragments.get("evaluation_governance")
    assert isinstance(fragment, dict)
    assert fragment.get("shadow_eval_enabled") == env.evaluation_profile.shadow_eval_enabled
