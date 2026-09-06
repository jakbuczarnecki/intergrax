# © Artur Czarnecki. All rights reserved.

"""Unit tests for DS-E2E qualification provider binding."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    build_scenario_environment_profile,
)
from intergrax.llm_adapters.registry.catalog_capabilities import (
    CatalogCapabilityAdapter,
    unwrap_catalog_capability_adapter,
)
from intergrax.llm_adapters.registry.model_catalog import ModelRecord
from testing_support.decision_e2e.provider_binding import (
    bind_qualification_llm_profile,
)

pytestmark = pytest.mark.unit


class _StubAdapter(LLMAdapter):
    provider = LLMProvider.OPENAI
    model = "gpt-4.1"

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response

        return build_adapter_response(content="ok")


def _resolver(_environment: ApplicationEnvironmentProfile) -> LLMAdapter:
    return _StubAdapter()


def test_wrapped_adapter_identity_uses_unwrapped_llm_adapter_fields() -> None:
    inner = _StubAdapter()
    inner.provider = LLMProvider.OPENAI
    inner.model = "gpt-4.1"
    wrapped = CatalogCapabilityAdapter(
        inner,
        ModelRecord(model_id="gpt-4.1", context_window_tokens=128_000),
    )

    assert unwrap_catalog_capability_adapter(wrapped) is inner

    environment = build_scenario_environment_profile()
    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_LLM_PROVIDER": "openai",
            "INTERGRAX_LLM_MODEL": "gpt-4.1",
            "OPENAI_API_KEY": "test-key",
        },
        clear=False,
    ):
        with patch.object(LLMProfile, "create_adapter", return_value=wrapped):
            binding, block_reason = bind_qualification_llm_profile(
                environment,
                adapter_resolver=lambda _env: wrapped,
            )

    assert block_reason is None
    assert binding is not None
    assert binding.resolved_provider == "openai"
    assert binding.resolved_model == "gpt-4.1"


def test_bind_openai_gpt41_resolves_requested_and_resolved_identity() -> None:
    environment = build_scenario_environment_profile()
    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_LLM_PROVIDER": "openai",
            "INTERGRAX_LLM_MODEL": "gpt-4.1",
            "OPENAI_API_KEY": "test-key",
        },
        clear=False,
    ):
        with patch.object(LLMProfile, "create_adapter", return_value=_StubAdapter()):
            binding, block_reason = bind_qualification_llm_profile(
                environment,
                adapter_resolver=_resolver,
            )

    assert block_reason is None
    assert binding is not None
    assert binding.requested_provider == "openai"
    assert binding.requested_model == "gpt-4.1"
    assert binding.resolved_provider == "openai"
    assert binding.resolved_model == "gpt-4.1"
    assert binding.binding_source == "application_environment_profile.llm_profile"
    assert environment.llm_profile is not None
    assert environment.llm_profile.model == "gpt-4.1"


def test_bind_ollama_qwen32_resolves_requested_and_resolved_identity() -> None:
    environment = build_scenario_environment_profile()
    adapter = _StubAdapter()
    adapter.provider = LLMProvider.OLLAMA
    adapter.model = "qwen2.5:32b"

    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_LLM_PROVIDER": "ollama",
            "INTERGRAX_LLM_MODEL": "qwen2.5:32b",
        },
        clear=False,
    ):
        with patch.object(LLMProfile, "create_adapter", return_value=adapter):
            binding, block_reason = bind_qualification_llm_profile(
                environment,
                adapter_resolver=lambda _env: adapter,
            )

    assert block_reason is None
    assert binding is not None
    assert binding.requested_provider == "ollama"
    assert binding.requested_model == "qwen2.5:32b"
    assert binding.resolved_provider == "ollama"
    assert binding.resolved_model == "qwen2.5:32b"


def test_explicit_model_differs_from_lab_default_uses_explicit_model() -> None:
    environment = build_scenario_environment_profile()
    adapter = _StubAdapter()
    adapter.provider = LLMProvider.OLLAMA
    adapter.model = "qwen2.5:32b"

    with patch.dict(
        "os.environ",
        {"INTERGRAX_LLM_PROVIDER": "ollama", "INTERGRAX_LLM_MODEL": "qwen2.5:32b"},
        clear=False,
    ):
        with patch.object(LLMProfile, "create_adapter", return_value=adapter):
            binding, block_reason = bind_qualification_llm_profile(
                environment,
                adapter_resolver=lambda _env: adapter,
            )

    assert block_reason is None
    assert binding is not None
    assert binding.resolved_model == "qwen2.5:32b"
    assert binding.resolved_model != "llama3.1:latest"


def test_invalid_explicit_provider_fails_closed() -> None:
    environment = build_scenario_environment_profile()
    with patch.dict(
        "os.environ",
        {"INTERGRAX_LLM_PROVIDER": "not_registered_slug_env"},
        clear=False,
    ):
        binding, block_reason = bind_qualification_llm_profile(
            environment,
            adapter_resolver=_resolver,
        )

    assert binding is None
    assert block_reason is not None
    assert "invalid explicit qualification provider/model" in block_reason


def test_invalid_explicit_adapter_resolution_fails_closed() -> None:
    environment = build_scenario_environment_profile()
    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_LLM_PROVIDER": "openai",
            "INTERGRAX_LLM_MODEL": "gpt-4.1",
        },
        clear=False,
    ):
        with patch.object(
            LLMProfile,
            "create_adapter",
            side_effect=RuntimeError("missing credentials"),
        ):
            binding, block_reason = bind_qualification_llm_profile(
                environment,
                adapter_resolver=_resolver,
            )

    assert binding is None
    assert block_reason is not None
    assert "explicit qualification provider/model could not be resolved" in block_reason


def test_missing_explicit_qualification_env_uses_documented_default_provider() -> None:
    environment = build_scenario_environment_profile()
    adapter = _StubAdapter()
    adapter.provider = LLMProvider.OLLAMA
    adapter.model = None

    env = {
        key: value
        for key, value in __import__("os").environ.items()
        if not key.startswith("INTERGRAX_LLM_")
    }
    with patch.dict("os.environ", env, clear=True):
        with patch.object(LLMProfile, "create_adapter", return_value=adapter):
            binding, block_reason = bind_qualification_llm_profile(
                environment,
                adapter_resolver=lambda _env: adapter,
            )

    assert block_reason is None
    assert binding is not None
    assert binding.requested_provider is None
    assert binding.requested_model is None
    assert binding.resolved_provider == "ollama"
