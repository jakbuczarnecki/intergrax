# © Artur Czarnecki. All rights reserved.

"""FC-3 Part A: embedding provider selector fail-closed (NT-FC3-E1..E6)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.integrations.contracts.base import LabSelectorOptIn
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager
from intergrax.rag.embedding.contracts.runtime_binding import EmbeddingProviderConfigurationError
from intergrax.rag.embedding.registry.profile import embedding_profile_from_env
from intergrax.rag.embedding.registry.provider_authority import lab_embedding_provider_slug
from intergrax.rag.embedding.runtime.resolver import resolve_embedding_provider_slug

from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = pytest.mark.unit

_LAB_OPT_IN = LabSelectorOptIn(
    selector=lab_embedding_provider_slug(),
    reason="FC-3 unit negative-test positive control",
)


@pytest.fixture(autouse=True)
def _bootstrap_embedding_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    yield
    clear_catalog()
    reset_default_integrations_state()


def test_nt_fc3_e1_resolve_slug_env_unset_no_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_EMBEDDING_PROVIDER", raising=False)
    with pytest.raises(EmbeddingProviderConfigurationError):
        resolve_embedding_provider_slug()


def test_nt_fc3_e2_embedding_profile_from_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("INTERGRAX_EMBEDDING_MODEL", raising=False)
    with pytest.raises(EmbeddingProviderConfigurationError):
        embedding_profile_from_env()


def test_nt_fc3_e3_create_default_embedding_manager_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_EMBEDDING_PROVIDER", raising=False)
    with patch(
        "intergrax.rag.embedding.bootstrap.default_embedding_engine.bind_embedding_provider",
    ) as bind_mock:
        with pytest.raises(EmbeddingProviderConfigurationError):
            create_default_embedding_manager()
        bind_mock.assert_not_called()


def test_nt_fc3_e4_create_default_embedding_manager_lab_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_EMBEDDING_PROVIDER", raising=False)
    fake_provider = MagicMock()
    fake_provider.provider_name.return_value = lab_embedding_provider_slug()
    with patch(
        "intergrax.rag.embedding.bootstrap.default_embedding_engine.bind_embedding_provider",
        return_value=fake_provider,
    ) as bind_mock:
        manager = create_default_embedding_manager(lab_opt_in=_LAB_OPT_IN)
    assert manager is not None
    assert bind_mock.call_args is not None
    call_kwargs = bind_mock.call_args.kwargs
    assert call_kwargs["embedding_profile"].provider == lab_embedding_provider_slug()


def test_nt_fc3_e5_invalid_slug_always_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_EMBEDDING_PROVIDER", "not_a_real_slug")
    with pytest.raises(ValueError, match="unknown embedding provider"):
        embedding_profile_from_env()
    with pytest.raises(ValueError, match="unknown embedding provider"):
        embedding_profile_from_env(
            lab_opt_in=LabSelectorOptIn(
                selector="not_a_real_slug",
                reason="invalid slug must not be legitimised",
            ),
        )


def test_nt_fc3_e6_wire_application_environment_strict_rag_no_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_EMBEDDING_PROVIDER", raising=False)
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc3.embedding.strict")
    env = env.model_copy(
        update={
            "meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
        },
    )
    assert env.context_profile.enable_rag is True
    with pytest.raises(EmbeddingProviderConfigurationError):
        wire_application_environment(
            manifest,
            env,
            integration_profile=IntegrationProfile(),
            conformance_check=False,
        )
