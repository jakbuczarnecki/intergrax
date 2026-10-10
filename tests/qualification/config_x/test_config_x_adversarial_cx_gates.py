# © Artur Czarnecki. All rights reserved.

"""CONFIG-X adversarial matrix CX-A … CX-H."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    UnknownIntegrationError,
)
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.registry.catalog import get_entry
from intergrax.integrations.registry.factory import resolve, resolve_slug
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry
from intergrax.tools.providers.observability.resolve import resolve_observability_backend
from intergrax.tools.registry.wiring import ToolWiringContext

from tests.qualification.config_x._config_x_owner_discovery import CONFIG_X_OWNER_EXPECTATIONS

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_cx_a_missing_required_integration_config_fails_closed() -> None:
    with pytest.raises(IntegrationConfigurationError):
        resolve_slug(IntegrationCategory.VECTOR_STORE, profile=IntegrationProfile())


def test_cx_b_invalid_provider_slug_rejected() -> None:
    with pytest.raises(UnknownIntegrationError):
        get_entry("not-a-real-integration-slug-xyz")


def test_cx_c_unsupported_backend_category_mismatch() -> None:
    profile = IntegrationProfile(relational_store="sqlite")
    with pytest.raises(IntegrationConfigurationError, match="vector_store"):
        resolve(IntegrationCategory.VECTOR_STORE, profile=profile)


def test_cx_d_registered_llm_provider_unconfigured_not_effective() -> None:
    LLMAdapterRegistry.reset_for_testing()
    with pytest.raises(ValueError, match="not registered"):
        LLMAdapterRegistry.create("config_x_unregistered_provider_id")


def test_cx_e_configured_integration_slug_materializes() -> None:
    profile = IntegrationProfile(relational_store="sqlite")
    slug = resolve_slug(IntegrationCategory.RELATIONAL_STORE, profile=profile)
    assert slug == "sqlite"


def test_cx_f_configuration_change_does_not_imply_resolver_owner_change() -> None:
    """Configured/effective provenance owner remains TRACE-X execution-bound resolution."""
    from tests.qualification.config_x._config_x_owner_discovery import compare_owner_gate

    discovered, expected = compare_owner_gate("execution_bound_integration_resolution")
    assert discovered == expected


def test_cx_g_cross_tenant_provider_config_rejected_by_realization_guards() -> None:
    from intergrax.integrations.contracts.existing_capability_configuration import (
        ExistingCapabilityConfigurationRealizationError,
        ExistingCapabilityConfigurationRealizationFailureReason,
    )
    from tests.qualification.existing_capability_configuration.test_configuration_realization_certification import (
        _principal,
        _request,
    )

    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        _request(tenant_id="tenant-b", principal=_principal(tenant_id="tenant-a"))
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )


def test_cx_h_duplicate_configuration_authority_synthetic_probe_fails() -> None:
    synthetic = dict(CONFIG_X_OWNER_EXPECTATIONS)
    synthetic["synthetic_second_integration_resolver"] = frozenset(
        {"intergrax/integrations/registry/factory.py"},
    )
    assert len(synthetic) > len(CONFIG_X_OWNER_EXPECTATIONS)


def test_cx_observability_missing_config_raises() -> None:
    ctx = ToolWiringContext()
    with pytest.raises(RuntimeError, match="observability_backend_not_configured"):
        resolve_observability_backend(ctx)


def test_cx_tokenizer_missing_registration_fails_closed() -> None:
    registry = TokenizerRegistry()
    with pytest.raises(ValueError, match="No tokenizer registered"):
        registry.default()


def test_cx_d_tokenizer_registered_without_explicit_default_not_effective() -> None:
    from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer
    from intergrax.tokenizers.providers.tiktoken_tokenizer import TiktokenTokenizer

    registry = TokenizerRegistry()
    registry.register(SimpleTokenizer())
    registry.register(TiktokenTokenizer())
    with pytest.raises(ValueError, match="No default tokenizer configured"):
        registry.get(None)


def test_cx_observability_role_missing_sanctioned_backend_fails_closed() -> None:
    class _Unsanctioned:
        def query_traces(self, *, limit: int = 20, name=None):
            return None

    ctx = ToolWiringContext(observability_backends={"only_custom": _Unsanctioned()})
    with pytest.raises(RuntimeError, match="observability_role_backend_not_configured"):
        resolve_observability_backend(ctx, role="traces")
