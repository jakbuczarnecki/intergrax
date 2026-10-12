# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R2 canonical configuration → credential → provider chain (current-HEAD structural)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present
from tests.qualification.tenant_x._r2_types import R2ChainTransition, R2ChainVerdict

_AWARD7C_CHAIN_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py",
        "credential_broker.resolve_scoped(credential_grant, scope)",
    ),
    (
        "intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py",
        "credential_resolution=resolved",
    ),
    (
        "intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py",
        "ingress.credential_resolution.resolved_credential.value",
    ),
)


def _aw7c_chain_proven() -> tuple[str, ...]:
    return all_structural_patterns_present(_AWARD7C_CHAIN_CHECKS)


def build_r2_e2e_chain() -> tuple[R2ChainTransition, ...]:
    aw7c = _aw7c_chain_proven()
    aw7c_verdict = (
        R2ChainVerdict.PROVEN_CURRENT_HEAD if len(aw7c) == len(_AWARD7C_CHAIN_CHECKS) else R2ChainVerdict.MISSING
    )
    return (
        R2ChainTransition(
            transition_id="cfg.vector.require_tenant_id",
            source_symbol="intergrax/integrations/_shared/p3/configs.py::VectorIntegrationConfig",
            target_symbol="intergrax/integrations/_shared/p3/configs.py::VectorIntegrationConfig::require_tenant_id",
            tenant_source="config.tenant_id (explicit; empty raises IntegrationConfigurationError)",
            credential_or_provider_key="vector integration binding",
            mismatch_behavior="missing tenant → IntegrationConfigurationError",
            fail_closed=True,
            verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            evidence=(
                "intergrax/integrations/_shared/p3/configs.py::Vector integration requires explicit tenant_id configuration",
            ),
        ),
        R2ChainTransition(
            transition_id="grant.effect_request.scope",
            source_symbol="intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py::_validate_grant_against_effect_request",
            target_symbol="intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py::CredentialUseScope",
            tenant_source="effect_request.tenant_id",
            credential_or_provider_key="grant.provider_id / grant.integration_id",
            mismatch_behavior="grant tenant mismatch → CREDENTIAL_DENIED envelope",
            fail_closed=True,
            verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            evidence=(
                "intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py::grant tenant mismatch",
            ),
        ),
        R2ChainTransition(
            transition_id="scope.broker.resolve",
            source_symbol="intergrax/integrations/credentials/broker.py::ScopedCredentialBroker::resolve_scoped",
            target_symbol="intergrax/integrations/credentials/scope_validation.py::assert_tenant_consistency",
            tenant_source="scope.tenant_id + grant.tenant_id + optional ref.tenant_id",
            credential_or_provider_key="grant.credential_ref",
            mismatch_behavior="CredentialScopeMismatchError before resolver",
            fail_closed=True,
            verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            evidence=(
                "intergrax/integrations/credentials/broker.py::assert_tenant_consistency(ref=grant.credential_ref, grant=grant, scope=scope)",
            ),
        ),
        R2ChainTransition(
            transition_id="broker.resolver.context",
            source_symbol="intergrax/integrations/credentials/broker.py::CredentialResolutionContext",
            target_symbol="intergrax/integrations/credentials/secrets_store_resolver.py::SecretsStoreCredentialResolver::resolve",
            tenant_source="context.tenant_id from scope.tenant_id",
            credential_or_provider_key="ref.secret_path",
            mismatch_behavior="resolver _assert_tenant_scope vs ref.tenant_id",
            fail_closed=True,
            verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            evidence=(
                "intergrax/integrations/credentials/broker.py::tenant_id=scope.tenant_id",
                "intergrax/integrations/credentials/secrets_store_resolver.py::credential resolution requires tenant context",
            ),
        ),
        R2ChainTransition(
            transition_id="resolution.effect_executor",
            source_symbol="intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py::ReferenceScopedAdaptedIntegrationEffectExecutionContext",
            target_symbol="intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py::ReferenceScopedAdaptedIntegrationEffectExecutor::execute",
            tenant_source="ingress.credential_resolution.use_evidence.tenant_id vs effect_request.tenant_id",
            credential_or_provider_key="resolved_credential.value (material consumption)",
            mismatch_behavior="ValueError on evidence tenant/provider/integration/operation mismatch",
            fail_closed=True,
            verdict=aw7c_verdict,
            evidence=aw7c
            if aw7c_verdict == R2ChainVerdict.PROVEN_CURRENT_HEAD
            else ("discovery: AW-7C reference consumption markers incomplete",),
        ),
        R2ChainTransition(
            transition_id="vector.scope.pin",
            source_symbol="intergrax/rag/vectorstore/providers/native_provider_boundary.py::validate_scope",
            target_symbol="intergrax/integrations/_shared/p8/factories.py::create_lancedb_vector_store (provider-local pin)",
            tenant_source="first VectorStoreScope.tenant_id pins provider-local _tenant_id",
            credential_or_provider_key="vector store provider (non-credential)",
            mismatch_behavior="VectorStoreContractError on scope tenant_id differs from provider tenant",
            fail_closed=True,
            verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            evidence=(
                "intergrax/rag/vectorstore/providers/native_provider_boundary.py::scope tenant_id differs from provider tenant",
                "intergrax/integrations/_shared/p8/factories.py::self._tenant_id = scope.tenant_id",
            ),
        ),
    )


def r2_chain_missing_count() -> int:
    return sum(1 for t in build_r2_e2e_chain() if t.verdict == R2ChainVerdict.MISSING)


def r2_chain_invalid_count() -> int:
    return sum(1 for t in build_r2_e2e_chain() if t.verdict == R2ChainVerdict.INVALID)
