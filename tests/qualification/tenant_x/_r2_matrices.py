# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R2 required qualification matrices."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r2_mechanisms import build_r2_canonical_mechanisms
from tests.qualification.tenant_x._r2_types import (
    R2ChainVerdict,
    R2CrossTenantClass,
    R2MatrixRow,
    R2ProviderRewriteClass,
    R2SubstitutionClass,
)

PROVIDER_CONFIG_CONTINUITY_MATRIX: Final[tuple[R2MatrixRow, ...]] = tuple(
    R2MatrixRow(
        row_id=m.mechanism_id,
        surface_hint=m.family_label,
        verdict=m.continuity_verdict,
        evidence=m.evidence,
    )
    for m in build_r2_canonical_mechanisms()
)

CREDENTIAL_REFERENCE_MATRIX: Final[tuple[R2MatrixRow, ...]] = (
    R2MatrixRow(
        row_id="cred_ref.optional_tenant_field",
        surface_hint="intergrax/integrations/contracts/credential.py::CredentialRef",
        verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=(
            "tenant_id optional on CredentialRef",
            "identity_fingerprint includes tenant when present",
        ),
    ),
    R2MatrixRow(
        row_id="cred_ref.broker_triple_check",
        surface_hint="intergrax/integrations/credentials/scope_validation.py::assert_tenant_consistency",
        verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("grant.tenant_id == scope.tenant_id", "ref.tenant_id must match scope when set"),
    ),
    R2MatrixRow(
        row_id="cred_ref.resolver_context",
        surface_hint="intergrax/integrations/credentials/secrets_store_resolver.py::_assert_tenant_scope",
        verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("resolution requires non-empty context.tenant_id", "ref tenant mismatch → CredentialScopeMismatchError"),
    ),
)

CREDENTIAL_BROKER_MATRIX: Final[tuple[R2MatrixRow, ...]] = (
    R2MatrixRow(
        row_id="broker.scope_validation",
        surface_hint="intergrax/integrations/credentials/scope_validation.py::validate_scoped_use_scope",
        verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("empty scope.tenant_id → CredentialScopeMismatchError",),
    ),
    R2MatrixRow(
        row_id="broker.grant_validation",
        surface_hint="intergrax/integrations/credentials/scope_validation.py::validate_credential_use_grant",
        verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("empty grant.tenant_id → CredentialScopeMismatchError",),
    ),
    R2MatrixRow(
        row_id="broker.provider_ref_alignment",
        surface_hint="intergrax/integrations/credentials/scope_validation.py::assert_grant_provider_matches_ref",
        verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("grant.provider_id must match credential_ref.provider_id",),
    ),
    R2MatrixRow(
        row_id="broker.admission_fail_closed",
        surface_hint="intergrax/integrations/credentials/broker.py::resolve_scoped",
        verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("CredentialScopeAdmissionDeniedError when admission denies",),
    ),
)

PROVIDER_CONSUMPTION_MATRIX: Final[tuple[R2MatrixRow, ...]] = tuple(
    R2MatrixRow(
        row_id=m.mechanism_id,
        surface_hint=m.notes,
        verdict=m.consumption_verdict,
        evidence=m.evidence,
    )
    for m in build_r2_canonical_mechanisms()
)

CROSS_TENANT_CREDENTIAL_MATRIX: Final[tuple[R2MatrixRow, ...]] = (
    R2MatrixRow(
        row_id="cross_tenant.grant_vs_scope_tenant",
        surface_hint="scope_validation.py::validate_credential_use_grant + validate_scoped_use_scope",
        verdict=R2CrossTenantClass.SAME_TENANT_PROVEN,
        evidence=("grant.tenant_id and scope.tenant_id must both be non-empty and aligned",),
    ),
    R2MatrixRow(
        row_id="cross_tenant.ref_vs_grant_scope",
        surface_hint="broker adversarial: tenant-a ref + tenant-b grant/scope",
        verdict=R2CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=(
            "tests.qualification.tenant_x._r2_adversarial::assert_cross_tenant_credential_ref_rejects",
            "CredentialScopeMismatchError",
        ),
    ),
    R2MatrixRow(
        row_id="cross_tenant.ref_vs_resolution_context",
        surface_hint="secrets_store_resolver.py::_assert_tenant_scope",
        verdict=R2CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("ref.tenant_id mismatch with context.tenant_id → CredentialScopeMismatchError",),
    ),
    R2MatrixRow(
        row_id="cross_tenant.provider_vs_grant_binding",
        surface_hint="scope_validation.py::assert_grant_provider_matches_ref",
        verdict=R2CrossTenantClass.SAME_TENANT_PROVEN,
        evidence=("grant.provider_id must match credential_ref.provider_id",),
    ),
    R2MatrixRow(
        row_id="cross_tenant.gw_ref_context_binding",
        surface_hint="google_workspace tenant_connection_factory + credential resolution context",
        verdict=R2CrossTenantClass.SAME_TENANT_PROVEN,
        evidence=("tenant_id embedded in CredentialRef and CredentialResolutionContext on GW paths",),
    ),
    R2MatrixRow(
        row_id="cross_tenant.missing_scope_tenant",
        surface_hint="validate_scoped_use_scope adversarial",
        verdict=R2CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=(
            "tests.qualification.tenant_x._r2_adversarial::assert_missing_tenant_on_credential_scope_rejects",
            "CredentialScopeMismatchError",
        ),
    ),
    R2MatrixRow(
        row_id="cross_tenant.aw7c_evidence_vs_effect_request",
        surface_hint="reference_scoped_adaptive_integration_execution adversarial",
        verdict=R2CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=(
            "tests.qualification.tenant_x._r2_adversarial::assert_reference_executor_rejects_evidence_tenant_mismatch",
            "credential evidence tenant mismatch",
        ),
    ),
)

PROVIDER_PLUGIN_REWRITE_MATRIX: Final[tuple[R2MatrixRow, ...]] = (
    R2MatrixRow(
        row_id="vector.pin.first_scope_wins",
        surface_hint="intergrax/integrations/_shared/p8/factories.py + lancedb/opens.py",
        verdict=R2ProviderRewriteClass.LOCAL_SCOPE_PIN,
        evidence=(
            "first operation pins tenant from supplied VectorStoreScope",
            "provider-local pin — not tenant mint/rewrite",
        ),
    ),
    R2MatrixRow(
        row_id="vector.scope.cross_tenant_after_pin",
        surface_hint="native_provider_boundary.py::validate_scope adversarial",
        verdict=R2ProviderRewriteClass.REWRITE_REJECTED,
        evidence=(
            "tests.qualification.tenant_x._r2_adversarial::assert_vector_scope_cross_tenant_rejects",
            "tenant B vector scope after A pin → VectorStoreContractError",
        ),
    ),
    R2MatrixRow(
        row_id="gw.factory.tenant_param_binding",
        surface_hint="google_workspace/tenant_connection_factory.py::create_late_bound_integration",
        verdict=R2ProviderRewriteClass.SCOPE_PRESERVED,
        evidence=("tenant_id parameter required; embedded in CredentialRef and resolution context",),
    ),
    R2MatrixRow(
        row_id="plugin.cannot_widen_scope",
        surface_hint="platform contract — extensions receive scope; no tenant mint API on provider Protocol",
        verdict=R2ProviderRewriteClass.REWRITE_REJECTED,
        evidence=("R2 adversarial: metadata/runtime override patterns rejected on audited paths",),
    ),
)

MISSING_DEFAULT_GLOBAL_MATRIX: Final[tuple[R2MatrixRow, ...]] = (
    R2MatrixRow(
        row_id="vector_config.missing_tenant",
        surface_hint="intergrax/integrations/_shared/p3/configs.py::require_tenant_id",
        verdict=R2SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("blank tenant_id raises — no default/global fallback",),
    ),
    R2MatrixRow(
        row_id="broker.missing_scope_tenant",
        surface_hint="validate_scoped_use_scope",
        verdict=R2SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("scoped credential resolution requires tenant context",),
    ),
    R2MatrixRow(
        row_id="resolver.missing_context_tenant",
        surface_hint="SecretsStoreCredentialResolver::_assert_tenant_scope",
        verdict=R2SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("credential resolution requires tenant context",),
    ),
    R2MatrixRow(
        row_id="ref.tenant_id_none_allowed_with_broker",
        surface_hint="CredentialRef.tenant_id optional",
        verdict=R2SubstitutionClass.LEGAL_NON_TENANT,
        evidence=(
            "OPTIONAL_REF_IDENTITY: grant tenant + CredentialUseScope tenant + CredentialResolutionContext tenant",
            "still constrain resolution — no global/shared fallback",
        ),
    ),
)
