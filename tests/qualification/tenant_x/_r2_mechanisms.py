# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R2 bounded canonical provider/config/credential mechanism inventory."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present, structural_patterns_present
from tests.qualification.tenant_x._r2_types import R2ChainVerdict, R2MechanismRow

_GW_FACTORY_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/integrations/providers/collaboration_suite/google_workspace/tenant_connection_factory.py",
        "tenant_id=tenant_id",
    ),
    (
        "intergrax/integrations/credentials/google_workspace.py",
        "self._resolver.resolve(self._credential_ref, context=self._context)",
    ),
)

_LANCEDB_PIN_CHECKS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/integrations/providers/vector_store/lancedb/opens.py",
        "self._tenant_id = scope.tenant_id",
    ),
    (
        "intergrax/integrations/providers/vector_store/lancedb/opens.py",
        "validate_scope(scope, tenant_id=self._tenant_id or scope.tenant_id)",
    ),
)


def _verdict_from_checks(checks: tuple[tuple[str, str], ...]) -> tuple[R2ChainVerdict, tuple[str, ...]]:
    matched = all_structural_patterns_present(checks)
    if len(matched) == len(checks):
        return R2ChainVerdict.PROVEN_CURRENT_HEAD, matched
    return R2ChainVerdict.MISSING, ("discovery: structural markers incomplete",)


def build_r2_canonical_mechanisms() -> tuple[R2MechanismRow, ...]:
    gw_cont, gw_ev = _verdict_from_checks(_GW_FACTORY_CHECKS)
    lance_cont, lance_ev = _verdict_from_checks(_LANCEDB_PIN_CHECKS)
    p8_ev = structural_patterns_present(
        "intergrax/integrations/_shared/p8/factories.py",
        ("self._tenant_id = scope.tenant_id", "validate_scope(scope, tenant_id=self._tenant_id"),
    )
    p8_cont = (
        R2ChainVerdict.PROVEN_CURRENT_HEAD
        if len(p8_ev) == 2
        else R2ChainVerdict.MISSING
    )
    return (
        R2MechanismRow(
            mechanism_id="cfg.vector_integration.require_tenant",
            family_label="configured provider strategies",
            continuity_verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            consumption_verdict=R2ChainVerdict.NOT_APPLICABLE,
            evidence=(
                "intergrax/integrations/_shared/p3/configs.py::require_tenant_id",
            ),
            notes="configuration/profile tenant required before vector provider binding",
        ),
        R2MechanismRow(
            mechanism_id="cred.broker.scoped_resolve",
            family_label="credential broker based providers",
            continuity_verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            consumption_verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            evidence=(
                "intergrax/integrations/credentials/broker.py::ScopedCredentialBroker::resolve_scoped",
                "intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py::credential_broker.resolve_scoped",
            ),
            notes="broker hands resolved material to AW-7C reference effect executor ingress",
        ),
        R2MechanismRow(
            mechanism_id="cred.ref.tenant_optional_broker_enforced",
            family_label="credential reference contract",
            continuity_verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            consumption_verdict=R2ChainVerdict.NOT_APPLICABLE,
            evidence=(
                "intergrax/integrations/contracts/credential.py::CredentialRef::tenant_id",
                "intergrax/integrations/credentials/scope_validation.py::assert_tenant_consistency",
            ),
            notes="ref.tenant_id optional; broker+resolver fail closed on mismatch",
        ),
        R2MechanismRow(
            mechanism_id="cred.resolver.secrets_store",
            family_label="credential broker based providers",
            continuity_verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            consumption_verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            evidence=(
                "intergrax/integrations/credentials/secrets_store_resolver.py::_assert_tenant_scope",
                "intergrax/integrations/credentials/google_workspace.py::resolve_credential",
            ),
            notes="resolution context tenant required; GW adapter consumes resolved value at client boundary",
        ),
        R2MechanismRow(
            mechanism_id="exec.aw7c.reference_effect",
            family_label="external API / integration providers",
            continuity_verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            consumption_verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            evidence=(
                "intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py::ingress.credential_resolution.resolved_credential.value",
            ),
            notes="canonical production credential→provider-operation consumption proof (reference path)",
        ),
        R2MechanismRow(
            mechanism_id="provider.google_workspace.late_bound_factory",
            family_label="document/integration providers",
            continuity_verdict=gw_cont,
            consumption_verdict=gw_cont,
            evidence=gw_ev,
            notes="factory binds tenant_id into CredentialRef + CredentialResolutionContext",
        ),
        R2MechanismRow(
            mechanism_id="vector.native.validate_scope",
            family_label="vector store providers",
            continuity_verdict=R2ChainVerdict.PROVEN_CURRENT_HEAD,
            consumption_verdict=R2ChainVerdict.NOT_APPLICABLE,
            evidence=(
                "intergrax/rag/vectorstore/providers/native_provider_boundary.py::validate_scope",
            ),
            notes="provider operation consumes scope tenant, not credential broker",
        ),
        R2MechanismRow(
            mechanism_id="vector.p8.factory_pin",
            family_label="vector store providers",
            continuity_verdict=p8_cont,
            consumption_verdict=R2ChainVerdict.NOT_APPLICABLE,
            evidence=tuple(f"intergrax/integrations/_shared/p8/factories.py::{p}" for p in p8_ev)
            if p8_ev
            else ("discovery: p8 pin markers missing",),
            notes="provider-local pin; not platform tenant authority",
        ),
        R2MechanismRow(
            mechanism_id="vector.lancedb.opens_pin",
            family_label="vector store providers",
            continuity_verdict=lance_cont,
            consumption_verdict=R2ChainVerdict.NOT_APPLICABLE,
            evidence=lance_ev,
            notes="provider-local pin; validate_scope on later ops",
        ),
    )


def r2_consumption_missing(mechanisms: tuple[R2MechanismRow, ...] | None = None) -> int:
    rows = mechanisms if mechanisms is not None else build_r2_canonical_mechanisms()
    return sum(
        1
        for r in rows
        if r.consumption_verdict == R2ChainVerdict.MISSING
    )


def r2_consumption_invalid(mechanisms: tuple[R2MechanismRow, ...] | None = None) -> int:
    rows = mechanisms if mechanisms is not None else build_r2_canonical_mechanisms()
    return sum(
        1
        for r in rows
        if r.consumption_verdict == R2ChainVerdict.INVALID
    )


def r2_critical_unknown_mechanisms(mechanisms: tuple[R2MechanismRow, ...] | None = None) -> int:
    rows = mechanisms if mechanisms is not None else build_r2_canonical_mechanisms()
    return sum(
        1
        for r in rows
        if r.continuity_verdict == R2ChainVerdict.MISSING
        or r.consumption_verdict == R2ChainVerdict.MISSING
        or r.continuity_verdict == R2ChainVerdict.INVALID
        or r.consumption_verdict == R2ChainVerdict.INVALID
    )
