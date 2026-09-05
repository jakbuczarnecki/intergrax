# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 3 query and scope contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryAvailabilityEvidence,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityIdentityKey,
    CapabilityKind,
    LogicalIdentityFilter,
    NORMATIVE_AVAILABILITY_DISPOSITIONS,
    SourceFilter,
)

pytestmark = pytest.mark.unit


def _enterprise_scope(
    *,
    organization_id: str = "org-acme",
    tenant_id: str = "tenant-a",
    application_id: str = "app-research",
    work_context_id: str | None = None,
) -> CapabilityDiscoveryScope:
    return CapabilityDiscoveryScope(
        organization_id=organization_id,
        tenant_id=tenant_id,
        application_id=application_id,
        work_context_id=work_context_id,
    )


def _identity_key(
    *,
    kind: CapabilityKind = CapabilityKind.TOOL,
    source_id: str = "official.catalog",
    logical_id: str = "tools.echo.ping",
) -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=kind,
        source_id=source_id,
        logical_id=logical_id,
    )


def test_enterprise_scope_requires_org_tenant_application() -> None:
    scope = _enterprise_scope()
    assert scope.mode is CapabilityDiscoveryScopeMode.ENTERPRISE
    assert scope.organization_id == "org-acme"
    assert scope.tenant_id == "tenant-a"
    assert scope.application_id == "app-research"


def test_enterprise_scope_rejects_missing_mandatory_ids() -> None:
    with pytest.raises(ValidationError, match="enterprise discovery scope requires"):
        CapabilityDiscoveryScope(
            organization_id="org-acme",
            tenant_id="tenant-a",
        )


def test_enterprise_scope_rejects_empty_scope_ids() -> None:
    with pytest.raises(ValidationError, match="scope identifier must be non-empty"):
        CapabilityDiscoveryScope(
            organization_id="  ",
            tenant_id="tenant-a",
            application_id="app-research",
        )


def test_global_scope_is_explicit_and_rejects_enterprise_ids() -> None:
    scope = CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL)
    assert scope.mode is CapabilityDiscoveryScopeMode.GLOBAL
    with pytest.raises(ValidationError, match="global discovery scope must not include"):
        CapabilityDiscoveryScope(
            mode=CapabilityDiscoveryScopeMode.GLOBAL,
            tenant_id="tenant-a",
        )


def test_discovery_query_supports_v1_capability_kinds() -> None:
    query = CapabilityDiscoveryQuery(
        scope=_enterprise_scope(),
        kinds=(CapabilityKind.AGENT, CapabilityKind.SKILL, CapabilityKind.TOOL),
    )
    assert query.kinds == (
        CapabilityKind.AGENT,
        CapabilityKind.SKILL,
        CapabilityKind.TOOL,
    )


def test_discovery_query_deduplicates_kinds() -> None:
    query = CapabilityDiscoveryQuery(
        scope=_enterprise_scope(),
        kinds=(CapabilityKind.TOOL, CapabilityKind.TOOL),
    )
    assert query.kinds == (CapabilityKind.TOOL,)


def test_logical_identity_filter_requires_constraint() -> None:
    with pytest.raises(ValidationError, match="logical identity filter requires"):
        LogicalIdentityFilter()


def test_source_filter_requires_constraint() -> None:
    with pytest.raises(ValidationError, match="source filter requires"):
        SourceFilter()


def test_discovery_query_is_frozen() -> None:
    query = CapabilityDiscoveryQuery(scope=_enterprise_scope())
    with pytest.raises(ValidationError):
        query.scope = CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL)


@pytest.mark.parametrize(
    ("model_factory", "foreign_version"),
    [
        (
            lambda: CapabilityDiscoveryScope(
                organization_id="org-acme",
                tenant_id="tenant-a",
                application_id="app-research",
            ),
            "capability_discovery_scope.v2",
        ),
        (
            lambda: CapabilityDiscoveryQuery(scope=_enterprise_scope()),
            "capability_discovery_query.v2",
        ),
        (
            lambda: CapabilityIdentityKey(
                kind=CapabilityKind.TOOL,
                source_id="official.catalog",
                logical_id="tools.echo.ping",
            ),
            "capability_identity_key.v2",
        ),
        (
            lambda: CapabilityDiscoveryAvailabilityEvidence(),
            "capability_discovery_availability_evidence.v2",
        ),
    ],
)
def test_stage3_schema_version_fail_closed(
    model_factory: object,
    foreign_version: str,
) -> None:
    model = model_factory()
    with pytest.raises(ValidationError):
        model.__class__.model_validate(
            {**model.model_dump(), "schema_version": foreign_version},
        )


def test_normative_availability_dispositions_match_contract() -> None:
    assert NORMATIVE_AVAILABILITY_DISPOSITIONS == frozenset(AvailabilityDisposition)
    assert AvailabilityDisposition.BLOCKED.value == "blocked"
    assert AvailabilityDisposition.SCOPE_UNAVAILABLE.value == "scope_unavailable"


def test_availability_evidence_rejects_duplicate_identity_keys() -> None:
    key = _identity_key()
    with pytest.raises(ValidationError, match="blocked_keys must not repeat"):
        CapabilityDiscoveryAvailabilityEvidence(
            blocked_keys=(key, key),
        )


def test_availability_constraints_deduplicated() -> None:
    query = CapabilityDiscoveryQuery(
        scope=_enterprise_scope(),
        availability_constraints=(
            AvailabilityDisposition.BLOCKED,
            AvailabilityDisposition.BLOCKED,
        ),
    )
    assert query.availability_constraints == (AvailabilityDisposition.BLOCKED,)
