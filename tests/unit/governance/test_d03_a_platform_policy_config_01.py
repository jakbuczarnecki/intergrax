# © Artur Czarnecki. All rights reserved.

"""D03-A platform activation policy. Configuration denial only.

Does not discover the repository, and does not exercise cloud IAM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.applications._shared.history_continuity_composition import (
    compose_history_continuity_authority,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.governance.history_continuity import (
    ContinuityDenialCode,
    ContinuityOutcome,
    ContinuityPermission,
    ContinuityPermissionGrant,
    ContinuityScope,
    HistoryBinding,
    HistoryContinuityActivation,
    HistoryContinuityActivationPolicy,
    HistoryContinuityAdapterKind,
    HistoryContinuityAdministrativeIdentities,
    HistoryContinuityHistoryScope,
    HistoryContinuityRetentionPolicy,
    HistoryContinuityRuntimeCredentialMode,
)
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    bundle_backed_control_plane_mutation_evaluator,
)
from intergrax.runtime.governance.local_history_continuity import (
    InactiveHistoryContinuityAuthority,
    LocalDeterministicHistoryContinuity,
    continuity_authorization_action,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

_REPO = Path(__file__).resolve().parents[3]
_ISSUER = "continuity-owner"
_APP = "app-1"
_ENV = "env-1"
_SEPARATED = tuple(ContinuityPermission)


def _boundary(
    allowed: frozenset[ContinuityPermission] | None = None,
) -> ControlPlaneMutationAuthorizationBoundary:
    permissions = set(ContinuityPermission) if allowed is None else set(allowed)
    rules = [
        PolicyBundleRule(
            rule_id=f"history_continuity.{permission.value}",
            description=f"Allow {permission.value}",
            match_action=continuity_authorization_action(permission),
            effect="allow",
        )
        for permission in sorted(permissions, key=lambda item: item.value)
    ]
    bundle = build_immutable_runtime_policy_bundle(
        bundle_id="d03a.policy.history_continuity",
        version="1.0.0",
        rules=rules,
        issued_at=datetime(2026, 10, 18, tzinfo=timezone.utc),
    )
    return ControlPlaneMutationAuthorizationBoundary(
        evaluator=bundle_backed_control_plane_mutation_evaluator(
            RuntimePolicyBundleEvaluator(bundle),
        ),
    )


def _governance(
    principal_tenant: str = "tenant-a",
    resolver_tenant: str = "tenant-a",
    *,
    allowed: frozenset[ContinuityPermission] | None = None,
) -> ProductionControlPlaneGovernance:
    return ProductionControlPlaneGovernance(
        principal=RequestIdentity(
            tenant_id=principal_tenant,
            user_id=_ISSUER,
            principal_type=PrincipalType.SERVICE,
            auth_subject=_ISSUER,
        ),
        mutation_authorization_boundary=_boundary(allowed),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id=resolver_tenant,
        ),
    )


def _retention() -> HistoryContinuityRetentionPolicy:
    return HistoryContinuityRetentionPolicy(
        retention_seconds=3600,
        ceiling_seconds=7200,
        legal_hold=True,
    )


def _administrators(
    primary: str = "primary-admin",
    witness: str = "witness-admin",
) -> HistoryContinuityAdministrativeIdentities:
    return HistoryContinuityAdministrativeIdentities(
        primary_administrator_id=primary,
        witness_administrator_id=witness,
    )


def _policy(**updates: object) -> HistoryContinuityActivationPolicy:
    payload: dict[str, object] = {
        "history_scope": HistoryContinuityHistoryScope.ENVIRONMENT,
        "administrators": _administrators(),
        "runtime_credential_mode": (
            HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL
        ),
        "separated_capabilities": _SEPARATED,
    }
    payload.update(updates)
    return HistoryContinuityActivationPolicy.model_validate(payload)


def _activation(**updates: object) -> HistoryContinuityActivation:
    payload: dict[str, object] = {
        "enabled": True,
        "adapter": HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
        "application_id": _APP,
        "environment_id": _ENV,
        "retention": _retention(),
        "policy": _policy(),
    }
    payload.update(updates)
    return HistoryContinuityActivation.model_validate(payload)


def _grant(permission: ContinuityPermission) -> ContinuityPermissionGrant:
    return ContinuityPermissionGrant(principal_id=_ISSUER, permission=permission)


def _scope(tenant_id: str = "tenant-a") -> ContinuityScope:
    return ContinuityScope(
        application_id=_APP,
        environment_id=_ENV,
        tenant_id=tenant_id,
    )


def _binding() -> HistoryBinding:
    return HistoryBinding(
        pis_history_id="pis-h",
        pis_committed_watermark=1,
        gca_history_id="gca-h",
        gca_committed_watermark=1,
    )


def _assert_inactive(authority: object) -> None:
    assert isinstance(authority, InactiveHistoryContinuityAuthority)
    assert not isinstance(authority, LocalDeterministicHistoryContinuity)
    assert authority.sealed_retention() is None
    assert (
        authority.initialize(
            _grant(ContinuityPermission.INITIALIZE),
            _scope(),
            _binding(),
        ).denial
        is ContinuityDenialCode.ACTIVATION_ABSENT
    )


def test_missing_activation_returns_inactive_authority() -> None:
    _assert_inactive(compose_history_continuity_authority(_governance(), None))
    with pytest.raises(ValidationError):
        HistoryContinuityActivation(
            enabled=True,
            adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
            application_id=_APP,
            environment_id=_ENV,
            retention=_retention(),
        )


def test_disabled_and_invalid_activation_deny() -> None:
    with pytest.raises(ValidationError):
        _activation(enabled=False)
    bypassed = HistoryContinuityActivation.model_construct(
        enabled=False,
        adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
        application_id=_APP,
        environment_id=_ENV,
        retention=_retention(),
        policy=_policy(),
    )
    _assert_inactive(compose_history_continuity_authority(_governance(), bypassed))


def test_unsupported_adapter_denies() -> None:
    with pytest.raises(ValidationError):
        _activation(adapter="unsupported-adapter")
    bypassed = HistoryContinuityActivation.model_construct(
        enabled=True,
        adapter="unsupported-adapter",
        application_id=_APP,
        environment_id=_ENV,
        retention=_retention(),
        policy=_policy(),
    )
    _assert_inactive(compose_history_continuity_authority(_governance(), bypassed))
    assert list(HistoryContinuityAdapterKind) == [
        HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
        HistoryContinuityAdapterKind.OFFLINE_EXTERNAL_PAIR,
    ]


def test_tenant_specific_history_scope_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _policy(history_scope=HistoryContinuityHistoryScope.TENANT)
    with pytest.raises(ValidationError):
        _policy(history_scope="global")
    bypassed_policy = HistoryContinuityActivationPolicy.model_construct(
        history_scope=HistoryContinuityHistoryScope.TENANT,
        administrators=_administrators(),
        runtime_credential_mode=(
            HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL
        ),
        separated_capabilities=_SEPARATED,
    )
    bypassed = HistoryContinuityActivation.model_construct(
        enabled=True,
        adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
        application_id=_APP,
        environment_id=_ENV,
        retention=_retention(),
        policy=bypassed_policy,
    )
    _assert_inactive(compose_history_continuity_authority(_governance(), bypassed))


def test_environment_tenant_mismatch_denies_without_cross_tenant_access() -> None:
    mismatched = compose_history_continuity_authority(
        _governance("tenant-b", "tenant-a"),
        _activation(),
    )
    _assert_inactive(mismatched)

    authority = compose_history_continuity_authority(_governance(), _activation())
    assert isinstance(authority, LocalDeterministicHistoryContinuity)
    denied = authority.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope("tenant-b"),
        _binding(),
    )
    assert denied.denial is ContinuityDenialCode.TENANT_MISMATCH
    assert authority.sealed_retention() == _retention()


def test_primary_and_witness_administrator_identities_must_be_distinct() -> None:
    with pytest.raises(ValidationError):
        _administrators(primary="same-admin", witness="same-admin")
    with pytest.raises(ValidationError):
        _administrators(primary="Same-Admin", witness="same-admin")
    with pytest.raises(ValidationError):
        _administrators(primary="  primary-admin", witness="witness-admin")
    with pytest.raises(ValidationError):
        _administrators(primary="   ", witness="witness-admin")
    with pytest.raises(ValidationError):
        HistoryContinuityAdministrativeIdentities(
            primary_administrator_id="primary-admin",
        )
    equal = HistoryContinuityAdministrativeIdentities.model_construct(
        primary_administrator_id="same-admin",
        witness_administrator_id="same-admin",
    )
    policy = HistoryContinuityActivationPolicy.model_construct(
        history_scope=HistoryContinuityHistoryScope.ENVIRONMENT,
        administrators=equal,
        runtime_credential_mode=(
            HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL
        ),
        separated_capabilities=_SEPARATED,
    )
    activation = HistoryContinuityActivation.model_construct(
        enabled=True,
        adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
        application_id=_APP,
        environment_id=_ENV,
        retention=_retention(),
        policy=policy,
    )
    _assert_inactive(compose_history_continuity_authority(_governance(), activation))


def test_unsupported_runtime_credential_modes_deny() -> None:
    rejected = (
        HistoryContinuityRuntimeCredentialMode.STATIC_LONG_LIVED,
        HistoryContinuityRuntimeCredentialMode.ADMINISTRATOR,
        HistoryContinuityRuntimeCredentialMode.WORKLOAD_FEDERATION,
    )
    for mode in rejected:
        with pytest.raises(ValidationError):
            _policy(runtime_credential_mode=mode)
        policy = HistoryContinuityActivationPolicy.model_construct(
            history_scope=HistoryContinuityHistoryScope.ENVIRONMENT,
            administrators=_administrators(),
            runtime_credential_mode=mode,
            separated_capabilities=_SEPARATED,
        )
        activation = HistoryContinuityActivation.model_construct(
            enabled=True,
            adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
            application_id=_APP,
            environment_id=_ENV,
            retention=_retention(),
            policy=policy,
        )
        _assert_inactive(compose_history_continuity_authority(_governance(), activation))
    with pytest.raises(ValidationError):
        _policy(secret="not-a-credential")
    assert "secret" not in HistoryContinuityActivationPolicy.model_fields
    assert "token" not in HistoryContinuityActivationPolicy.model_fields
    assert "shared_key" not in HistoryContinuityActivationPolicy.model_fields


def test_operational_capabilities_stay_distinct_and_require_governance_authorization() -> None:
    assert {item.value for item in ContinuityPermission} == {
        "write",
        "initialize",
        "reconcile",
        "invalidate",
        "recover",
    }
    with pytest.raises(ValidationError):
        _policy(separated_capabilities=(ContinuityPermission.WRITE,))
    with pytest.raises(ValidationError):
        _policy(separated_capabilities=_SEPARATED + (ContinuityPermission.WRITE,))
    with pytest.raises(ValidationError):
        _policy(administrator_grant=True)

    collapsed = HistoryContinuityActivationPolicy.model_construct(
        history_scope=HistoryContinuityHistoryScope.ENVIRONMENT,
        administrators=_administrators(),
        runtime_credential_mode=(
            HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL
        ),
        separated_capabilities=(ContinuityPermission.WRITE,),
    )
    collapsed_activation = HistoryContinuityActivation.model_construct(
        enabled=True,
        adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
        application_id=_APP,
        environment_id=_ENV,
        retention=_retention(),
        policy=collapsed,
    )
    _assert_inactive(
        compose_history_continuity_authority(_governance(), collapsed_activation),
    )

    limited = compose_history_continuity_authority(
        _governance(allowed=frozenset({ContinuityPermission.WRITE})),
        _activation(),
    )
    assert isinstance(limited, LocalDeterministicHistoryContinuity)
    assert (
        limited.initialize(
            _grant(ContinuityPermission.INITIALIZE),
            _scope(),
            _binding(),
        ).denial
        is ContinuityDenialCode.PERMISSION_DENIED
    )
    forged = ContinuityPermissionGrant.model_construct(
        principal_id=_ISSUER,
        permission="history_continuity.write",
    )
    assert (
        limited.initialize(forged, _scope(), _binding()).denial
        is ContinuityDenialCode.PERMISSION_DENIED
    )


def test_retention_weakening_stays_forbidden() -> None:
    with pytest.raises(ValidationError):
        HistoryContinuityRetentionPolicy(
            retention_seconds=10,
            ceiling_seconds=5,
            legal_hold=True,
        )
    with pytest.raises(ValidationError):
        HistoryContinuityRetentionPolicy(
            retention_seconds=10,
            ceiling_seconds=10,
            legal_hold=False,
        )
    weakened = HistoryContinuityRetentionPolicy.model_construct(
        retention_seconds=10,
        ceiling_seconds=5,
        legal_hold=True,
    )
    bypassed = HistoryContinuityActivation.model_construct(
        enabled=True,
        adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
        application_id=_APP,
        environment_id=_ENV,
        retention=weakened,
        policy=_policy(),
    )
    _assert_inactive(compose_history_continuity_authority(_governance(), bypassed))

    authority = compose_history_continuity_authority(_governance(), _activation())
    assert isinstance(authority, LocalDeterministicHistoryContinuity)
    opened = authority.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope(),
        _binding(),
    )
    assert opened.outcome is ContinuityOutcome.COMMITTED
    sealed = authority.sealed_retention()
    assert sealed == _retention()
    assert (
        authority.attempt_retention_change(
            _grant(ContinuityPermission.WRITE),
            _scope(),
            retention_seconds=3599,
            ceiling_seconds=7200,
            legal_hold=True,
        ).denial
        is ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN
    )
    assert (
        authority.attempt_retention_change(
            _grant(ContinuityPermission.WRITE),
            _scope(),
            retention_seconds=3600,
            ceiling_seconds=7199,
            legal_hold=True,
        ).denial
        is ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN
    )
    assert (
        authority.attempt_retention_change(
            _grant(ContinuityPermission.WRITE),
            _scope(),
            retention_seconds=3600,
            ceiling_seconds=7200,
            legal_hold=False,
        ).denial
        is ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN
    )
    assert authority.sealed_retention() == sealed


def test_valid_environment_policy_preserves_local_activation() -> None:
    authority = compose_history_continuity_authority(_governance(), _activation())
    assert isinstance(authority, LocalDeterministicHistoryContinuity)
    opened = authority.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope(),
        _binding(),
    )
    assert opened.outcome is ContinuityOutcome.COMMITTED
    assert opened.head is not None
    assert opened.head.revision == 1
    assert opened.head.tenant_id == "tenant-a"


def test_contract_has_no_vendor_types_and_create_gate_is_not_a_caller() -> None:
    sources = (
        _REPO / "intergrax" / "contracts" / "governance" / "history_continuity.py",
        _REPO / "intergrax" / "applications" / "_shared" / "history_continuity_composition.py",
        _REPO / "intergrax" / "runtime" / "governance" / "local_history_continuity.py",
    )
    for path in sources:
        lowered = path.read_text(encoding="utf-8").lower()
        assert "spanner" not in lowered
        assert "azure" not in lowered
        assert "google.cloud" not in lowered
    guarded = (
        _REPO
        / "applications"
        / "governed_contractor_application"
        / "host"
        / "provider_invocation_lifecycle.py",
        _REPO / "intergrax" / "applications" / "_shared" / "production_lifecycle.py",
        _REPO / "intergrax" / "applications" / "_shared" / "production_process_lifecycle.py",
    )
    for path in guarded:
        text = path.read_text(encoding="utf-8")
        assert "history_continuity" not in text
        assert "compose_history_continuity_authority" not in text
