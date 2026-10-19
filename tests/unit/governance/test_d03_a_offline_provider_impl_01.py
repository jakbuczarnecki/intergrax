# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""D03-A offline Spanner/Azure provider stack. Not cloud qualification."""

from __future__ import annotations

import inspect
import sys
import threading
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.applications._shared import history_continuity_composition as composition
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
    HistoryContinuityCloudQualification,
    HistoryContinuityExecutionDomain,
    HistoryContinuityHistoryScope,
    HistoryContinuityIdentityBinding,
    HistoryContinuityProviderSelection,
    HistoryContinuityRetentionPolicy,
    HistoryContinuityRuntimeCredentialMode,
    history_continuity_activation_is_admitted,
    history_continuity_offline_provider_execution_permitted,
)
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.governance.continuity_providers.azure_sdk import (
    open_azure_append_client,
)
from intergrax.runtime.governance.continuity_providers.azure_sdk import (
    cloud_execution_authorized as azure_cloud_authorized,
)
from intergrax.runtime.governance.continuity_providers.credentials import (
    OfflineProviderRefusal,
)
from intergrax.runtime.governance.continuity_providers.errors import (
    map_transport_exception,
    ProviderFault,
)
from intergrax.runtime.governance.continuity_providers.primary_register import (
    InMemoryPrimaryDatabase,
    PrimaryCommitKind,
    SpannerPrimaryRegister,
    TransitionRecord,
    request_digest,
)
from intergrax.runtime.governance.continuity_providers.spanner_sdk import (
    open_spanner_sdk_database,
)
from intergrax.runtime.governance.continuity_providers.spanner_sdk import (
    cloud_execution_authorized as spanner_cloud_authorized,
)
from intergrax.runtime.governance.continuity_providers.targets import (
    AzureWitnessTarget,
    SpannerPrimaryTarget,
)
from intergrax.runtime.governance.continuity_providers.witness_log import (
    AzureAppendWitness,
    InMemoryAppendWitness,
    decode_block,
    encode_block,
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
from intergrax.runtime.governance.offline_history_continuity import (
    OfflineProviderBundle,
    OfflineSpannerAzureHistoryContinuity,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

_ISSUER = "continuity-owner"
_APP = "app-1"
_ENV = "env-1"
_BLOCK = 4096


def _boundary() -> ControlPlaneMutationAuthorizationBoundary:
    rules = [
        PolicyBundleRule(
            rule_id=f"history_continuity.{permission.value}",
            description=f"Allow {permission.value}",
            match_action=continuity_authorization_action(permission),
            effect="allow",
        )
        for permission in sorted(ContinuityPermission, key=lambda item: item.value)
    ]
    bundle = build_immutable_runtime_policy_bundle(
        bundle_id="d03a.offline.history_continuity",
        version="1.0.0",
        rules=rules,
        issued_at=datetime(2026, 10, 19, tzinfo=timezone.utc),
    )
    return ControlPlaneMutationAuthorizationBoundary(
        evaluator=bundle_backed_control_plane_mutation_evaluator(
            RuntimePolicyBundleEvaluator(bundle),
        ),
    )


def _governance(
    principal_tenant: str = "tenant-a",
    resolver_tenant: str = "tenant-a",
) -> ProductionControlPlaneGovernance:
    return ProductionControlPlaneGovernance(
        principal=RequestIdentity(
            tenant_id=principal_tenant,
            user_id=_ISSUER,
            principal_type=PrincipalType.SERVICE,
            auth_subject=_ISSUER,
        ),
        mutation_authorization_boundary=_boundary(),
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


def _binding(pis: int = 1, gca: int = 1) -> HistoryBinding:
    return HistoryBinding(
        pis_history_id="pis-h",
        pis_committed_watermark=pis,
        gca_history_id="gca-h",
        gca_committed_watermark=gca,
    )


def _scope(
    tenant_id: str = "tenant-a",
    *,
    environment_id: str = _ENV,
    application_id: str = _APP,
) -> ContinuityScope:
    return ContinuityScope(
        application_id=application_id,
        environment_id=environment_id,
        tenant_id=tenant_id,
    )


def _grant(permission: ContinuityPermission) -> ContinuityPermissionGrant:
    return ContinuityPermissionGrant(principal_id=_ISSUER, permission=permission)


def _binding_record(
    mode: HistoryContinuityRuntimeCredentialMode = (
        HistoryContinuityRuntimeCredentialMode.WORKLOAD_FEDERATION
    ),
) -> HistoryContinuityIdentityBinding:
    return HistoryContinuityIdentityBinding(
        binding_id="offline-binding-1",
        credential_mode=mode,
        subject="offline-subject",
        audience="offline-audience",
    )


def _selection(
    *,
    environment_id: str = _ENV,
    binding: HistoryContinuityIdentityBinding | None = None,
    include_binding: bool = True,
) -> HistoryContinuityProviderSelection:
    return HistoryContinuityProviderSelection(
        adapter=HistoryContinuityAdapterKind.OFFLINE_EXTERNAL_PAIR,
        execution_domain=HistoryContinuityExecutionDomain.OFFLINE_CONTRACT,
        cloud_qualification=HistoryContinuityCloudQualification.UNQUALIFIED,
        application_id=_APP,
        environment_id=environment_id,
        retention=_retention(),
        administrators=HistoryContinuityAdministrativeIdentities(
            primary_administrator_id="primary-admin",
            witness_administrator_id="witness-admin",
        ),
        identity_binding=_binding_record() if include_binding and binding is None else binding,
    )


def _targets() -> tuple[SpannerPrimaryTarget, AzureWitnessTarget]:
    return (
        SpannerPrimaryTarget(
            project_id="offline-project",
            instance_id="offline-instance",
            database_id="offline-database",
            edition="enterprise_plus",
            instance_configuration="eur3",
        ),
        AzureWitnessTarget(
            region="northeurope",
            account_sku="Standard_RAGZRS",
            account_kind="StorageV2",
            access_tier="Hot",
        ),
    )


def _bundle(
    *,
    primary: InMemoryPrimaryDatabase | None = None,
    witness: InMemoryAppendWitness | None = None,
    segment_block_limit: int = 49_000,
    recovery_block_bound: int = 128,
) -> OfflineProviderBundle:
    primary_target, witness_target = _targets()
    return OfflineProviderBundle(
        primary=primary or InMemoryPrimaryDatabase(),
        witness=witness or InMemoryAppendWitness(),
        primary_target=primary_target,
        witness_target=witness_target,
        segment_block_limit=segment_block_limit,
        recovery_block_bound=recovery_block_bound,
    )


def _authority(
    bundle: OfflineProviderBundle | None = None,
    *,
    governance: ProductionControlPlaneGovernance | None = None,
    selection: HistoryContinuityProviderSelection | None = None,
) -> OfflineSpannerAzureHistoryContinuity:
    authority = compose_history_continuity_authority(
        _governance() if governance is None else governance,
        None,
        provider_selection=_selection() if selection is None else selection,
        offline_bundle=_bundle() if bundle is None else bundle,
    )
    assert isinstance(authority, OfflineSpannerAzureHistoryContinuity)
    return authority


def _initialize(authority: OfflineSpannerAzureHistoryContinuity):
    result = authority.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope(),
        _binding(),
    )
    assert result.outcome is ContinuityOutcome.COMMITTED
    assert result.head is not None
    assert result.head.revision == 1
    return result.head


def _advance(authority, head, *, pis: int = 2, gca: int = 2, fence: str | None = None):
    return authority.advance(
        _grant(ContinuityPermission.WRITE),
        _scope(),
        expected_revision=head.revision,
        expected_fencing_token=head.fencing_token if fence is None else fence,
        binding=_binding(pis, gca),
    )


def test_spanner_azure_name_does_not_admit_cloud_activation() -> None:
    with pytest.raises(ValidationError):
        HistoryContinuityActivation(
            enabled=True,
            adapter=HistoryContinuityAdapterKind.OFFLINE_EXTERNAL_PAIR,
            application_id=_APP,
            environment_id=_ENV,
            retention=_retention(),
            policy=HistoryContinuityActivationPolicy(
                history_scope=HistoryContinuityHistoryScope.ENVIRONMENT,
                administrators=HistoryContinuityAdministrativeIdentities(
                    primary_administrator_id="primary-admin",
                    witness_administrator_id="witness-admin",
                ),
                runtime_credential_mode=(
                    HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL
                ),
                separated_capabilities=tuple(ContinuityPermission),
            ),
        )
    bypassed = HistoryContinuityActivation.model_construct(
        enabled=True,
        adapter=HistoryContinuityAdapterKind.OFFLINE_EXTERNAL_PAIR,
        application_id=_APP,
        environment_id=_ENV,
        retention=_retention(),
        policy=HistoryContinuityActivationPolicy(
            history_scope=HistoryContinuityHistoryScope.ENVIRONMENT,
            administrators=HistoryContinuityAdministrativeIdentities(
                primary_administrator_id="primary-admin",
                witness_administrator_id="witness-admin",
            ),
            runtime_credential_mode=(
                HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL
            ),
            separated_capabilities=tuple(ContinuityPermission),
        ),
    )
    assert history_continuity_activation_is_admitted(bypassed) is False
    inactive = compose_history_continuity_authority(_governance(), bypassed)
    assert isinstance(inactive, InactiveHistoryContinuityAuthority)
    assert inactive.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope(),
        _binding(),
    ).denial is ContinuityDenialCode.ACTIVATION_ABSENT


def test_missing_binding_and_unqualified_selection_deny() -> None:
    missing = compose_history_continuity_authority(
        _governance(),
        None,
        provider_selection=_selection(include_binding=False),
        offline_bundle=_bundle(),
    )
    assert isinstance(missing, InactiveHistoryContinuityAuthority)
    static = _selection(
        binding=_binding_record(HistoryContinuityRuntimeCredentialMode.STATIC_LONG_LIVED),
    )
    assert history_continuity_offline_provider_execution_permitted(static) is False
    claimed = HistoryContinuityProviderSelection.model_construct(
        adapter=HistoryContinuityAdapterKind.OFFLINE_EXTERNAL_PAIR,
        execution_domain=HistoryContinuityExecutionDomain.OFFLINE_CONTRACT,
        cloud_qualification="qualified",
        application_id=_APP,
        environment_id=_ENV,
        retention=_retention(),
        administrators=HistoryContinuityAdministrativeIdentities(
            primary_administrator_id="primary-admin",
            witness_administrator_id="witness-admin",
        ),
        identity_binding=_binding_record(),
    )
    assert history_continuity_offline_provider_execution_permitted(claimed) is False
    mixed = compose_history_continuity_authority(
        _governance(),
        HistoryContinuityActivation(
            enabled=True,
            adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
            application_id=_APP,
            environment_id=_ENV,
            retention=_retention(),
            policy=HistoryContinuityActivationPolicy(
                history_scope=HistoryContinuityHistoryScope.ENVIRONMENT,
                administrators=HistoryContinuityAdministrativeIdentities(
                    primary_administrator_id="primary-admin",
                    witness_administrator_id="witness-admin",
                ),
                runtime_credential_mode=(
                    HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL
                ),
                separated_capabilities=tuple(ContinuityPermission),
            ),
        ),
        provider_selection=_selection(),
        offline_bundle=_bundle(),
    )
    assert isinstance(mixed, InactiveHistoryContinuityAuthority)
    assert mixed.sealed_retention() is None


def test_sdk_boundary_refuses_live_clients_and_retention_duration() -> None:
    assert spanner_cloud_authorized() is False
    assert azure_cloud_authorized() is False
    primary, witness = _targets()
    binding = _binding_record()
    with pytest.raises(OfflineProviderRefusal):
        open_spanner_sdk_database(primary, binding)
    with pytest.raises(OfflineProviderRefusal):
        open_azure_append_client(witness, binding)
    with pytest.raises(OfflineProviderRefusal):
        open_spanner_sdk_database(primary, None)
    assert "google.cloud" not in sys.modules
    assert "azure.storage" not in sys.modules
    witness_payload = {
        "region": "northeurope",
        "account_sku": "Standard_RAGZRS",
        "account_kind": "StorageV2",
        "access_tier": "Hot",
    }
    with pytest.raises(ValidationError):
        AzureWitnessTarget.model_validate({**witness_payload, "immutability_period_days": 3650})
    with pytest.raises(ValidationError):
        AzureWitnessTarget.model_validate({**witness_payload, "protected_append": "qualified"})

    class DeadlineExceeded(Exception):
        pass

    class HttpResponseError(Exception):
        def __init__(self) -> None:
            super().__init__("precondition")
            self.status_code = 412

    assert map_transport_exception(DeadlineExceeded()) is ProviderFault.OUTCOME_UNKNOWN
    assert map_transport_exception(HttpResponseError()) is ProviderFault.APPEND_POSITION_CONFLICT


def test_sanctioned_composition_selects_local_or_offline_pair() -> None:
    source = inspect.getsource(composition)
    assert "GovernedProviderInvocationDispatchGate" not in source
    assert "google.cloud" not in source
    assert "azure.storage" not in source
    assert source.count("def compose_history_continuity_authority") == 1
    local = compose_history_continuity_authority(
        _governance(),
        HistoryContinuityActivation(
            enabled=True,
            adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
            application_id=_APP,
            environment_id=_ENV,
            retention=_retention(),
            policy=HistoryContinuityActivationPolicy(
                history_scope=HistoryContinuityHistoryScope.ENVIRONMENT,
                administrators=HistoryContinuityAdministrativeIdentities(
                    primary_administrator_id="primary-admin",
                    witness_administrator_id="witness-admin",
                ),
                runtime_credential_mode=(
                    HistoryContinuityRuntimeCredentialMode.LOCAL_NO_RUNTIME_CREDENTIAL
                ),
                separated_capabilities=tuple(ContinuityPermission),
            ),
        ),
    )
    offline = _authority()
    assert isinstance(local, LocalDeterministicHistoryContinuity)
    assert type(local) is not type(offline)
    for name in (
        "initialize",
        "advance",
        "reconcile",
        "invalidate",
        "recover",
        "read_head",
        "attempt_retention_change",
        "sealed_retention",
    ):
        assert callable(getattr(local, name))
        assert callable(getattr(offline, name))
    with pytest.raises(RuntimeError, match="sanctioned composition"):
        OfflineSpannerAzureHistoryContinuity(
            tenant_resolver=StaticApplicationEnvironmentTenantResolver(tenant_id="tenant-a"),
            selection=_selection(),
            authorization_boundary=_boundary(),
            principal=RequestIdentity(
                tenant_id="tenant-a",
                user_id=_ISSUER,
                principal_type=PrincipalType.SERVICE,
                auth_subject=_ISSUER,
            ),
            issuer_principal_id=_ISSUER,
            bundle=_bundle(),
            composition_token=object(),
        )


def test_offline_initialize_advance_and_retention_do_not_weaken() -> None:
    bundle = _bundle()
    authority = _authority(bundle)
    head = _initialize(authority)
    current = authority.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert current.witness_high_watermark == 1
    advanced = _advance(authority, head)
    assert advanced.outcome is ContinuityOutcome.COMMITTED
    assert advanced.head is not None
    assert advanced.head.revision == 2
    assert advanced.witness_high_watermark == 2
    assert bundle.witness.length(0) == 2 * _BLOCK
    raw = bundle.witness.read_range(0, 0, _BLOCK)
    assert raw is not None and len(raw) == _BLOCK
    assert len(encode_block({"kind": "probe"})) == _BLOCK
    weakened = authority.attempt_retention_change(
        _grant(ContinuityPermission.WRITE),
        _scope(),
        retention_seconds=3599,
        ceiling_seconds=7200,
        legal_hold=True,
    )
    assert weakened.denial is ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN
    assert authority.sealed_retention() == _retention()


def test_concurrent_cas_and_stale_fence() -> None:
    authority = _authority()
    head = _initialize(authority)
    barrier = threading.Barrier(8)
    results: list[object] = []
    lock = threading.Lock()

    def _worker() -> None:
        barrier.wait()
        result = _advance(authority, head)
        with lock:
            results.append(result)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    committed = [
        result
        for result in results
        if getattr(result, "outcome", None) is ContinuityOutcome.COMMITTED
    ]
    stale = [
        result
        for result in results
        if getattr(result, "denial", None) is ContinuityDenialCode.REVISION_STALE
    ]
    assert len(committed) == 1
    assert len(stale) == 7
    current = authority.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert current.head is not None and current.head.revision == 2
    fenced = _advance(authority, current.head, fence="stale-fence", pis=3, gca=3)
    assert fenced.denial is ContinuityDenialCode.FENCE_STALE
    reread = authority.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert reread.head is not None and reread.head.revision == 2


def test_register_cas_is_transactional_and_idempotent() -> None:
    bundle = _bundle()
    authority = _authority(bundle)
    head = _initialize(authority)
    register = SpannerPrimaryRegister()
    scope = (_APP, _ENV, "tenant-a")
    first = _sample_transition(revision=2, fence=head.fencing_token, watermark=2)
    other = _sample_transition(revision=2, fence=head.fencing_token, watermark=9)
    won = register.prepare(
        bundle.primary,
        application_id=scope[0],
        environment_id=scope[1],
        tenant_id=scope[2],
        expected_revision=1,
        expected_fencing_token=head.fencing_token,
        transition=first,
        authority_incarnation="offline-incarnation:1",
    )
    assert won.kind is PrimaryCommitKind.COMMITTED
    again = register.prepare(
        bundle.primary,
        application_id=scope[0],
        environment_id=scope[1],
        tenant_id=scope[2],
        expected_revision=1,
        expected_fencing_token=head.fencing_token,
        transition=first,
        authority_incarnation="offline-incarnation:1",
    )
    assert again.kind is PrimaryCommitKind.COMMITTED
    mismatched = register.prepare(
        bundle.primary,
        application_id=scope[0],
        environment_id=scope[1],
        tenant_id=scope[2],
        expected_revision=1,
        expected_fencing_token=head.fencing_token,
        transition=other,
        authority_incarnation="offline-incarnation:1",
    )
    assert mismatched.fault is ProviderFault.DIGEST_MISMATCH
    isolated = bundle.primary.read_head(_APP, _ENV, "tenant-b")
    assert isolated is None


def test_unknown_primary_and_witness_outcomes_reconcile_once() -> None:
    bundle = _bundle()
    authority = _authority(bundle)
    head = _initialize(authority)
    bundle.primary.arm_fault("unknown_before")
    unresolved = _advance(authority, head)
    assert unresolved.denial is ContinuityDenialCode.TRANSITION_UNRESOLVED
    stored = bundle.primary.read_head(_APP, _ENV, "tenant-a")
    assert stored is not None and stored.phase == "committed" and stored.revision == 1
    assert bundle.primary.read_transition(_APP, _ENV, "tenant-a", f"{_ENV}:2:{head.fencing_token}") is None
    bundle.primary.clear_fault()
    committed = _advance(authority, head)
    assert committed.outcome is ContinuityOutcome.COMMITTED
    assert committed.head is not None and committed.head.revision == 2
    assert bundle.witness.length(0) == 2 * _BLOCK

    bundle = _bundle()
    authority = _authority(bundle)
    head = _initialize(authority)
    bundle.primary.arm_fault("unknown_after")
    bundle.witness.arm_fault("unavailable")
    paused = _advance(authority, head)
    assert paused.denial is ContinuityDenialCode.RECONCILIATION_UNAVAILABLE
    prepared = bundle.primary.read_head(_APP, _ENV, "tenant-a")
    assert prepared is not None and prepared.phase == "prepared" and prepared.revision == 1
    assert bundle.witness.length(0) == _BLOCK
    bundle.primary.clear_fault()
    bundle.witness.clear_fault()
    incarnation = prepared.authority_incarnation
    recovered = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert recovered.outcome is ContinuityOutcome.RECOVERED
    assert recovered.head is not None and recovered.head.revision == 2
    assert bundle.witness.length(0) == 2 * _BLOCK
    finished_head = bundle.primary.read_head(_APP, _ENV, "tenant-a")
    assert finished_head is not None
    assert finished_head.authority_incarnation == incarnation
    again = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert again.outcome is ContinuityOutcome.RECOVERED
    assert again.head is not None and again.head.revision == 2
    assert bundle.witness.length(0) == 2 * _BLOCK

    bundle = _bundle()
    authority = _authority(bundle)
    head = _initialize(authority)
    bundle.witness.arm_fault("unknown_after")
    hidden = _advance(authority, head)
    assert hidden.outcome is ContinuityOutcome.COMMITTED
    assert bundle.witness.length(0) == 2 * _BLOCK

    bundle = _bundle()
    authority = _authority(bundle)
    head = _initialize(authority)
    bundle.witness.arm_fault("unknown_before")
    lost = _advance(authority, head)
    assert lost.denial is ContinuityDenialCode.TRANSITION_UNRESOLVED
    assert bundle.witness.length(0) == _BLOCK
    bundle.witness.clear_fault()
    restored = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert restored.outcome is ContinuityOutcome.RECOVERED
    assert restored.head is not None and restored.head.revision == 2
    assert bundle.witness.length(0) == 2 * _BLOCK
    reread = authority.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert reread.outcome is ContinuityOutcome.COMMITTED
    assert reread.head is not None and reread.head.revision == 2


def test_rollback_corrupt_witness_and_missing_prefix_do_not_promote() -> None:
    bundle = _bundle()
    authority = _authority(bundle)
    head = _initialize(authority)
    advanced = _advance(authority, head)
    assert advanced.head is not None
    committed_head = bundle.primary.read_head(_APP, _ENV, "tenant-a")
    assert committed_head is not None
    incarnation = committed_head.authority_incarnation
    bundle.primary.testing_rollback_head(_APP, _ENV, "tenant-a")
    denied = authority.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert denied.outcome is ContinuityOutcome.DENIED
    assert denied.denial is ContinuityDenialCode.CONTINUITY_REVISION_STALE
    recovered = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert recovered.outcome is ContinuityOutcome.DENIED
    assert recovered.denial is ContinuityDenialCode.CONTINUITY_REVISION_STALE
    rolled = bundle.primary.read_head(_APP, _ENV, "tenant-a")
    assert rolled is not None
    assert rolled.revision == 1
    assert rolled.authority_incarnation == incarnation

    bundle = _bundle()
    authority = _authority(bundle)
    head = _initialize(authority)
    _advance(authority, head)
    bundle.witness.testing_truncate(0, _BLOCK)
    denied = authority.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert denied.denial is ContinuityDenialCode.WITNESS_MISMATCH
    recovered = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert recovered.outcome is ContinuityOutcome.DENIED
    truncated = bundle.primary.read_head(_APP, _ENV, "tenant-a")
    assert truncated is not None and truncated.revision == 2

    bundle.witness.testing_replace_block(0, 0, bytes(_BLOCK))
    corrupt = authority.read_head(_grant(ContinuityPermission.RECOVER), _scope())
    assert corrupt.denial is ContinuityDenialCode.WITNESS_MISMATCH

    bundle = _bundle()
    authority = _authority(bundle)
    head = _initialize(authority)
    _advance(authority, head)
    bundle.witness.testing_cut_prefix(0, _BLOCK)
    missing = authority.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert missing.denial is ContinuityDenialCode.WITNESS_MISMATCH
    assert authority.recover(_grant(ContinuityPermission.RECOVER), _scope()).outcome is (
        ContinuityOutcome.DENIED
    )
    cut = bundle.primary.read_head(_APP, _ENV, "tenant-a")
    assert cut is not None and cut.revision == 2


def test_scope_isolation_and_bounded_recovery() -> None:
    primary = InMemoryPrimaryDatabase()
    left = _authority(
        _bundle(primary=primary),
        selection=_selection(environment_id="env-left"),
    )
    right = _authority(
        _bundle(primary=primary),
        selection=_selection(environment_id="env-right"),
    )
    assert left.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope(environment_id="env-left"),
        _binding(),
    ).outcome is ContinuityOutcome.COMMITTED
    assert right.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope(environment_id="env-right"),
        _binding(),
    ).outcome is ContinuityOutcome.COMMITTED
    assert primary.read_head(_APP, "env-left", "tenant-b") is None
    right_head = primary.read_head(_APP, "env-right", "tenant-a")
    assert right_head is not None and right_head.revision == 1
    denied = left.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope(tenant_id="tenant-b", environment_id="env-left"),
        _binding(),
    )
    assert denied.denial is ContinuityDenialCode.TENANT_MISMATCH
    assert left.advance(
        _grant(ContinuityPermission.WRITE),
        _scope(environment_id="env-other"),
        expected_revision=1,
        expected_fencing_token="1",
        binding=_binding(2, 2),
    ).denial is ContinuityDenialCode.ENVIRONMENT_MISMATCH

    bounded = _authority(_bundle(recovery_block_bound=2))
    head = _initialize(bounded)
    second = _advance(bounded, head)
    assert second.outcome is ContinuityOutcome.COMMITTED
    assert second.head is not None
    third = _advance(bounded, second.head, pis=3, gca=3)
    assert third.denial is ContinuityDenialCode.RECONCILIATION_UNAVAILABLE
    bounded_head = bounded.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert bounded_head.head is not None and bounded_head.head.revision == 2


def test_segment_rollover_and_incomplete_successor_recovery() -> None:
    bundle = _bundle(segment_block_limit=3, recovery_block_bound=32)
    authority = _authority(bundle)
    head = _initialize(authority)
    second = _advance(authority, head)
    assert second.outcome is ContinuityOutcome.COMMITTED
    assert second.head is not None
    bundle.witness.arm_fault("unknown_before", call_index=2)
    crashed = _advance(authority, second.head, pis=3, gca=3)
    assert crashed.denial is ContinuityDenialCode.TRANSITION_UNRESOLVED
    assert bundle.witness.length(1) == 0
    closed = decode_block(bundle.witness.read_range(0, 2 * _BLOCK, _BLOCK) or b"")
    assert closed is not None and closed["kind"] == "segment_closed"
    prepared = bundle.primary.read_head(_APP, _ENV, "tenant-a")
    assert prepared is not None and prepared.revision == 2 and prepared.phase == "prepared"
    bundle.witness.clear_fault()
    recovered = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert recovered.outcome is ContinuityOutcome.RECOVERED
    assert recovered.head is not None and recovered.head.revision == 3
    genesis = decode_block(bundle.witness.read_range(1, 0, _BLOCK) or b"")
    seal = decode_block(bundle.witness.read_range(1, _BLOCK, _BLOCK) or b"")
    assert genesis is not None and genesis["kind"] == "segment_genesis"
    assert seal is not None and seal["kind"] == "seal" and seal["revision"] == 3
    rolled_forward = bundle.primary.read_head(_APP, _ENV, "tenant-a")
    assert rolled_forward is not None and rolled_forward.revision == 3


def test_direct_witness_scope_mismatch_does_not_append() -> None:
    store = InMemoryAppendWitness()
    witness = AzureAppendWitness(store, segment_block_limit=8, recovery_block_bound=16)
    first = witness.append_seal(
        application_id=_APP,
        environment_id=_ENV,
        tenant_id="tenant-a",
        revision=1,
        fencing_token="1",
        transition_id="env-1:1:1",
        binding_digest="digest-1",
        status="active",
        identity_epoch=1,
    )
    assert first.kind.value == "committed"
    length = store.length(0)
    rejected = witness.append_seal(
        application_id=_APP,
        environment_id=_ENV,
        tenant_id="tenant-b",
        revision=2,
        fencing_token="1",
        transition_id="env-1:2:1",
        binding_digest="digest-2",
        status="active",
        identity_epoch=1,
    )
    assert rejected.fault is ProviderFault.SCOPE_MISMATCH
    assert store.length(0) == length


def _sample_transition(*, revision: int, fence: str, watermark: int) -> TransitionRecord:
    binding = _binding(watermark, watermark)
    from intergrax.runtime.governance.local_history_continuity import continuity_binding_digest

    digest = continuity_binding_digest(binding)
    transition_id = f"{_ENV}:{revision}:{fence}"
    return TransitionRecord(
        application_id=_APP,
        environment_id=_ENV,
        tenant_id="tenant-a",
        transition_id=transition_id,
        request_digest=request_digest(
            application_id=_APP,
            environment_id=_ENV,
            tenant_id="tenant-a",
            revision=revision,
            fencing_token=fence,
            transition_id=transition_id,
            binding_digest=digest,
            status="active",
            identity_epoch=1,
            pis_history_id=binding.pis_history_id,
            pis_watermark=binding.pis_committed_watermark,
            gca_history_id=binding.gca_history_id,
            gca_watermark=binding.gca_committed_watermark,
        ),
        revision=revision,
        fencing_token=fence,
        binding_digest=digest,
        status="active",
        identity_epoch=1,
        phase="prepared",
        pis_history_id=binding.pis_history_id,
        pis_watermark=binding.pis_committed_watermark,
        gca_history_id=binding.gca_history_id,
        gca_watermark=binding.gca_committed_watermark,
    )
