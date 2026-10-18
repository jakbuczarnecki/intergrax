# © Artur Czarnecki. All rights reserved.

"""D03-A-LOCAL-01 — local ELCR-IIRW continuity, fail-closed and isolated."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_reference_production_control_plane_mutation_boundary,
)
from intergrax.applications._shared.history_continuity_composition import (
    compose_history_continuity_authority,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.governance.history_continuity import (
    ContinuityDenialCode,
    ContinuityHead,
    ContinuityHeadStatus,
    ContinuityOutcome,
    ContinuityPermission,
    ContinuityPermissionGrant,
    ContinuityScope,
    HistoryBinding,
    HistoryContinuityActivation,
    HistoryContinuityAdapterKind,
    HistoryContinuityRetentionPolicy,
)
from intergrax.runtime.governance.local_history_continuity import (
    LocalContinuityCrashPoint,
    LocalDeterministicHistoryContinuity,
)

_REPO = Path(__file__).resolve().parents[3]
_ISSUER = "continuity-owner"
_APP = "app-1"
_ENV = "env-1"


def _governance(
    principal_tenant: str = "tenant-a",
    resolver_tenant: str = "tenant-a",
) -> ProductionControlPlaneGovernance:
    env = ApplicationEnvironmentProfile(profile_id=principal_tenant)
    return ProductionControlPlaneGovernance(
        principal=RequestIdentity(
            tenant_id=principal_tenant,
            user_id=_ISSUER,
            principal_type=PrincipalType.SERVICE,
            auth_subject=_ISSUER,
        ),
        mutation_authorization_boundary=build_reference_production_control_plane_mutation_boundary(
            env,
        ),
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


def _activation() -> HistoryContinuityActivation:
    return HistoryContinuityActivation(
        enabled=True,
        adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
        application_id=_APP,
        environment_id=_ENV,
        retention=_retention(),
    )


def _grant(permission: ContinuityPermission) -> ContinuityPermissionGrant:
    return ContinuityPermissionGrant(principal_id=_ISSUER, permission=permission)


def _scope(
    *,
    tenant_id: str = "tenant-a",
    environment_id: str = _ENV,
    application_id: str = _APP,
) -> ContinuityScope:
    return ContinuityScope(
        application_id=application_id,
        environment_id=environment_id,
        tenant_id=tenant_id,
    )


def _binding(
    pis: int = 1,
    gca: int = 1,
    *,
    pis_id: str = "pis-h",
    gca_id: str = "gca-h",
) -> HistoryBinding:
    return HistoryBinding(
        pis_history_id=pis_id,
        pis_committed_watermark=pis,
        gca_history_id=gca_id,
        gca_committed_watermark=gca,
    )


def _authority() -> LocalDeterministicHistoryContinuity:
    authority = compose_history_continuity_authority(_governance(), _activation())
    assert isinstance(authority, LocalDeterministicHistoryContinuity)
    return authority


def _initialize(authority: LocalDeterministicHistoryContinuity) -> ContinuityHead:
    result = authority.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope(),
        _binding(),
    )
    assert result.outcome is ContinuityOutcome.COMMITTED
    assert result.head is not None
    return result.head


def _advance(
    authority: LocalDeterministicHistoryContinuity,
    head: ContinuityHead,
    *,
    pis: int | None = None,
    gca: int | None = None,
    expected_revision: int | None = None,
    expected_fencing_token: str | None = None,
    permission: ContinuityPermission = ContinuityPermission.WRITE,
    scope: ContinuityScope | None = None,
):
    return authority.advance(
        _grant(permission),
        _scope() if scope is None else scope,
        expected_revision=head.revision if expected_revision is None else expected_revision,
        expected_fencing_token=(
            head.fencing_token if expected_fencing_token is None else expected_fencing_token
        ),
        binding=_binding(
            head.binding.pis_committed_watermark + 1 if pis is None else pis,
            head.binding.gca_committed_watermark if gca is None else gca,
        ),
    )


def test_missing_activation_and_cross_tenant_owner_deny() -> None:
    inactive = compose_history_continuity_authority(_governance(), None)
    assert inactive.sealed_retention() is None
    assert (
        inactive.initialize(_grant(ContinuityPermission.INITIALIZE), _scope(), _binding()).denial
        is ContinuityDenialCode.ACTIVATION_ABSENT
    )
    assert (
        inactive.advance(
            _grant(ContinuityPermission.WRITE),
            _scope(),
            expected_revision=1,
            expected_fencing_token="1",
            binding=_binding(2, 1),
        ).denial
        is ContinuityDenialCode.ACTIVATION_ABSENT
    )
    assert (
        inactive.recover(_grant(ContinuityPermission.RECOVER), _scope()).denial
        is ContinuityDenialCode.ACTIVATION_ABSENT
    )
    mismatched = compose_history_continuity_authority(
        _governance("tenant-b", "tenant-a"),
        _activation(),
    )
    assert not isinstance(mismatched, LocalDeterministicHistoryContinuity)
    assert (
        mismatched.initialize(
            _grant(ContinuityPermission.INITIALIZE),
            _scope(),
            _binding(),
        ).denial
        is ContinuityDenialCode.ACTIVATION_ABSENT
    )


def test_initialize_and_advance_are_monotonic() -> None:
    authority = _authority()
    denied = _advance(authority, _initialize_placeholder_head(), expected_revision=1)
    assert denied.denial is ContinuityDenialCode.NOT_INITIALIZED

    head = _initialize(authority)
    assert head.revision == 1
    assert head.fencing_token == "1"
    assert head.status is ContinuityHeadStatus.ACTIVE
    assert head.identity_epoch == 1
    again = authority.initialize(
        _grant(ContinuityPermission.INITIALIZE),
        _scope(),
        _binding(),
    )
    assert again.denial is ContinuityDenialCode.ALREADY_INITIALIZED

    advanced = _advance(authority, head)
    assert advanced.outcome is ContinuityOutcome.COMMITTED
    assert advanced.head is not None
    assert advanced.head.revision == 2
    assert advanced.head.proof_revision == 2
    assert advanced.witness_high_watermark == 2
    assert advanced.head.binding.pis_committed_watermark == 2

    stalled = _advance(authority, advanced.head, pis=2, gca=1)
    assert stalled.denial is ContinuityDenialCode.CONTINUITY_REVISION_STALE
    regressed = _advance(authority, advanced.head, pis=1, gca=1)
    assert regressed.denial is ContinuityDenialCode.CONTINUITY_REVISION_STALE
    assert authority.read_head(_grant(ContinuityPermission.WRITE), _scope()).head == advanced.head


def _initialize_placeholder_head() -> ContinuityHead:
    return ContinuityHead(
        application_id=_APP,
        environment_id=_ENV,
        tenant_id="tenant-a",
        revision=1,
        fencing_token="1",
        status=ContinuityHeadStatus.ACTIVE,
        binding=_binding(),
        identity_epoch=1,
        proof_revision=1,
    )


def test_stale_revision_fence_scope_and_permissions() -> None:
    authority = _authority()
    for permission in (
        ContinuityPermission.WRITE,
        ContinuityPermission.RECONCILE,
        ContinuityPermission.INVALIDATE,
        ContinuityPermission.RECOVER,
    ):
        denied = authority.initialize(_grant(permission), _scope(), _binding())
        assert denied.denial is ContinuityDenialCode.PERMISSION_DENIED
    foreign = ContinuityPermissionGrant(
        principal_id="other-principal",
        permission=ContinuityPermission.INITIALIZE,
    )
    assert (
        authority.initialize(foreign, _scope(), _binding()).denial
        is ContinuityDenialCode.PERMISSION_DENIED
    )

    head = _initialize(authority)
    assert _advance(authority, head, expected_revision=99).denial is ContinuityDenialCode.REVISION_STALE
    assert (
        _advance(authority, head, expected_fencing_token="9").denial
        is ContinuityDenialCode.FENCE_STALE
    )
    assert (
        _advance(authority, head, scope=_scope(tenant_id="tenant-b")).denial
        is ContinuityDenialCode.TENANT_MISMATCH
    )
    assert (
        _advance(authority, head, scope=_scope(environment_id="env-2")).denial
        is ContinuityDenialCode.ENVIRONMENT_MISMATCH
    )
    assert (
        _advance(authority, head, scope=_scope(application_id="app-2")).denial
        is ContinuityDenialCode.SCOPE_MISMATCH
    )
    for permission in (
        ContinuityPermission.INITIALIZE,
        ContinuityPermission.RECONCILE,
        ContinuityPermission.INVALIDATE,
        ContinuityPermission.RECOVER,
    ):
        assert (
            _advance(authority, head, permission=permission).denial
            is ContinuityDenialCode.PERMISSION_DENIED
        )
    assert authority.read_head(_grant(ContinuityPermission.WRITE), _scope()).head == head


def test_quarantine_reconcile_revocation_and_fence() -> None:
    authority = _authority()
    head = _initialize(authority)
    premature = authority.reconcile(
        _grant(ContinuityPermission.RECONCILE),
        _scope(),
        expected_revision=head.revision,
        expected_fencing_token=head.fencing_token,
        binding=_binding(2, 1),
    )
    assert premature.denial is ContinuityDenialCode.RECONCILIATION_UNAVAILABLE
    assert (
        authority.invalidate(
            _grant(ContinuityPermission.WRITE),
            _scope(),
            expected_revision=head.revision,
            expected_fencing_token=head.fencing_token,
            status=ContinuityHeadStatus.QUARANTINED,
        ).denial
        is ContinuityDenialCode.PERMISSION_DENIED
    )
    quarantined = authority.invalidate(
        _grant(ContinuityPermission.INVALIDATE),
        _scope(),
        expected_revision=head.revision,
        expected_fencing_token=head.fencing_token,
        status=ContinuityHeadStatus.QUARANTINED,
    )
    assert quarantined.outcome is ContinuityOutcome.COMMITTED
    assert quarantined.head is not None
    assert quarantined.head.status is ContinuityHeadStatus.QUARANTINED
    assert quarantined.head.revision == 2
    assert quarantined.head.fencing_token == "2"
    assert (
        _advance(authority, quarantined.head).denial is ContinuityDenialCode.QUARANTINED
    )
    active_reconcile = authority.reconcile(
        _grant(ContinuityPermission.WRITE),
        _scope(),
        expected_revision=2,
        expected_fencing_token="2",
        binding=_binding(2, 1),
    )
    assert active_reconcile.denial is ContinuityDenialCode.PERMISSION_DENIED
    reconciled = authority.reconcile(
        _grant(ContinuityPermission.RECONCILE),
        _scope(),
        expected_revision=2,
        expected_fencing_token="2",
        binding=_binding(2, 1, pis_id="pis-r"),
    )
    assert reconciled.outcome is ContinuityOutcome.COMMITTED
    assert reconciled.head is not None
    assert reconciled.head.status is ContinuityHeadStatus.ACTIVE
    assert reconciled.head.identity_epoch == 2
    assert reconciled.head.fencing_token == "3"
    assert reconciled.head.revision == 3
    assert (
        _advance(authority, reconciled.head, expected_fencing_token="1").denial
        is ContinuityDenialCode.FENCE_STALE
    )

    revoked = authority.invalidate(
        _grant(ContinuityPermission.INVALIDATE),
        _scope(),
        expected_revision=reconciled.head.revision,
        expected_fencing_token=reconciled.head.fencing_token,
        status=ContinuityHeadStatus.REVOKED,
    )
    assert revoked.head is not None
    assert revoked.head.status is ContinuityHeadStatus.REVOKED
    assert _advance(authority, revoked.head).denial is ContinuityDenialCode.REVOKED
    assert (
        authority.reconcile(
            _grant(ContinuityPermission.RECONCILE),
            _scope(),
            expected_revision=revoked.head.revision,
            expected_fencing_token=revoked.head.fencing_token,
            binding=_binding(3, 1, pis_id="pis-x"),
        ).denial
        is ContinuityDenialCode.REVOKED
    )


def test_retention_hard_limits_and_writer_cannot_weaken() -> None:
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
    with pytest.raises(ValidationError):
        HistoryContinuityRetentionPolicy(
            retention_seconds=0,
            ceiling_seconds=10,
            legal_hold=True,
        )
    with pytest.raises(ValidationError):
        HistoryContinuityActivation(
            enabled=False,
            adapter=HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
            application_id=_APP,
            environment_id=_ENV,
            retention=_retention(),
        )
    assert list(HistoryContinuityAdapterKind) == [
        HistoryContinuityAdapterKind.LOCAL_DETERMINISTIC,
    ]

    authority = _authority()
    _initialize(authority)
    sealed = authority.sealed_retention()
    assert sealed == _retention()

    def _change(**kwargs: object):
        payload = {
            "retention_seconds": 3600,
            "ceiling_seconds": 7200,
            "legal_hold": True,
        }
        payload.update(kwargs)
        return authority.attempt_retention_change(
            _grant(ContinuityPermission.WRITE),
            _scope(),
            retention_seconds=int(payload["retention_seconds"]),
            ceiling_seconds=int(payload["ceiling_seconds"]),
            legal_hold=bool(payload["legal_hold"]),
        )

    assert _change(retention_seconds=3599).denial is (
        ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN
    )
    assert _change(ceiling_seconds=7199).denial is (
        ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN
    )
    assert _change(legal_hold=False).denial is (
        ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN
    )
    assert _change(retention_seconds=3601).denial is ContinuityDenialCode.PERMISSION_DENIED
    assert (
        authority.attempt_retention_change(
            _grant(ContinuityPermission.RECOVER),
            _scope(),
            retention_seconds=1,
            ceiling_seconds=7200,
            legal_hold=True,
        ).denial
        is ContinuityDenialCode.RETENTION_WEAKENING_FORBIDDEN
    )
    assert authority.sealed_retention() == sealed


def test_concurrent_cas_has_one_winner() -> None:
    authority = _authority()
    head = _initialize(authority)
    barrier = threading.Barrier(8)
    results: list[object] = []
    results_lock = threading.Lock()

    def _worker() -> None:
        barrier.wait()
        result = _advance(authority, head)
        with results_lock:
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
    assert committed[0].head is not None
    assert committed[0].head.revision == 2
    assert committed[0].witness_high_watermark == 2


def test_crash_before_witness_recovers_without_commit() -> None:
    authority = _authority()
    head = _initialize(authority)
    authority.arm_local_crash(LocalContinuityCrashPoint.AFTER_PREPARE)
    crashed = _advance(authority, head)
    assert crashed.denial is ContinuityDenialCode.TRANSITION_UNRESOLVED
    assert crashed.head is not None
    assert crashed.head.revision == 1
    assert crashed.witness_high_watermark == 1
    assert _advance(authority, head).denial is ContinuityDenialCode.TRANSITION_UNRESOLVED
    recovered = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert recovered.outcome is ContinuityOutcome.RECOVERED
    assert recovered.head is not None
    assert recovered.head.revision == 1
    assert recovered.witness_high_watermark == 1
    resumed = _advance(authority, recovered.head)
    assert resumed.outcome is ContinuityOutcome.COMMITTED
    assert resumed.head is not None
    assert resumed.head.revision == 2


def test_crash_after_witness_recover_commits() -> None:
    authority = _authority()
    head = _initialize(authority)
    authority.arm_local_crash(LocalContinuityCrashPoint.AFTER_WITNESS_SEAL)
    crashed = _advance(authority, head)
    assert crashed.denial is ContinuityDenialCode.TRANSITION_UNRESOLVED
    assert crashed.head is not None
    assert crashed.head.revision == 1
    assert crashed.witness_high_watermark == 2
    recovered = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert recovered.outcome is ContinuityOutcome.RECOVERED
    assert recovered.head is not None
    assert recovered.head.revision == 2
    assert recovered.witness_high_watermark == 2
    assert recovered.head.binding.pis_committed_watermark == 2


def test_primary_rollback_quarantines_without_silent_allow() -> None:
    authority = _authority()
    head = _initialize(authority)
    advanced = _advance(authority, head)
    assert advanced.outcome is ContinuityOutcome.COMMITTED
    authority.testing_rollback_primary()
    observed = authority.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert observed.denial is ContinuityDenialCode.CONTINUITY_REVISION_STALE
    assert observed.head is not None
    assert observed.head.revision == 1
    assert observed.head.status is ContinuityHeadStatus.ACTIVE
    assert observed.witness_high_watermark == 2
    recovered = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert recovered.denial is ContinuityDenialCode.CONTINUITY_REVISION_STALE
    assert recovered.head is not None
    assert recovered.head.revision == 1
    assert recovered.head.status is ContinuityHeadStatus.QUARANTINED
    assert recovered.witness_high_watermark == 2
    assert (
        _advance(authority, recovered.head).denial is ContinuityDenialCode.QUARANTINED
    )


def test_witness_rollback_denies_mismatch() -> None:
    authority = _authority()
    head = _initialize(authority)
    advanced = _advance(authority, head)
    assert advanced.head is not None
    authority.testing_rollback_witness()
    observed = authority.read_head(_grant(ContinuityPermission.WRITE), _scope())
    assert observed.denial is ContinuityDenialCode.WITNESS_MISMATCH
    assert observed.head is not None
    assert observed.head.revision == 2
    assert observed.head.status is ContinuityHeadStatus.ACTIVE
    assert observed.witness_high_watermark == 1
    recovered = authority.recover(_grant(ContinuityPermission.RECOVER), _scope())
    assert recovered.denial is ContinuityDenialCode.WITNESS_MISMATCH
    assert recovered.head is not None
    assert recovered.head.revision == 2
    assert recovered.head.status is ContinuityHeadStatus.QUARANTINED
    assert recovered.witness_high_watermark == 1


def test_direct_construction_and_production_paths_stay_closed() -> None:
    with pytest.raises(RuntimeError, match="compose_history_continuity_authority"):
        LocalDeterministicHistoryContinuity(
            tenant_resolver=StaticApplicationEnvironmentTenantResolver(tenant_id="tenant-a"),
            activation=_activation(),
            issuer_principal_id=_ISSUER,
            composition_token=object(),
        )

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

    local_sources = (
        _REPO / "intergrax" / "contracts" / "governance" / "history_continuity.py",
        _REPO / "intergrax" / "runtime" / "governance" / "local_history_continuity.py",
        _REPO / "intergrax" / "applications" / "_shared" / "history_continuity_composition.py",
    )
    for path in local_sources:
        lowered = path.read_text(encoding="utf-8").lower()
        assert "spanner" not in lowered
        assert "azure" not in lowered
        assert "google.cloud" not in lowered

    opener = "_open_" + "local_history_continuity"
    composer = "compose_history_continuity_authority"
    opener_hits: list[Path] = []
    composer_hits: list[Path] = []
    roots = (
        "intergrax/applications",
        "intergrax/runtime",
        "applications/governed_contractor_application",
        "agents",
    )
    for root in roots:
        base = _REPO / root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if opener in text:
                opener_hits.append(path)
            if composer in text and path.name != Path(__file__).name:
                composer_hits.append(path)
    local_module = _REPO / "intergrax" / "runtime" / "governance" / "local_history_continuity.py"
    composition_module = (
        _REPO / "intergrax" / "applications" / "_shared" / "history_continuity_composition.py"
    )
    assert _same_paths(opener_hits, (local_module, composition_module))
    assert _same_paths(composer_hits, (local_module, composition_module))


def _same_paths(found: list[Path], expected: tuple[Path, ...]) -> bool:
    return sorted(path.resolve() for path in found) == sorted(path.resolve() for path in expected)
