# © Artur Czarnecki. All rights reserved.
"""T-IGX-051-A Part E: install CAS — occupied slot refusal and per-slot locking."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from intergrax.agent_distribution.admin_models import (
    BindAgentRequest,
    InstallAgentRequest,
    UpgradeAgentRequest,
)
from intergrax.agent_distribution.errors import InstallationSlotConflict
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.core.qualification import QualificationStatus
from testing_support.agent_platform_admin_harness import admin_test_principal
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _APP,
    _DIGEST,
    _ENV,
    _META_REF,
    _PACKAGE_ID,
    _trust,
    build_admin_stack,
)

pytestmark = pytest.mark.unit

_SLOT = "slot-search"
_DIGEST_B = "sha256:" + ("b" * 64)


def _install_request(
    *,
    installation_id: str,
    mutation_id: str,
    digest: str = _DIGEST,
) -> InstallAgentRequest:
    package = AgentPackageIdentity(
        distribution_package_id=_PACKAGE_ID,
        package_version="1.0.0",
        package_digest=digest,
    )
    trust = AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=digest,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:service:0",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )
    return InstallAgentRequest(
        mutation_id=mutation_id,
        installation_id=installation_id,
        installation_slot_id=_SLOT,
        package_identity=package,
        artifact_store_ref=f"store://artifacts/{installation_id}",
        trust_record=trust,
        agent_project_metadata_ref=_META_REF,
    )


def test_ten_05a_install_over_occupied_slot_refused() -> None:
    stack = build_admin_stack()
    principal = admin_test_principal()
    first = _install_request(installation_id="inst-1", mutation_id="mut-install-1")
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=first,
        principal=principal,
    )
    active_before = stack.service._installation_service.resolve_active_for_slot(_ENV, _SLOT)
    assert active_before is not None
    assert active_before.installation_id == "inst-1"

    second = _install_request(
        installation_id="inst-2",
        mutation_id="mut-install-2",
        digest=_DIGEST_B,
    )
    with pytest.raises(InstallationSlotConflict, match="slot occupied"):
        stack.service.install_agent(
            application_id=_APP,
            application_environment_id=_ENV,
            request=second,
            principal=principal,
        )
    active_after = stack.service._installation_service.resolve_active_for_slot(_ENV, _SLOT)
    assert active_after is not None
    assert active_after.installation_id == "inst-1"


def test_ten_05b_idempotent_replay_unchanged() -> None:
    stack = build_admin_stack()
    principal = admin_test_principal()
    request = _install_request(installation_id="inst-1", mutation_id="mut-install-1")
    first = stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=principal,
    )
    second = stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=request,
        principal=principal,
    )
    assert second.installation.installation_id == first.installation.installation_id
    assert second.audit_event_types == ()


def test_ten_05c_two_concurrent_installs_one_wins() -> None:
    stack = build_admin_stack()
    principal = admin_test_principal()
    barrier = threading.Barrier(2)
    results: list[object] = []
    errors: list[BaseException] = []

    def _attempt(installation_id: str, mutation_id: str, digest: str) -> None:
        barrier.wait()
        try:
            outcome = stack.service.install_agent(
                application_id=_APP,
                application_environment_id=_ENV,
                request=_install_request(
                    installation_id=installation_id,
                    mutation_id=mutation_id,
                    digest=digest,
                ),
                principal=principal,
            )
            results.append(outcome)
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(_attempt, "inst-a", "mut-concurrent-a", _DIGEST)
        pool.submit(_attempt, "inst-b", "mut-concurrent-b", _DIGEST_B)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], InstallationSlotConflict)
    active = stack.service._installation_service.resolve_active_for_slot(_ENV, _SLOT)
    assert active is not None
    assert active.installation_state is InstallationState.INSTALLED_ACTIVE
    installed = stack.service._installation_store.list_installations_for_environment(_ENV)
    active_rows = [
        row
        for row in installed
        if row.installation_slot_id == _SLOT and row.installation_state is InstallationState.INSTALLED_ACTIVE
    ]
    assert len(active_rows) == 1


def test_ten_05d_upgrade_with_expected_token_unchanged() -> None:
    stack = build_admin_stack()
    principal = admin_test_principal()
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(installation_id="inst-1", mutation_id="mut-install-1"),
        principal=principal,
    )
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=BindAgentRequest(
            mutation_id="mut-bind-1",
            application_binding_id="bind-search",
            logical_agent_id="researcher",
            installation_slot_id=_SLOT,
        ),
        principal=principal,
    )
    upgrade = UpgradeAgentRequest(
        mutation_id="mut-upgrade-1",
        installation_id="inst-2",
        installation_slot_id=_SLOT,
        package_identity=AgentPackageIdentity(
            distribution_package_id=_PACKAGE_ID,
            package_version="1.1.0",
            package_digest=_DIGEST_B,
        ),
        artifact_store_ref="store://artifacts/inst-2",
        trust_record=AgentInstallationTrustRecord(
            qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
            package_digest=_DIGEST_B,
            publisher_identity_ref="publisher:acme",
            source_provider_id="builtin",
            trust_evidence_refs=_trust().trust_evidence_refs,
        ),
        agent_project_metadata_ref=_META_REF,
        expected_active_installation_id="inst-1",
    )
    result = stack.service.upgrade_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=upgrade,
        principal=principal,
    )
    assert result.installation.installation_id == "inst-2"
    assert result.installation.installation_state is InstallationState.INSTALLED_ACTIVE
