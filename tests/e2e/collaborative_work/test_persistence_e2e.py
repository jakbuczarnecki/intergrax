# © Artur Czarnecki. All rights reserved.

"""Persistence, CAS, idempotency, reopen, and provider failure E2E proofs."""

from __future__ import annotations

import threading

import pytest

from intergrax.collaborative_work.persistence import (
    open_postgresql_collaborative_work_repositories,
)
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.collaborative_work.repository import (
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipIdempotencyConflict,
    WorkspaceMembershipRevisionConflict,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.collaborative_work import MembershipStatus, WorkspaceMembershipRole
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from tests.e2e.collaborative_work.harness.composition import (
    MultiplayerE2EContext,
    reopen_multiplayer_e2e_context,
)
from tests.e2e.collaborative_work.harness.constants import FIXED_NOW
from tests.e2e.collaborative_work.harness.fixtures import seed_direct_allow_fixture
from tests.e2e.collaborative_work.harness.runtime_policy import MutableRuntimePolicyEvaluator
from tests.e2e.collaborative_work.harness.scenario_runner import (
    allow_runtime_decision,
    assert_authorization_allowed,
    build_enforcement_request,
    run_multiplayer_e2e_scenario,
)
from tests.e2e.collaborative_work.harness.side_effect_probe import SideEffectProbe
from tests.unit.collaborative_work import test_repository_contracts as contract_suite

pytestmark = pytest.mark.e2e


def _membership_command(**overrides: object):
    return contract_suite._membership_command(**overrides)


def test_sqlite_stale_cas_write_rejected(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    created = sqlite_e2e_context.bundle.membership.create(
        _membership_command(membership_id="cas-membership")
    )
    sqlite_e2e_context.bundle.membership.update(
        UpdateWorkspaceMembershipCommand(
            scope=WorkspaceMembershipScopeKey(
                tenant_id=contract_suite._TENANT_A,
                workspace_id=contract_suite._WORKSPACE_A,
                membership_id="cas-membership",
            ),
            expected_revision=created.revision,
            role=WorkspaceMembershipRole.ADMIN,
            status=MembershipStatus.SUSPENDED,
        )
    )
    with pytest.raises(WorkspaceMembershipRevisionConflict):
        sqlite_e2e_context.bundle.membership.update(
            UpdateWorkspaceMembershipCommand(
                scope=WorkspaceMembershipScopeKey(
                    tenant_id=contract_suite._TENANT_A,
                    workspace_id=contract_suite._WORKSPACE_A,
                    membership_id="cas-membership",
                ),
                expected_revision=created.revision,
                role=WorkspaceMembershipRole.MEMBER,
                status=MembershipStatus.ACTIVE,
            )
        )


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_stale_cas_write_rejected(
    postgresql_e2e_context: MultiplayerE2EContext,
) -> None:
    created = postgresql_e2e_context.bundle.membership.create(
        _membership_command(membership_id="cas-membership-pg")
    )
    postgresql_e2e_context.bundle.membership.update(
        UpdateWorkspaceMembershipCommand(
            scope=WorkspaceMembershipScopeKey(
                tenant_id=contract_suite._TENANT_A,
                workspace_id=contract_suite._WORKSPACE_A,
                membership_id="cas-membership-pg",
            ),
            expected_revision=created.revision,
            role=WorkspaceMembershipRole.ADMIN,
            status=MembershipStatus.SUSPENDED,
        )
    )
    with pytest.raises(WorkspaceMembershipRevisionConflict):
        postgresql_e2e_context.bundle.membership.update(
            UpdateWorkspaceMembershipCommand(
                scope=WorkspaceMembershipScopeKey(
                    tenant_id=contract_suite._TENANT_A,
                    workspace_id=contract_suite._WORKSPACE_A,
                    membership_id="cas-membership-pg",
                ),
                expected_revision=created.revision,
                role=WorkspaceMembershipRole.MEMBER,
                status=MembershipStatus.ACTIVE,
            )
        )


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_concurrent_cas_race_one_winner(
    postgresql_e2e_context: MultiplayerE2EContext,
) -> None:
    created = postgresql_e2e_context.bundle.membership.create(
        _membership_command(membership_id="race-membership")
    )
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        bundle = open_postgresql_collaborative_work_repositories(
            config=postgresql_e2e_context.bundle.store.config,
            schema_name=postgresql_e2e_context.bundle.store.schema_name,
        )
        try:
            barrier.wait(timeout=5)
            bundle.membership.update(
                UpdateWorkspaceMembershipCommand(
                    scope=WorkspaceMembershipScopeKey(
                        tenant_id=contract_suite._TENANT_A,
                        workspace_id=contract_suite._WORKSPACE_A,
                        membership_id="race-membership",
                    ),
                    expected_revision=created.revision,
                    role=WorkspaceMembershipRole.ADMIN,
                    status=MembershipStatus.SUSPENDED,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            bundle.close()

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], WorkspaceMembershipRevisionConflict)


def test_sqlite_idempotency_replay_and_conflict(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    command = _membership_command(
        membership_id="idem-membership",
        idempotency_key="idem-key-e2e",
    )
    first = sqlite_e2e_context.bundle.membership.create(command)
    second = sqlite_e2e_context.bundle.membership.create(command)
    assert second.membership_id == first.membership_id
    conflicting = _membership_command(
        membership_id="idem-membership-other",
        idempotency_key="idem-key-e2e",
        principal_id="other-principal",
    )
    with pytest.raises(WorkspaceMembershipIdempotencyConflict):
        sqlite_e2e_context.bundle.membership.create(conflicting)


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_idempotency_replay_and_conflict(
    postgresql_e2e_context: MultiplayerE2EContext,
) -> None:
    command = _membership_command(
        membership_id="idem-membership-pg",
        idempotency_key="idem-key-pg-e2e",
    )
    first = postgresql_e2e_context.bundle.membership.create(command)
    second = postgresql_e2e_context.bundle.membership.create(command)
    assert second.membership_id == first.membership_id
    conflicting = _membership_command(
        membership_id="idem-membership-pg-other",
        idempotency_key="idem-key-pg-e2e",
        principal_id="other-principal",
    )
    with pytest.raises(WorkspaceMembershipIdempotencyConflict):
        postgresql_e2e_context.bundle.membership.create(conflicting)


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_independent_reopen_preserves_enforcement(
    postgresql_e2e_profile: IntegrationProfile,
    postgresql_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_direct_allow_fixture(postgresql_e2e_context.bundle)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        postgresql_e2e_context,
        build_enforcement_request(),
        probe,
    )
    assert_authorization_allowed(result, probe)
    postgresql_e2e_context.bundle.close()

    reopened = reopen_multiplayer_e2e_context(
        postgresql_e2e_profile,
        MutableRuntimePolicyEvaluator(allow_runtime_decision()),
        clock=lambda: FIXED_NOW,
    )
    try:
        probe_reopened = SideEffectProbe()
        result_reopened = run_multiplayer_e2e_scenario(
            reopened,
            build_enforcement_request(),
            probe_reopened,
        )
        assert_authorization_allowed(result_reopened, probe_reopened)
    finally:
        reopened.bundle.close()


def test_sqlite_independent_reopen_preserves_enforcement(
    sqlite_e2e_profile: IntegrationProfile,
    sqlite_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_direct_allow_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(),
        probe,
    )
    assert probe.count == 1
    sqlite_e2e_context.bundle.close()

    reopened = reopen_multiplayer_e2e_context(
        sqlite_e2e_profile,
        MutableRuntimePolicyEvaluator(allow_runtime_decision()),
        clock=lambda: FIXED_NOW,
    )
    try:
        probe_reopened = SideEffectProbe()
        result = run_multiplayer_e2e_scenario(
            reopened,
            build_enforcement_request(),
            probe_reopened,
        )
        assert_authorization_allowed(result, probe_reopened)
    finally:
        reopened.bundle.close()


def test_provider_failure_has_no_fallback(invalid_postgresql_profile_fixture: IntegrationProfile) -> None:
    with pytest.raises(IntegrationConfigurationError):
        resolve_collaborative_work_repositories(invalid_postgresql_profile_fixture)


def test_sqlite_authorization_denied_has_zero_side_effects(
    sqlite_e2e_context: MultiplayerE2EContext,
) -> None:
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(),
        probe,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert probe.count == 0


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_unique_membership_create_race(
    postgresql_e2e_context: MultiplayerE2EContext,
) -> None:
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt(membership_id: str) -> None:
        bundle = open_postgresql_collaborative_work_repositories(
            config=postgresql_e2e_context.bundle.store.config,
            schema_name=postgresql_e2e_context.bundle.store.schema_name,
        )
        try:
            barrier.wait(timeout=5)
            bundle.membership.create(_membership_command(membership_id=membership_id))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            bundle.close()

    threads = [
        threading.Thread(target=attempt, args=("race-a",)),
        threading.Thread(target=attempt, args=("race-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], WorkspaceMembershipAlreadyExists)
