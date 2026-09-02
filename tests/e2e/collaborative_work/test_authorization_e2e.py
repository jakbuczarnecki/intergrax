# © Artur Czarnecki. All rights reserved.

"""Authorization and policy composition E2E — public boundary behavior."""

from __future__ import annotations

import pytest

from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from tests.e2e.collaborative_work.harness.composition import MultiplayerE2EContext
from tests.e2e.collaborative_work.harness.constants import (
    PRINCIPAL_ALICE,
    PRINCIPAL_DELEGATE,
    PRINCIPAL_DELEGATOR,
    RESOURCE_B,
    TENANT_B,
    WORKSPACE_B,
)
from tests.e2e.collaborative_work.harness.fixtures import (
    seed_delegation_amplification_fixture,
    seed_direct_allow_fixture,
    seed_inactive_membership_fixture,
    seed_missing_authority_fixture,
    seed_missing_policy_layer_fixture,
    seed_policy_composition_deny_fixture,
    seed_resource_mismatch_fixture,
    seed_tenant_isolation_fixture,
    seed_valid_delegation_fixture,
)
from tests.e2e.collaborative_work.harness.scenario_runner import (
    assert_authorization_allowed,
    assert_authorization_denied,
    build_enforcement_request,
    run_multiplayer_e2e_scenario,
)
from tests.e2e.collaborative_work.harness.side_effect_probe import SideEffectProbe

pytestmark = pytest.mark.e2e


def _audit_has_authoritative_context(result: MeaningfulSideEffectAuthorizationResult) -> None:
    payload = result.decision.audit_payload or {}
    layers = payload.get("contributing_layers")
    if layers is None:
        assert result.enforcement_result.authority_scope is not None
    assert result.enforcement_result.operation_id


def test_sqlite_direct_authority_allow(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    seed_direct_allow_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    request = build_enforcement_request()
    result = run_multiplayer_e2e_scenario(sqlite_e2e_context, request, probe)
    assert_authorization_allowed(result, probe)
    assert isinstance(result, str)


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_direct_authority_allow(postgresql_e2e_context: MultiplayerE2EContext) -> None:
    seed_direct_allow_fixture(postgresql_e2e_context.bundle)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(postgresql_e2e_context, build_enforcement_request(), probe)
    assert_authorization_allowed(result, probe)


def test_sqlite_missing_authority_deny(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    seed_missing_authority_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(),
        probe,
    )
    assert_authorization_denied(result, probe)


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_missing_authority_deny(postgresql_e2e_context: MultiplayerE2EContext) -> None:
    seed_missing_authority_fixture(postgresql_e2e_context.bundle)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        postgresql_e2e_context,
        build_enforcement_request(),
        probe,
    )
    assert_authorization_denied(result, probe)


def test_sqlite_inactive_membership_deny(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    seed_inactive_membership_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(),
        probe,
    )
    assert_authorization_denied(result, probe)


def test_sqlite_valid_delegation_allow(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    delegation = seed_valid_delegation_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    request = build_enforcement_request(
        acting_principal_id=PRINCIPAL_DELEGATE,
        delegator_principal_id=PRINCIPAL_DELEGATOR,
    )
    request = request.model_copy(
        update={
            "delegation": delegation,
        }
    )
    result = run_multiplayer_e2e_scenario(sqlite_e2e_context, request, probe)
    assert_authorization_allowed(result, probe)


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_valid_delegation_allow(postgresql_e2e_context: MultiplayerE2EContext) -> None:
    delegation = seed_valid_delegation_fixture(postgresql_e2e_context.bundle)
    probe = SideEffectProbe()
    request = build_enforcement_request(
        acting_principal_id=PRINCIPAL_DELEGATE,
        delegator_principal_id=PRINCIPAL_DELEGATOR,
    ).model_copy(update={"delegation": delegation})
    result = run_multiplayer_e2e_scenario(postgresql_e2e_context, request, probe)
    assert_authorization_allowed(result, probe)


def test_sqlite_delegation_amplification_deny(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    delegation = seed_delegation_amplification_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    request = build_enforcement_request(
        acting_principal_id=PRINCIPAL_DELEGATE,
        delegator_principal_id=PRINCIPAL_DELEGATOR,
    ).model_copy(update={"delegation": delegation})
    result = run_multiplayer_e2e_scenario(sqlite_e2e_context, request, probe)
    assert_authorization_denied(result, probe)


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_delegation_amplification_deny(
    postgresql_e2e_context: MultiplayerE2EContext,
) -> None:
    delegation = seed_delegation_amplification_fixture(postgresql_e2e_context.bundle)
    probe = SideEffectProbe()
    request = build_enforcement_request(
        acting_principal_id=PRINCIPAL_DELEGATE,
        delegator_principal_id=PRINCIPAL_DELEGATOR,
    ).model_copy(update={"delegation": delegation})
    result = run_multiplayer_e2e_scenario(postgresql_e2e_context, request, probe)
    assert_authorization_denied(result, probe)


def test_sqlite_resource_scope_mismatch_deny(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    seed_resource_mismatch_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(resource_scope=RESOURCE_B),
        probe,
    )
    assert_authorization_denied(result, probe)


def test_sqlite_tenant_workspace_isolation_deny(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    seed_tenant_isolation_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(
            tenant_id=TENANT_B,
            workspace_id=WORKSPACE_B,
            acting_principal_id=PRINCIPAL_ALICE,
        ),
        probe,
    )
    assert_authorization_denied(result, probe)


@pytest.mark.integration
@pytest.mark.network
def test_postgresql_tenant_workspace_isolation_deny(
    postgresql_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_tenant_isolation_fixture(postgresql_e2e_context.bundle)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        postgresql_e2e_context,
        build_enforcement_request(
            tenant_id=TENANT_B,
            workspace_id=WORKSPACE_B,
            acting_principal_id=PRINCIPAL_ALICE,
        ),
        probe,
    )
    assert_authorization_denied(result, probe)


def test_sqlite_policy_composition_resource_deny(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    seed_policy_composition_deny_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    captured: list[MeaningfulSideEffectAuthorizationResult] = []

    def record(result: MeaningfulSideEffectAuthorizationResult) -> None:
        captured.append(result)

    result = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(),
        probe,
        on_authorization=record,
    )
    assert_authorization_denied(result, probe)
    assert captured
    assert captured[0].decision.action is PolicyAction.DENY
    determining = captured[0].enforcement_result.composition.determining_layer
    assert determining is not None


def test_sqlite_missing_required_policy_layer_deny(
    sqlite_e2e_context: MultiplayerE2EContext,
) -> None:
    seed_missing_policy_layer_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    result = run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(),
        probe,
    )
    assert_authorization_denied(result, probe)


def test_sqlite_allow_path_has_audit_evidence(sqlite_e2e_context: MultiplayerE2EContext) -> None:
    seed_direct_allow_fixture(sqlite_e2e_context.bundle)
    probe = SideEffectProbe()
    captured: list[MeaningfulSideEffectAuthorizationResult] = []

    run_multiplayer_e2e_scenario(
        sqlite_e2e_context,
        build_enforcement_request(),
        probe,
        on_authorization=lambda result: captured.append(result),
    )
    assert probe.count == 1
    assert captured
    _audit_has_authoritative_context(captured[0])
