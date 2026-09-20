# © Artur Czarnecki. All rights reserved.

"""MP-7C — host composition boundary, lifecycle, and real ALLOW/DENY proofs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.runtime.governance.orchestration_decision_bound_effect_composition import (
    OrchestrationDecisionBoundCompositionError,
)
from tests.qualification.multiplayer.mp7b.consumer import (
    Tier3MultiplayerAuthorizationDeniedError,
    Tier3MultiplayerConsumer,
)
from tests.qualification.multiplayer.mp7c.host_composition import (
    _ObservableCollaborativeWorkStore,
    bound_host_active_execution,
    compose_host_default_allow_scenario,
    compose_host_default_deny_scenario,
    compose_host_default_fail_closed_no_runtime_rule_scenario,
    compose_host_materialized_default_wiring,
    empty_in_memory_repositories,
    non_strict_host_environment,
    resolve_host_wiring,
    strict_host_environment,
)

pytestmark = pytest.mark.unit


def test_strict_host_default_resolves_real_public_port() -> None:
    store = _ObservableCollaborativeWorkStore()
    bundle = empty_in_memory_repositories(store)
    wiring = resolve_host_wiring(
        strict_host_environment(),
        collaborative_work_repositories=bundle,
    )
    assert wiring.authorization_port is not None
    assert isinstance(wiring.authorization_port, MeaningfulSideEffectAuthorizationPort)
    # Injected repos → caller owns lifecycle (not host-owned).
    assert wiring.owned_collaborative_work_persistence is None
    assert store.closed is False


def test_non_strict_host_returns_none_port_not_fail_open() -> None:
    wiring = resolve_host_wiring(non_strict_host_environment())
    assert wiring.authorization_port is None
    assert wiring.owned_collaborative_work_persistence is None
    # Absence of port means capability not enabled for host mode — not ALLOW.


def test_resolver_materialized_repositories_are_host_owned() -> None:
    store = _ObservableCollaborativeWorkStore()
    bundle = empty_in_memory_repositories(store)
    calls: list[IntegrationProfile] = []

    def _resolve(profile: IntegrationProfile) -> CollaborativeWorkRepositories:
        calls.append(profile)
        return bundle

    wiring = compose_host_materialized_default_wiring(resolve_repositories=_resolve)
    assert wiring.authorization_port is not None
    assert isinstance(wiring.authorization_port, MeaningfulSideEffectAuthorizationPort)
    assert wiring.owned_collaborative_work_persistence is bundle
    assert len(calls) == 1


def test_injected_repositories_are_reused_without_duplicate_materialization() -> None:
    store = _ObservableCollaborativeWorkStore()
    bundle = empty_in_memory_repositories(store)
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring.resolve_collaborative_work_repositories",
    ) as resolve_mock:
        wiring = resolve_host_wiring(
            strict_host_environment(),
            collaborative_work_repositories=bundle,
        )
    resolve_mock.assert_not_called()
    assert wiring.authorization_port is not None
    assert wiring.owned_collaborative_work_persistence is None
    assert store.materialization_token == "mp7c-in-memory-store"


def test_explicit_collaborative_work_profile_beats_environment_profile() -> None:
    store = _ObservableCollaborativeWorkStore()
    bundle = empty_in_memory_repositories(store)
    explicit_profile = IntegrationProfile(
        relational_store=IntegrationBinding(manifest=SQLITE),
        options={SQLITE.slug: {"mp7c": "explicit-profile"}},
    )
    seen: list[IntegrationProfile] = []

    def _resolve(profile: IntegrationProfile) -> CollaborativeWorkRepositories:
        seen.append(profile)
        return bundle

    wiring = compose_host_materialized_default_wiring(
        resolve_repositories=_resolve,
        collaborative_work_integration_profile=explicit_profile,
    )
    assert wiring.authorization_port is not None
    assert len(seen) == 1
    assert seen[0] is explicit_profile


def test_strict_materialization_failure_raises_not_none_port() -> None:
    def _broken(_profile: IntegrationProfile) -> CollaborativeWorkRepositories:
        raise RuntimeError("mp7c-provider-materialization-failed")

    with pytest.raises(RuntimeError, match="mp7c-provider-materialization-failed"):
        compose_host_materialized_default_wiring(resolve_repositories=_broken)


def test_strict_missing_decision_policy_fails_closed() -> None:
    from intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring import (
        resolve_harness_host_meaningful_side_effect_authorization_wiring,
    )

    env = strict_host_environment()
    bundle = empty_in_memory_repositories()
    with pytest.raises(OrchestrationDecisionBoundCompositionError):
        resolve_harness_host_meaningful_side_effect_authorization_wiring(
            env,
            collaborative_work_repositories=bundle,
        )


def test_host_resolved_real_platform_allow_through_consumer() -> None:
    scenario = compose_host_default_allow_scenario()
    consumer = Tier3MultiplayerConsumer(authorization=scenario.authorization_port)
    with bound_host_active_execution(
        task_id=scenario.task_id,
        run_id=scenario.run_id,
        attempt_id=scenario.attempt_id,
        execution_id=scenario.execution_id,
    ):
        result = consumer.request_authorized_side_effect(scenario.request)
    assert result.permitted is True
    assert result.decision.action is PolicyAction.ALLOW


def test_host_resolved_real_platform_deny_through_consumer() -> None:
    scenario = compose_host_default_deny_scenario()
    consumer = Tier3MultiplayerConsumer(authorization=scenario.authorization_port)
    with bound_host_active_execution(
        task_id=scenario.task_id,
        run_id=scenario.run_id,
        attempt_id=scenario.attempt_id,
        execution_id=scenario.execution_id,
    ):
        with pytest.raises(Tier3MultiplayerAuthorizationDeniedError):
            consumer.request_authorized_side_effect(scenario.request)


def test_host_default_empty_evaluator_fail_closed_not_allow() -> None:
    """Allow-ish CW state + default empty RuntimePolicyEngine → not ALLOW."""
    scenario = compose_host_default_fail_closed_no_runtime_rule_scenario()
    consumer = Tier3MultiplayerConsumer(authorization=scenario.authorization_port)
    with bound_host_active_execution(
        task_id=scenario.task_id,
        run_id=scenario.run_id,
        attempt_id=scenario.attempt_id,
        execution_id=scenario.execution_id,
    ):
        with pytest.raises(Tier3MultiplayerAuthorizationDeniedError):
            consumer.request_authorized_side_effect(scenario.request)


def test_caller_supplied_request_fields_are_not_authority() -> None:
    """Embedded membership without authoritative store state still DENY (MP-7B invariant)."""
    scenario = compose_host_default_deny_scenario()
    assert scenario.request.membership is not None
    assert scenario.request.tenant_id == "mp7c-tenant"
    consumer = Tier3MultiplayerConsumer(authorization=scenario.authorization_port)
    with bound_host_active_execution(
        task_id=scenario.task_id,
        run_id=scenario.run_id,
        attempt_id=scenario.attempt_id,
        execution_id=scenario.execution_id,
    ):
        with pytest.raises(Tier3MultiplayerAuthorizationDeniedError):
            consumer.request_authorized_side_effect(scenario.request)
