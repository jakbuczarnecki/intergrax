# © Artur Czarnecki. All rights reserved.

"""MP-7C-C1 — runtime policy evaluator contract injection at host composition."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring import (
    build_harness_host_meaningful_side_effect_authorization_port,
    resolve_harness_host_meaningful_side_effect_authorization_wiring,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from tests.qualification.multiplayer.mp7b.custom_ports import AllowingAuthorizationPort
from tests.qualification.multiplayer.mp7c.host_composition import (
    compose_host_materialized_default_wiring,
    empty_in_memory_repositories,
    resolve_host_wiring,
    strict_host_environment,
    strict_host_governance_evidence_persistence,
)

_STRICT_GEP = strict_host_governance_evidence_persistence()

pytestmark = pytest.mark.unit


class _ExplodingRuntimePolicyEvaluator:
    """Conforming evaluator that must never be consulted when whole-port override wins."""

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        del request
        raise AssertionError("exploding evaluator must not be consulted")


class _RecordingRuntimePolicyEvaluator:
    """Lightweight custom conforming evaluator (pluginability proof)."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        del request
        self.calls += 1
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="mp7c-c1-custom-evaluator",
            policy_rule_id="mp7c.c1.custom",
        )


def test_injected_evaluator_identity_reaches_production_builder() -> None:
    injected = RuntimePolicyEngine()
    bundle = empty_in_memory_repositories()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
        "build_production_orchestration_meaningful_side_effect_authorization_boundary",
    ) as build_mock:
        build_mock.return_value = MagicMock()
        resolve_harness_host_meaningful_side_effect_authorization_wiring(
            strict_host_environment(),
            collaborative_work_repositories=bundle,
            decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
            runtime_policy_evaluator=injected,
            governance_evidence_persistence=_STRICT_GEP,
        )
    build_mock.assert_called_once()
    assert build_mock.call_args.kwargs["runtime_policy_evaluator"] is injected


def test_default_path_constructs_runtime_policy_engine() -> None:
    bundle = empty_in_memory_repositories()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
        "RuntimePolicyEngine",
        wraps=RuntimePolicyEngine,
    ) as engine_ctor:
        with patch(
            "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
            "build_production_orchestration_meaningful_side_effect_authorization_boundary",
        ) as build_mock:
            build_mock.return_value = MagicMock()
            resolve_harness_host_meaningful_side_effect_authorization_wiring(
                strict_host_environment(),
                collaborative_work_repositories=bundle,
                decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
                governance_evidence_persistence=_STRICT_GEP,
            )
    engine_ctor.assert_called_once_with()
    forwarded = build_mock.call_args.kwargs["runtime_policy_evaluator"]
    assert isinstance(forwarded, RuntimePolicyEngine)


def test_injected_evaluator_skips_default_runtime_policy_engine_construction() -> None:
    injected = RuntimePolicyEngine()
    bundle = empty_in_memory_repositories()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
        "RuntimePolicyEngine",
    ) as engine_ctor:
        with patch(
            "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
            "build_production_orchestration_meaningful_side_effect_authorization_boundary",
        ) as build_mock:
            build_mock.return_value = MagicMock()
            resolve_harness_host_meaningful_side_effect_authorization_wiring(
                strict_host_environment(),
                collaborative_work_repositories=bundle,
                decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
                runtime_policy_evaluator=injected,
                governance_evidence_persistence=_STRICT_GEP,
            )
    engine_ctor.assert_not_called()
    assert build_mock.call_args.kwargs["runtime_policy_evaluator"] is injected


def test_explicit_authorization_port_does_not_consult_exploding_evaluator() -> None:
    custom = AllowingAuthorizationPort()
    exploding = _ExplodingRuntimePolicyEvaluator()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
        "resolve_collaborative_work_repositories",
    ) as resolve_mock:
        with patch(
            "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
            "_build_port_from_materialized_repositories",
        ) as build_mock:
            with patch(
                "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
                "RuntimePolicyEngine",
            ) as engine_ctor:
                wiring = resolve_host_wiring(
                    strict_host_environment(),
                    explicit=custom,
                    runtime_policy_evaluator=exploding,
                )
    resolve_mock.assert_not_called()
    build_mock.assert_not_called()
    engine_ctor.assert_not_called()
    assert wiring.authorization_port is custom


def test_injected_repositories_path_forwards_evaluator_identity() -> None:
    injected = _RecordingRuntimePolicyEvaluator()
    bundle = empty_in_memory_repositories()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
        "build_production_orchestration_meaningful_side_effect_authorization_boundary",
    ) as build_mock:
        build_mock.return_value = MagicMock()
        build_harness_host_meaningful_side_effect_authorization_port(
            strict_host_environment(),
            collaborative_work_repositories=bundle,
            decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
            runtime_policy_evaluator=injected,
            governance_evidence_persistence=_STRICT_GEP,
        )
    assert build_mock.call_args.kwargs["runtime_policy_evaluator"] is injected


def test_materialized_repositories_path_forwards_evaluator_identity() -> None:
    injected = _RecordingRuntimePolicyEvaluator()
    store_bundle = empty_in_memory_repositories()

    def _resolve(_profile: IntegrationProfile) -> CollaborativeWorkRepositories:
        return store_bundle

    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
        "build_production_orchestration_meaningful_side_effect_authorization_boundary",
    ) as build_mock:
        build_mock.return_value = MagicMock()
        compose_host_materialized_default_wiring(
            resolve_repositories=_resolve,
            runtime_policy_evaluator=injected,
        )
    assert build_mock.call_args.kwargs["runtime_policy_evaluator"] is injected


def test_custom_conforming_evaluator_is_accepted_without_concrete_branching() -> None:
    custom = _RecordingRuntimePolicyEvaluator()
    bundle = empty_in_memory_repositories()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring."
        "build_production_orchestration_meaningful_side_effect_authorization_boundary",
    ) as build_mock:
        build_mock.return_value = MagicMock()
        resolve_harness_host_meaningful_side_effect_authorization_wiring(
            strict_host_environment(),
            collaborative_work_repositories=bundle,
            decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
            runtime_policy_evaluator=custom,
            governance_evidence_persistence=_STRICT_GEP,
        )
    assert build_mock.call_args.kwargs["runtime_policy_evaluator"] is custom
    assert not isinstance(custom, RuntimePolicyEngine)
