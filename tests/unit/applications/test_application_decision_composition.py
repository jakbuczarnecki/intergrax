# © Artur Czarnecki. All rights reserved.

"""P0-A architecture tests for application Decision plugin composition."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.application_decision_composition import (
    ApplicationDecisionCompositionError,
    compose_application_decision,
    verification_stage_kinds_present,
)
from intergrax.applications._shared.decision_wiring import application_decision_wiring_spec
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DecisionPluginProfile,
    DecisionProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.decision_strategy import (
    DecisionStrategyKind,
    DecisionStrategyRegistration,
    validate_decision_strategy_kind,
)
from intergrax.contracts.decision_verification import (
    VerificationStageOutcome,
    validate_verification_stage_kind,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
)
from intergrax.contracts.decision_record import CandidateDecision, candidate_decision_ref
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_DECISION_STRATEGIES,
    EP_DECISION_VERIFICATION_STAGES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _StageA:
    kind: str = "plugin.stage_alpha"

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        return VerificationStageExecutionClass.DETERMINISTIC

    async def verify(self, candidate: CandidateDecision[object]) -> object:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class _StageB:
    kind: str = "plugin.stage_beta"

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        return VerificationStageExecutionClass.DETERMINISTIC

    async def verify(self, candidate: CandidateDecision[object]) -> object:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class _ExternalStrategy:
    kind: DecisionStrategyKind = validate_decision_strategy_kind("plugin.external_strategy")


def _registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _env_with_plugins(**plugin_kwargs: object) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="decision.composition.plugins")
    env.decision_profile = DecisionProfile(
        plugins=DecisionPluginProfile(**plugin_kwargs),
    )
    return env


def test_installed_but_not_selected_keeps_plugin_stages_inactive() -> None:
    env = _env_with_plugins(
        discover_entry_points=True,
        verification_stage_kinds=[],
    )
    registry = _registry()
    contract = registry.get_contract("echo")
    composition = compose_application_decision(
        environment=env,
        contract=contract,
        spec=application_decision_wiring_spec(),
    )
    assert verification_stage_kinds_present(composition, ("structural",))
    assert not verification_stage_kinds_present(composition, ("plugin.stage_alpha",))


def test_selected_plugin_stage_merges_into_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_entry_point_spec_cache_for_tests()
    env = _env_with_plugins(
        discover_entry_points=True,
        verification_stage_kinds=["plugin.stage_alpha"],
    )
    registry = _registry()
    contract = registry.get_contract("echo")

    registration = VerificationStageRegistration(
        kind=validate_verification_stage_kind("plugin.stage_alpha"),
        stage=_StageA(),
        required=True,
    )

    def _fake_load(registry_in, *, policy=None, discover_entry_points=False):
        from intergrax.contracts.decision_verification_stage import register_verification_stage
        from intergrax.runtime.decision_plugin_composition import VerificationStagePluginLoadOutcome
        from intergrax.core.plugins.admission import DomainPluginLoadReport

        updated = register_verification_stage(registry_in, registration)
        return VerificationStagePluginLoadOutcome(
            registry=updated,
            report=DomainPluginLoadReport.empty(EP_DECISION_VERIFICATION_STAGES),
        )

    with patch(
        "intergrax.applications._shared.application_decision_composition.load_verification_stage_plugins",
        side_effect=_fake_load,
    ):
        composition = compose_application_decision(
            environment=env,
            contract=contract,
            spec=application_decision_wiring_spec(),
        )
    assert verification_stage_kinds_present(
        composition,
        ("structural", "plugin.stage_alpha"),
    )


def test_deterministic_ordering_independent_of_discovery_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_entry_point_spec_cache_for_tests()
    env = _env_with_plugins(
        discover_entry_points=True,
        verification_stage_kinds=["plugin.stage_alpha", "plugin.stage_beta"],
    )
    registry = _registry()
    contract = registry.get_contract("echo")

    def _fake_load(registry_in, *, policy=None, discover_entry_points=False):
        from intergrax.contracts.decision_verification_stage import register_verification_stage
        from intergrax.runtime.decision_plugin_composition import VerificationStagePluginLoadOutcome
        from intergrax.core.plugins.admission import DomainPluginLoadReport

        updated = register_verification_stage(
            registry_in,
            VerificationStageRegistration(
                kind=validate_verification_stage_kind("plugin.stage_beta"),
                stage=_StageB(),
                required=True,
            ),
        )
        updated = register_verification_stage(
            updated,
            VerificationStageRegistration(
                kind=validate_verification_stage_kind("plugin.stage_alpha"),
                stage=_StageA(),
                required=True,
            ),
        )
        return VerificationStagePluginLoadOutcome(
            registry=updated,
            report=DomainPluginLoadReport.empty(EP_DECISION_VERIFICATION_STAGES),
        )

    with patch(
        "intergrax.applications._shared.application_decision_composition.load_verification_stage_plugins",
        side_effect=_fake_load,
    ):
        composition = compose_application_decision(
            environment=env,
            contract=contract,
            spec=application_decision_wiring_spec(),
        )
    kinds = composition.activated_verification_stage_kinds
    assert kinds.index("plugin.stage_alpha") < kinds.index("plugin.stage_beta")


def test_strict_mode_fail_closed_on_plugin_rejection() -> None:
    from intergrax.core.plugins.admission import DomainPluginLoadReport, PluginAdmissionRejection
    from intergrax.core.plugins.discovery import EntryPointSpec
    from intergrax.runtime.decision_plugin_composition import DecisionStrategyPluginLoadOutcome

    env = ApplicationEnvironmentProfile.strict_multi_agent_defaults()
    env.execution_mode = ExecutionMode.STRICT
    env.decision_profile = DecisionProfile(
        plugins=DecisionPluginProfile(
            discover_entry_points=True,
            strategy_kinds=["plugin.external_strategy"],
        ),
    )
    registry = _registry()
    contract = registry.get_contract("echo")
    rejection = PluginAdmissionRejection(
        spec=EntryPointSpec(
            name="bad",
            group=EP_DECISION_STRATEGIES,
            value="missing:target",
            distribution=None,
        ),
        reason_code=PluginAdmissionReasonCode.MANIFEST_CAPABILITY_BINDING_MISSING,
        reason="binding missing",
        fail_closed=True,
    )
    report = DomainPluginLoadReport(
        group=EP_DECISION_STRATEGIES,
        accepted=(),
        rejected=(rejection,),
        failed=(),
        registered_count=0,
    )

    def _fake_strategy_load(registry_in, *, policy=None, discover_entry_points=False):
        return DecisionStrategyPluginLoadOutcome(registry=registry_in, report=report)

    with patch(
        "intergrax.applications._shared.application_decision_composition.load_decision_strategy_plugins",
        side_effect=_fake_strategy_load,
    ):
        with pytest.raises(ApplicationDecisionCompositionError):
            compose_application_decision(
                environment=env,
                contract=contract,
                spec=application_decision_wiring_spec(),
            )


def test_max_revision_profile_reaches_gate() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="decision.composition.revision")
    env.decision_profile = DecisionProfile.model_validate(
        {
            "flow": {"max_revisions": 4},
        },
    )
    registry = _registry()
    contract = registry.get_contract("echo")
    composition = compose_application_decision(
        environment=env,
        contract=contract,
        spec=application_decision_wiring_spec(max_revisions=4),
    )
    assert composition.gate.capabilities.revision_policy.max_revisions == 4
