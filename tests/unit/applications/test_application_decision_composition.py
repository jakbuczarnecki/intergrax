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
from intergrax.core.plugins.selection_ref import PlatformPluginSelectionRef
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
        verification_stage_plugins=[],
    )
    env.execution_mode = ExecutionMode.STRICT
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
        verification_stage_plugins=[
            PlatformPluginSelectionRef(
                plugin_id="plugin.stage_alpha",
                entry_point_group=EP_DECISION_VERIFICATION_STAGES,
                entry_point_name="stage_alpha",
                distribution="decision-composition-test-pkg",
            ),
        ],
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
        verification_stage_plugins=[
            PlatformPluginSelectionRef(
                plugin_id="plugin.stage_alpha",
                entry_point_group=EP_DECISION_VERIFICATION_STAGES,
                entry_point_name="stage_alpha",
                distribution="decision-composition-test-pkg",
            ),
            PlatformPluginSelectionRef(
                plugin_id="plugin.stage_beta",
                entry_point_group=EP_DECISION_VERIFICATION_STAGES,
                entry_point_name="stage_beta",
                distribution="decision-composition-test-pkg",
            ),
        ],
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
            strategy_plugins=[
                PlatformPluginSelectionRef(
                    plugin_id="plugin.external_strategy",
                    entry_point_group=EP_DECISION_STRATEGIES,
                    entry_point_name="external_strategy",
                    distribution="decision-composition-test-pkg",
                ),
            ],
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


def test_no_applications_decision_wiring_compat_shim() -> None:
    from pathlib import Path

    shim = Path(__file__).resolve().parents[3] / "applications/_shared/decision_wiring.py"
    assert not shim.exists()


def test_selected_but_missing_plugin_fails_closed() -> None:
    env = _env_with_plugins(
        discover_entry_points=True,
        verification_stage_plugins=[
            PlatformPluginSelectionRef(
                plugin_id="plugin.missing_stage",
                entry_point_group=EP_DECISION_VERIFICATION_STAGES,
                entry_point_name="missing_stage",
                distribution="decision-composition-test-pkg",
            ),
        ],
    )
    registry = _registry()
    contract = registry.get_contract("echo")
    with pytest.raises(ApplicationDecisionCompositionError, match="not activated"):
        compose_application_decision(
            environment=env,
            contract=contract,
            spec=application_decision_wiring_spec(),
        )


@pytest.mark.asyncio
async def test_semantic_stage_reads_artifact_content() -> None:
    from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
    from intergrax.contracts.decision_identity import (
        DecisionExecutionLineage,
        DecisionIdentity,
        DecisionScope,
        initial_decision_version,
        mint_decision_id,
    )
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )
    from intergrax.contracts.decision_record import (
        candidate_decision,
        validate_decision_artifact_kind,
    )
    from intergrax.contracts.decision_verification import VerificationStageOutcome
    from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput

    @dataclass(frozen=True, slots=True)
    class _Judge:
        seen_text: list[str]

        def is_available(self) -> bool:
            return True

        def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
            self.seen_text.append(params.output_text)
            return EvalJudgeOutput(rubric_id=params.rubric_id, score=1.0, passed=True)

    judge = _Judge(seen_text=[])
    bridge = type("_Bridge", (), {"semantic_judge": judge, "trajectory_evaluator": None})()

    env = _env_with_plugins()
    env.decision_profile.verification.semantic_enabled = True
    registry = _registry()
    contract = registry.get_contract("echo")
    composition = compose_application_decision(
        environment=env,
        contract=contract,
        spec=application_decision_wiring_spec(),
        eval_bridge=bridge,
    )
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="test", subject="subject"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    candidate = candidate_decision(
        identity=identity,
        artifact_kind=validate_decision_artifact_kind("agent.execution.result"),
        payload=AgentExecutionResult(
            agent_id="echo",
            run_id="run-1",
            status=AgentExecutionStatus.COMPLETED,
            summary="typed summary content",
        ),
    )
    result = await composition.verification_pipeline.verify(candidate)
    semantic_records = [
        record
        for record in result.stage_records
        if str(record.stage) == "semantic"
    ]
    assert semantic_records
    assert semantic_records[0].outcome is VerificationStageOutcome.PASSED
    assert judge.seen_text == ["typed summary content"]


@pytest.mark.asyncio
async def test_trajectory_stage_reads_artifact_content_agent_id() -> None:
    from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
    from intergrax.contracts.decision_identity import (
        DecisionExecutionLineage,
        DecisionIdentity,
        DecisionScope,
        initial_decision_version,
        mint_decision_id,
    )
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )
    from intergrax.contracts.decision_record import (
        candidate_decision,
        validate_decision_artifact_kind,
    )
    from intergrax.contracts.decision_verification import VerificationStageOutcome
    from intergrax.tools.providers.eval.contracts import EvalTrajectoryInput, EvalTrajectoryOutput

    @dataclass(frozen=True, slots=True)
    class _Evaluator:
        seen_agent_ids: list[str]

        def is_available(self) -> bool:
            return True

        def evaluate(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
            self.seen_agent_ids.append(params.agent_id)
            return EvalTrajectoryOutput(
                run_id=params.run_id,
                score=1.0,
                passed=True,
            )

    evaluator = _Evaluator(seen_agent_ids=[])
    bridge = type("_Bridge", (), {"semantic_judge": None, "trajectory_evaluator": evaluator})()

    env = _env_with_plugins()
    env.decision_profile.verification.trajectory_enabled = True
    registry = _registry()
    contract = registry.get_contract("echo")
    composition = compose_application_decision(
        environment=env,
        contract=contract,
        spec=application_decision_wiring_spec(),
        eval_bridge=bridge,
    )
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="test", subject="subject"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    candidate = candidate_decision(
        identity=identity,
        artifact_kind=validate_decision_artifact_kind("agent.execution.result"),
        payload=AgentExecutionResult(
            agent_id="trajectory-agent",
            run_id="run-2",
            status=AgentExecutionStatus.COMPLETED,
            summary="trajectory bounded output",
        ),
    )
    result = await composition.verification_pipeline.verify(candidate)
    trajectory_records = [
        record
        for record in result.stage_records
        if str(record.stage) == "trajectory"
    ]
    assert trajectory_records
    assert trajectory_records[0].outcome is VerificationStageOutcome.PASSED
    assert evaluator.seen_agent_ids == ["trajectory-agent"]


def test_strict_mode_requires_manifest_binding_without_profile_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.core.plugins.admission import DomainPluginLoadReport, PluginAdmissionRejection
    from intergrax.core.plugins.discovery import EntryPointSpec
    from intergrax.runtime.decision_plugin_composition import DecisionStrategyPluginLoadOutcome

    env = ApplicationEnvironmentProfile.strict_multi_agent_defaults()
    env.execution_mode = ExecutionMode.STRICT
    env.decision_profile = DecisionProfile(
        plugins=DecisionPluginProfile(
            discover_entry_points=True,
            strategy_plugins=[
                PlatformPluginSelectionRef(
                    plugin_id="plugin.external_strategy",
                    entry_point_group=EP_DECISION_STRATEGIES,
                    entry_point_name="external_strategy",
                    distribution="decision-composition-test-pkg",
                ),
            ],
            require_manifest_capability_binding=False,
        ),
    )
    registry = _registry()
    contract = registry.get_contract("echo")
    captured_policy: list[object] = []

    def _fake_strategy_load(registry_in, *, policy=None, discover_entry_points=False):
        captured_policy.append(policy)
        return DecisionStrategyPluginLoadOutcome(
            registry=registry_in,
            report=DomainPluginLoadReport.empty(EP_DECISION_STRATEGIES),
        )

    with patch(
        "intergrax.applications._shared.application_decision_composition.load_decision_strategy_plugins",
        side_effect=_fake_strategy_load,
    ):
        with pytest.raises(ApplicationDecisionCompositionError, match="not activated"):
            compose_application_decision(
                environment=env,
                contract=contract,
                spec=application_decision_wiring_spec(),
            )
    assert captured_policy
    assert captured_policy[0].require_manifest_capability_binding is True


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
