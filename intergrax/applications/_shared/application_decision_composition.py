# © Artur Czarnecki. All rights reserved.

"""Canonical Tier-3 Application Decision composition (SCENARIO-1-P0-A).

Composes built-in verification, profile-controlled semantic/trajectory stages, and
explicitly selected Decision platform plugins into one immutable Decision flow gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import DecisionPluginProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.decision_artifact_registry import (
    DecisionArtifactKindRegistry,
    decision_artifact_kind_registry,
)
from intergrax.contracts.decision_record import CandidateDecision, validate_decision_artifact_kind
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.decision_strategy import (
    DecisionStrategyRegistry,
    decision_strategy_registry,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageRegistry,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.semantic_verification import (
    ResolvedSemanticRubric,
    SemanticRubricRef,
    VerifierIndependenceMode,
    resolved_semantic_rubric,
    semantic_rubric_ref,
    semantic_verification_independence_config,
)
from intergrax.contracts.trajectory_verification import TrajectoryAgentIdProvider
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.plugins.discovery import (
    EP_DECISION_ARTIFACT_KINDS,
    EP_DECISION_STRATEGIES,
    EP_DECISION_VERIFICATION_STAGES,
)
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGate,
    DecisionFlowGateCapabilities,
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import build_agent_execution_verification_pipeline
from intergrax.runtime.decision_plugin_composition import (
    DecisionPluginLoadPolicy,
    load_decision_artifact_kind_plugins,
    load_decision_strategy_plugins,
    load_verification_stage_plugins,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_composition import (
    ToolWiringEvalVerificationBridge,
)
from intergrax.runtime.decision_verification_stages.semantic import (
    SEMANTIC_VERIFICATION_STAGE_KIND,
    SemanticVerificationStage,
)
from intergrax.runtime.decision_verification_stages.trajectory import (
    TRAJECTORY_VERIFICATION_STAGE_KIND,
    TrajectoryVerificationStage,
)
from intergrax.contracts.semantic_verification import SemanticJudge
from intergrax.tools.providers.eval.contracts import (
    EvalJudgeInput,
    EvalJudgeOutput,
    EvalTrajectoryInput,
    EvalTrajectoryOutput,
)
from intergrax.runtime.execution.inference_profile import InferenceProfileId

_DEFAULT_AGENT_EXECUTION_RUBRIC_ID = "platform.tier3.agent_execution.summary"
_PLATFORM_RUBRIC_PROVENANCE = "platform.application_decision_composition"


class ApplicationDecisionCompositionError(ValueError):
    """Raised when application Decision composition fails closed."""


@dataclass(frozen=True, slots=True)
class ApplicationDecisionWiringSpec:
    """Explicit application-composition contract for canonical Decision wiring."""

    verify_graph_final: bool = True
    verify_uaep_step: bool = False
    max_revisions: int = 0


@dataclass(frozen=True, slots=True)
class ApplicationDecisionPluginEvidence:
    """Observability snapshot for Decision plugin bootstrap."""

    strategy_report: DomainPluginLoadReport
    verification_stage_report: DomainPluginLoadReport
    artifact_kind_report: DomainPluginLoadReport


@dataclass(frozen=True, slots=True)
class ApplicationDecisionComposition:
    """Immutable composed Decision capabilities for one Tier-3 host."""

    gate: DecisionFlowGate[AgentExecutionResult]
    strategy_registry: DecisionStrategyRegistry
    verification_stage_registry: VerificationStageRegistry[AgentExecutionResult]
    artifact_kind_registry: DecisionArtifactKindRegistry
    verification_pipeline: VerificationPipeline[AgentExecutionResult]
    plugin_evidence: ApplicationDecisionPluginEvidence
    verify_graph_final: bool
    verify_uaep_step: bool
    activated_verification_stage_kinds: tuple[str, ...]
    activated_strategy_kinds: tuple[str, ...]
    activated_artifact_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AgentExecutionSummarySemanticExtractor:
    def extract(self, candidate: CandidateDecision[AgentExecutionResult]) -> str:
        content = candidate.artifact.content
        summary = content.summary.strip()
        if summary:
            return summary
        structured = content.structured_data.get("text")
        if isinstance(structured, str) and structured.strip():
            return structured.strip()
        return ""


@dataclass(frozen=True, slots=True)
class _AgentExecutionTrajectoryAgentIdProvider:
    def resolve(self, candidate: CandidateDecision[AgentExecutionResult]) -> str:
        agent_id = candidate.artifact.content.agent_id.strip()
        if not agent_id:
            raise ValueError("agent execution artifact missing agent_id for trajectory verification")
        return agent_id


@dataclass(frozen=True, slots=True)
class _UnavailableSemanticJudge:
    def is_available(self) -> bool:
        return False

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        raise RuntimeError("semantic judge invoked while unavailable")


@dataclass(frozen=True, slots=True)
class _UnavailableTrajectoryEvaluator:
    def is_available(self) -> bool:
        return False

    def evaluate(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        raise RuntimeError("trajectory evaluator invoked while unavailable")


@dataclass(frozen=True, slots=True)
class _PlatformDefaultSemanticRubricResolver:
    rubric: ResolvedSemanticRubric

    def is_available(self) -> bool:
        return True

    def resolve(self, ref: SemanticRubricRef) -> ResolvedSemanticRubric:
        if ref.rubric_id != self.rubric.ref.rubric_id or ref.version != self.rubric.ref.version:
            raise ValueError("semantic rubric not found for configured reference")
        return self.rubric


def _resolve_discover_entry_points(profile: DecisionPluginProfile) -> bool:
    if profile.discover_entry_points:
        return True
    return discover_plugins_enabled()


def _decision_plugin_load_policy(
    profile: DecisionPluginProfile,
    *,
    execution_mode: ExecutionMode,
) -> DecisionPluginLoadPolicy:
    require_manifest_binding = profile.require_manifest_capability_binding
    if execution_mode is ExecutionMode.STRICT:
        require_manifest_binding = True
    return DecisionPluginLoadPolicy(
        require_manifest_capability_binding=require_manifest_binding,
        requested_strategy_plugins=(
            tuple(profile.strategy_plugins) if profile.strategy_plugins else None
        ),
        requested_verification_stage_plugins=(
            tuple(profile.verification_stage_plugins)
            if profile.verification_stage_plugins
            else None
        ),
        requested_artifact_plugins=(
            tuple(profile.artifact_plugins) if profile.artifact_plugins else None
        ),
    )


def _plugin_bootstrap_errors(
    evidence: ApplicationDecisionPluginEvidence,
) -> tuple[str, ...]:
    errors: list[str] = []
    for report in (
        evidence.strategy_report,
        evidence.verification_stage_report,
        evidence.artifact_kind_report,
    ):
        for item in report.failed:
            errors.append(f"decision plugin load failed: {item.spec.name}: {item.error}")
        for item in report.rejected:
            if item.fail_closed:
                errors.append(
                    "decision plugin admission rejected: "
                    f"{item.spec.name}: {item.reason_code.value}",
                )
    return tuple(errors)


def assert_strict_decision_plugin_composition_acceptable(
    env: ApplicationEnvironmentProfile,
    composition: ApplicationDecisionComposition,
) -> None:
    if env.execution_mode is not ExecutionMode.STRICT:
        return
    errors = _plugin_bootstrap_errors(composition.plugin_evidence)
    if not errors:
        return
    raise ApplicationDecisionCompositionError("; ".join(errors))


def _validate_requested_plugin_kinds_activated(
    profile: DecisionPluginProfile,
    composition: ApplicationDecisionComposition,
) -> None:
    """Fail closed when profile selects plugin kinds that did not activate."""
    activated_verification = set(composition.activated_verification_stage_kinds)
    activated_strategies = set(composition.activated_strategy_kinds)
    activated_artifacts = set(composition.activated_artifact_kinds)

    missing: list[str] = []
    for ref in profile.verification_stage_plugins:
        if ref.plugin_id not in activated_verification:
            missing.append(f"verification stage {ref.plugin_id!r}")
    for ref in profile.strategy_plugins:
        if ref.plugin_id not in activated_strategies:
            missing.append(f"decision strategy {ref.plugin_id!r}")
    for ref in profile.artifact_plugins:
        if ref.plugin_id not in activated_artifacts:
            missing.append(f"artifact kind {ref.plugin_id!r}")
    if not missing:
        return
    raise ApplicationDecisionCompositionError(
        "selected Decision plugin kinds were not activated: " + ", ".join(missing),
    )


def _resolve_semantic_rubric_ref(
    env: ApplicationEnvironmentProfile,
) -> SemanticRubricRef:
    configured = env.decision_profile.verification.semantic_rubric_ref
    rubric_id = configured.strip() if configured else _DEFAULT_AGENT_EXECUTION_RUBRIC_ID
    return semantic_rubric_ref(rubric_id=rubric_id, version=1)


def _default_platform_semantic_rubric(ref: SemanticRubricRef) -> ResolvedSemanticRubric:
    return resolved_semantic_rubric(
        ref=ref,
        criteria=("agent output must be explicit and bounded",),
        min_score=0.5,
        provenance_ref=_PLATFORM_RUBRIC_PROVENANCE,
    )


def _structural_pipeline(
    *,
    contract: AgentContract,
    capability: str | None,
    plan_criteria: tuple[str, ...],
) -> VerificationPipeline[AgentExecutionResult]:
    return build_agent_execution_verification_pipeline(
        contract=contract,
        capability=capability,
        plan_criteria=plan_criteria,
    )


def _merge_plugin_verification_registry(
    base: VerificationStageRegistry[AgentExecutionResult],
    *,
    env: ApplicationEnvironmentProfile,
    discover: bool,
) -> tuple[VerificationStageRegistry[AgentExecutionResult], DomainPluginLoadReport]:
    plugin_profile = env.decision_profile.plugins
    if not discover or not plugin_profile.verification_stage_plugins:
        return base, DomainPluginLoadReport.empty(EP_DECISION_VERIFICATION_STAGES)
    policy = _decision_plugin_load_policy(
        plugin_profile,
        execution_mode=env.execution_mode,
    )
    outcome = load_verification_stage_plugins(
        base,
        policy=policy,
        discover_entry_points=True,
    )
    return outcome.registry, outcome.report


def _compose_strategy_registry(
    env: ApplicationEnvironmentProfile,
    *,
    discover: bool,
) -> tuple[DecisionStrategyRegistry, DomainPluginLoadReport]:
    base = decision_strategy_registry()
    plugin_profile = env.decision_profile.plugins
    if not discover or not plugin_profile.strategy_plugins:
        return base, DomainPluginLoadReport.empty(EP_DECISION_STRATEGIES)
    policy = _decision_plugin_load_policy(
        plugin_profile,
        execution_mode=env.execution_mode,
    )
    outcome = load_decision_strategy_plugins(
        base,
        policy=policy,
        discover_entry_points=True,
    )
    return outcome.registry, outcome.report


def _compose_artifact_kind_registry(
    env: ApplicationEnvironmentProfile,
    *,
    discover: bool,
) -> tuple[DecisionArtifactKindRegistry, DomainPluginLoadReport]:
    base = decision_artifact_kind_registry(
        (validate_decision_artifact_kind("agent.execution.result"),),
    )
    plugin_profile = env.decision_profile.plugins
    if not discover or not plugin_profile.artifact_plugins:
        return base, DomainPluginLoadReport.empty(EP_DECISION_ARTIFACT_KINDS)
    policy = _decision_plugin_load_policy(
        plugin_profile,
        execution_mode=env.execution_mode,
    )
    outcome = load_decision_artifact_kind_plugins(
        base,
        policy=policy,
        discover_entry_points=True,
    )
    return outcome.registry, outcome.report


def _resolve_semantic_judge(
    eval_bridge: ToolWiringEvalVerificationBridge | None,
) -> SemanticJudge:
    if eval_bridge is not None:
        return eval_bridge.semantic_judge
    return _UnavailableSemanticJudge()


def _profile_verification_registrations(
    env: ApplicationEnvironmentProfile,
    *,
    eval_bridge: ToolWiringEvalVerificationBridge | None,
    base_registrations: tuple[VerificationStageRegistration[AgentExecutionResult], ...],
) -> tuple[VerificationStageRegistration[AgentExecutionResult], ...]:
    verification = env.decision_profile.verification
    registrations: list[VerificationStageRegistration[AgentExecutionResult]] = list(
        base_registrations,
    )
    runtime_ready = eval_bridge is not None

    if verification.semantic_enabled:
        rubric_ref = _resolve_semantic_rubric_ref(env)
        resolver = _PlatformDefaultSemanticRubricResolver(
            rubric=_default_platform_semantic_rubric(rubric_ref),
        )
        semantic_stage = SemanticVerificationStage(
            rubric_ref=rubric_ref,
            rubric_resolver=resolver,
            content_provider=_AgentExecutionSummarySemanticExtractor(),
            judge=_resolve_semantic_judge(eval_bridge),
            independence=semantic_verification_independence_config(
                mode=VerifierIndependenceMode.SHARED_PROFILE,
                producer_profile_id=InferenceProfileId("tier3.agent_execution"),
                verifier_profile_id=InferenceProfileId("tier3.agent_execution"),
            ),
        )
        registrations.append(
            VerificationStageRegistration(
                kind=SEMANTIC_VERIFICATION_STAGE_KIND,
                stage=semantic_stage,
                required=runtime_ready,
            ),
        )

    if verification.trajectory_enabled:
        evaluator = (
            eval_bridge.trajectory_evaluator
            if eval_bridge is not None
            else None
        )
        if evaluator is None:
            evaluator = _UnavailableTrajectoryEvaluator()

        trajectory_stage = TrajectoryVerificationStage(
            evaluator=evaluator,
            agent_id_provider=_AgentExecutionTrajectoryAgentIdProvider(),
        )
        registrations.append(
            VerificationStageRegistration(
                kind=TRAJECTORY_VERIFICATION_STAGE_KIND,
                stage=trajectory_stage,
                required=runtime_ready,
            ),
        )

    return tuple(registrations)


def _activated_kind_names(
    registry: VerificationStageRegistry[AgentExecutionResult],
) -> tuple[str, ...]:
    return tuple(sorted(str(registration.kind) for registration in registry.registrations))


def _activated_strategy_names(registry: DecisionStrategyRegistry) -> tuple[str, ...]:
    return tuple(sorted(str(registration.kind) for registration in registry.registrations))


def _activated_artifact_names(registry: DecisionArtifactKindRegistry) -> tuple[str, ...]:
    return tuple(sorted(str(kind) for kind in registry.kinds))


def compose_application_decision(
    *,
    environment: ApplicationEnvironmentProfile,
    contract: AgentContract,
    spec: ApplicationDecisionWiringSpec,
    capability: str | None = None,
    plan_criteria: tuple[str, ...] = (),
    eval_bridge: ToolWiringEvalVerificationBridge | None = None,
) -> ApplicationDecisionComposition:
    """Compose canonical Decision gate and plugin registries for one application host."""
    if type(environment) is not ApplicationEnvironmentProfile:
        raise TypeError("environment must be ApplicationEnvironmentProfile")
    if type(contract) is not AgentContract:
        raise TypeError("contract must be AgentContract")

    discover = _resolve_discover_entry_points(environment.decision_profile.plugins)

    structural = _structural_pipeline(
        contract=contract,
        capability=capability,
        plan_criteria=plan_criteria,
    )
    plugin_stage_registry, verification_report = _merge_plugin_verification_registry(
        structural.registry,
        env=environment,
        discover=discover,
    )

    stage_registrations = _profile_verification_registrations(
        environment,
        eval_bridge=eval_bridge,
        base_registrations=plugin_stage_registry.registrations,
    )
    pipeline = VerificationPipeline(
        registry=verification_stage_registry(stage_registrations),
    )

    strategy_registry, strategy_report = _compose_strategy_registry(
        environment,
        discover=discover,
    )
    artifact_registry, artifact_report = _compose_artifact_kind_registry(
        environment,
        discover=discover,
    )

    evidence = ApplicationDecisionPluginEvidence(
        strategy_report=strategy_report,
        verification_stage_report=verification_report,
        artifact_kind_report=artifact_report,
    )
    composition = ApplicationDecisionComposition(
        gate=_build_gate(spec=spec, pipeline=pipeline),
        strategy_registry=strategy_registry,
        verification_stage_registry=pipeline.registry,
        artifact_kind_registry=artifact_registry,
        verification_pipeline=pipeline,
        plugin_evidence=evidence,
        verify_graph_final=spec.verify_graph_final,
        verify_uaep_step=spec.verify_uaep_step,
        activated_verification_stage_kinds=_activated_kind_names(pipeline.registry),
        activated_strategy_kinds=_activated_strategy_names(strategy_registry),
        activated_artifact_kinds=_activated_artifact_names(artifact_registry),
    )
    _validate_requested_plugin_kinds_activated(
        environment.decision_profile.plugins,
        composition,
    )
    assert_strict_decision_plugin_composition_acceptable(environment, composition)
    return composition


def _build_gate(
    *,
    spec: ApplicationDecisionWiringSpec,
    pipeline: VerificationPipeline[AgentExecutionResult],
) -> DecisionFlowGate[AgentExecutionResult]:
    scopes: set[DecisionFlowScope] = set()
    if spec.verify_graph_final:
        scopes.add(DecisionFlowScope.GRAPH_FINAL)
    if spec.verify_uaep_step:
        scopes.add(DecisionFlowScope.UAEP_STEP)
    return CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=pipeline,
            revision_policy=decision_revision_policy(max_revisions=spec.max_revisions),
            scopes=frozenset(scopes),
        ),
    )


def verification_stage_kinds_present(
    composition: ApplicationDecisionComposition,
    kinds: Sequence[str],
) -> bool:
    """Return whether all given stage kinds are registered on the composed pipeline."""
    present = set(composition.activated_verification_stage_kinds)
    return all(kind in present for kind in kinds)
