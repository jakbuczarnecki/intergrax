# © Artur Czarnecki. All rights reserved.

"""SCENARIO-1-P0-B — single platform Decision System authority gates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionScope,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_record import (
    CandidateDecision,
    candidate_decision_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import DecisionResolution
from intergrax.contracts.decision_revision import (
    DecisionRevisionDisposition,
    decision_revision_policy,
)
from intergrax.contracts.decision_verification import (
    VerificationStageOutcome,
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGateCapabilities,
    DecisionFlowHostAction,
    DecisionFlowIdentitySeed,
    DecisionFlowRequest,
    DecisionFlowScope,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    build_scenario_environment_profile,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    derive_terminal_outcome,
    platform_decision_accepted,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCENARIO_APP = (
    _REPO_ROOT / "platform_proofs" / "scenarios" / "ai_incident_investigation" / "application"
)

_FORBIDDEN_SCENARIO_AUTHORITY_PATTERNS = (
    re.compile(r"EvaluatorLoopGraphBinding"),
    re.compile(r"EvaluatorLoopSpec"),
    re.compile(r"rewire_scenario_decision_wiring\([^)]*validation_engine"),
    re.compile(r"IncidentInvestigationValidationEngine\(\)"),
    re.compile(r"reconcile_investigation_completion\("),
    re.compile(r"enforce_pre_reconciliation_validation_clean_transition\("),
)


def _scenario_application_sources() -> list[tuple[Path, str]]:
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(_SCENARIO_APP.glob("*.py"))
    ]


def test_scenario_graph_spec_has_no_evaluator_loop_binding() -> None:
    env = build_scenario_environment_profile()
    spec = env.graph_spec
    assert isinstance(spec, ApplicationGraphSpec)
    assert spec.evaluator_loop is None


def test_scenario_decision_path_has_no_legacy_decision_authority_patterns() -> None:
    watched = (
        _SCENARIO_APP / "scenario.py",
        _SCENARIO_APP / "runtime_composition.py",
    )
    violations: list[str] = []
    for path in watched:
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for pattern in _FORBIDDEN_SCENARIO_AUTHORITY_PATTERNS:
            if pattern.search(source):
                violations.append(f"{rel} matches forbidden pattern {pattern.pattern}")
    assert violations == []


def test_derive_terminal_outcome_requires_platform_acceptance() -> None:
    with pytest.raises(RuntimeError, match="incident_terminal_state_not_accepted"):
        derive_terminal_outcome(
            decision_accepted=False,
            has_supported_diagnosis=True,
            completion_mode="supported_diagnosis",
        )


def test_business_unresolved_is_valid_after_platform_acceptance() -> None:
    outcome = derive_terminal_outcome(
        decision_accepted=True,
        has_supported_diagnosis=False,
        completion_mode="unresolved",
    )
    assert outcome == "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class _BusinessUnresolvedPayload:
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class _PassedStage:
    kind: str = "scenario.business_unresolved.fixture"
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(self, candidate: CandidateDecision[object]) -> object:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class _ChallengedStage:
    kind: str = "scenario.revision.fixture"
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(self, candidate: CandidateDecision[object]) -> object:
        proposal_ref = candidate_decision_ref(candidate)
        finding = verification_finding(
            code=validate_verification_finding_code("scenario.revision.challenged"),
            message="revision required",
        )
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=verification_challenge(
                proposal_ref=proposal_ref,
                stage=validate_verification_stage_kind(self.kind),
                requirement_code=validate_verification_requirement_code(
                    "scenario.revision.requirement",
                ),
                finding=finding,
            ),
        )


def _pipeline(stage: object) -> VerificationPipeline[object]:
    registry = verification_stage_registry(
        (
            VerificationStageRegistration(
                kind=validate_verification_stage_kind(stage.kind),
                stage=stage,
                required=True,
            ),
        ),
    )
    return VerificationPipeline(registry=registry)


def _identity_seed() -> DecisionFlowIdentitySeed:
    return DecisionFlowIdentitySeed(
        scope=DecisionScope(namespace="scenario.p0b", subject="revision"),
        tenant_id="scenario-tenant",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
        decision_id=mint_decision_id(),
    )


@pytest.fixture
def lifecycle_binding():
    token = bind_active_decision_lifecycle_host(CanonicalDecisionLifecycleHost())
    yield
    reset_active_decision_lifecycle_host(token)


@pytest.mark.asyncio
async def test_platform_revision_lifecycle_owned_by_decision_system(lifecycle_binding) -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(_ChallengedStage()),
            revision_policy=decision_revision_policy(max_revisions=1),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
        ),
    )
    artifact_kind = validate_decision_artifact_kind("scenario.revision.payload")
    payload_v1 = _BusinessUnresolvedPayload(status="UNRESOLVED", reason="insufficient evidence")
    first = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=artifact_kind,
            payload=payload_v1,
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    assert first.host_action is DecisionFlowHostAction.BLOCK
    assert first.revision_decision is not None
    assert first.revision_decision.disposition is DecisionRevisionDisposition.ALLOWED
    assert first.candidate is not None
    assert first.candidate.identity.version.value == 1
    assert first.resolution_record is None
    assert first.lifecycle_state.stage is DecisionLifecycleStage.REVISION


@pytest.mark.asyncio
async def test_verification_pass_yields_authoritative_acceptance(lifecycle_binding) -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(_PassedStage()),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
        ),
    )
    payload = _BusinessUnresolvedPayload(status="UNRESOLVED", reason="insufficient evidence")
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("scenario.business_unresolved"),
            payload=payload,
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.CONTINUE
    assert result.accepted_decision is not None
    assert result.resolution_record is None
    assert result.lifecycle_state.stage is DecisionLifecycleStage.FINALIZATION


@pytest.mark.asyncio
async def test_revision_exhaustion_is_platform_terminal_state(lifecycle_binding) -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(_ChallengedStage()),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
            request_human_on_revision_exhausted=False,
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("scenario.revision.payload"),
            payload=_BusinessUnresolvedPayload(status="UNRESOLVED", reason="insufficient evidence"),
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.BLOCK
    assert result.resolution_record is not None
    assert result.resolution_record.resolution is DecisionResolution.REJECTED


def test_platform_decision_accepted_maps_completed_task_only() -> None:
    assert platform_decision_accepted("completed")
    assert not platform_decision_accepted("failed")
