# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    DecisionVersionLineage,
    candidate_decision_ref,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_verification import (
    VerificationStageKind,
    VerificationStageOutcome,
    VerificationStageRecord,
    validate_verification_stage_kind,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStage,
    VerificationStageAlreadyRegisteredError,
    VerificationStageExecutionClass,
    VerificationStageNotRegisteredError,
    VerificationStageRegistration,
    VerificationStageRegistry,
    is_verification_stage_registered,
    register_verification_stage,
    require_registered_verification_stage,
    verification_stage_registry,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "intergrax"
    / "contracts"
    / "decision_verification_stage.py"
)

_FORBIDDEN_IMPORT_FRAGMENTS = (
    "runtime.nexus",
    "runtime.critic",
    "runtime.human",
    "runtime.policy",
    "runtime.governance",
)

_FORBIDDEN_PRODUCTION_PATTERNS = (
    r"\bAny\b",
    r"\bcast\b",
    r"type:\s*ignore",
    r"pyright:\s*ignore",
    r"\bgetattr\b",
    r"\bsetattr\b",
    r"\bhasattr\b",
    r"\binspect\b",
    r"\bexec\b",
    r"\beval\b",
    r"object\.__setattr__",
    r"dict\[str,\s*Any\]",
)


@dataclass(frozen=True, slots=True)
class IncidentDecisionPayload:
    recommendation: str


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _candidate() -> CandidateDecision[IncidentDecisionPayload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("incident_resolution"),
        content=IncidentDecisionPayload(recommendation="escalate"),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )


@dataclass(frozen=True, slots=True)
class PassedStage:
    kind: VerificationStageKind
    execution_class: VerificationStageExecutionClass

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=self.kind,
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class MismatchKindStage:
    kind: VerificationStageKind = validate_verification_stage_kind("other")
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(
        self,
        candidate: CandidateDecision[IncidentDecisionPayload],
    ) -> VerificationStageRecord:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=self.kind,
            outcome=VerificationStageOutcome.PASSED,
        )


def _registration(
    *,
    kind: str,
    stage: VerificationStage[IncidentDecisionPayload],
    required: bool = True,
) -> VerificationStageRegistration[IncidentDecisionPayload]:
    return VerificationStageRegistration(
        kind=validate_verification_stage_kind(kind),
        stage=stage,
        required=required,
    )


@pytest.mark.unit
@pytest.mark.gate
def test_stage_kind_registration_works() -> None:
    stage = PassedStage(
        kind=validate_verification_stage_kind("schema"),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    registration = _registration(kind="schema", stage=stage)
    registry = verification_stage_registry((registration,))
    assert registry.registrations == (registration,)
    assert is_verification_stage_registered(registry, "schema") is True


@pytest.mark.unit
@pytest.mark.gate
def test_kind_mismatch_rejected() -> None:
    stage = MismatchKindStage()
    with pytest.raises(ValueError, match="must match stage.kind"):
        _registration(kind="schema", stage=stage)


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_kind_rejected() -> None:
    stage = PassedStage(
        kind=validate_verification_stage_kind("schema"),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    registration = _registration(kind="schema", stage=stage)
    registry = verification_stage_registry((registration,))
    with pytest.raises(VerificationStageAlreadyRegisteredError):
        register_verification_stage(registry, registration)


@pytest.mark.unit
@pytest.mark.gate
def test_registry_immutable() -> None:
    stage = PassedStage(
        kind=validate_verification_stage_kind("schema"),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    registration = _registration(kind="schema", stage=stage)
    registry = verification_stage_registry((registration,))
    with pytest.raises((AttributeError, FrozenInstanceError)):
        setattr(registry, "registrations", ())


@pytest.mark.unit
@pytest.mark.gate
def test_register_returns_new_registry() -> None:
    alpha = _registration(
        kind="alpha",
        stage=PassedStage(
            kind=validate_verification_stage_kind("alpha"),
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
        ),
    )
    beta = _registration(
        kind="beta",
        stage=PassedStage(
            kind=validate_verification_stage_kind("beta"),
            execution_class=VerificationStageExecutionClass.PROBABILISTIC,
        ),
    )
    registry1 = verification_stage_registry((alpha,))
    registry2 = register_verification_stage(registry1, beta)
    assert registry2 is not registry1
    assert registry2.registrations == (alpha, beta)


@pytest.mark.unit
@pytest.mark.gate
def test_required_metadata_preserved() -> None:
    required = _registration(
        kind="schema",
        stage=PassedStage(
            kind=validate_verification_stage_kind("schema"),
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
        ),
        required=True,
    )
    optional = _registration(
        kind="semantic",
        stage=PassedStage(
            kind=validate_verification_stage_kind("semantic"),
            execution_class=VerificationStageExecutionClass.PROBABILISTIC,
        ),
        required=False,
    )
    registry = verification_stage_registry((required, optional))
    assert registry.registrations[0].required is True
    assert registry.registrations[1].required is False


@pytest.mark.unit
@pytest.mark.gate
def test_execution_class_typed() -> None:
    deterministic = PassedStage(
        kind=validate_verification_stage_kind("rules"),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    probabilistic = PassedStage(
        kind=validate_verification_stage_kind("semantic"),
        execution_class=VerificationStageExecutionClass.PROBABILISTIC,
    )
    assert deterministic.execution_class is VerificationStageExecutionClass.DETERMINISTIC
    assert probabilistic.execution_class is VerificationStageExecutionClass.PROBABILISTIC


@pytest.mark.unit
@pytest.mark.gate
def test_unknown_stage_lookup_fails_closed() -> None:
    stage = PassedStage(
        kind=validate_verification_stage_kind("schema"),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    registration = _registration(kind="schema", stage=stage)
    registry = verification_stage_registry((registration,))
    unknown = validate_verification_stage_kind("unknown")
    with pytest.raises(VerificationStageNotRegisteredError):
        require_registered_verification_stage(registry, unknown)


@pytest.mark.unit
@pytest.mark.gate
def test_no_runtime_critic_nexus_dependencies() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    lowered = source.lower()
    for fragment in _FORBIDDEN_IMPORT_FRAGMENTS:
        assert fragment not in lowered
    assert "importlib" not in source
    assert "entry_points" not in source


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_stage_protocol_accepts_candidate_and_returns_record() -> None:
    @dataclass(frozen=True, slots=True)
    class LocalStage:
        kind: VerificationStageKind = validate_verification_stage_kind("local")
        execution_class: VerificationStageExecutionClass = (
            VerificationStageExecutionClass.DETERMINISTIC
        )

        async def verify(
            self,
            candidate: CandidateDecision[IncidentDecisionPayload],
        ) -> VerificationStageRecord:
            return verification_stage_record(
                proposal_ref=candidate_decision_ref(candidate),
                stage=self.kind,
                outcome=VerificationStageOutcome.PASSED,
            )

    stage = LocalStage()
    assert isinstance(stage, VerificationStage)
    candidate = _candidate()
    record = await stage.verify(candidate)
    assert type(record) is VerificationStageRecord
    assert record.proposal_ref == candidate_decision_ref(candidate)
    assert record.stage == stage.kind


@pytest.mark.unit
@pytest.mark.gate
def test_forbidden_production_patterns_absent() -> None:
    import re

    source = _MODULE_PATH.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_PRODUCTION_PATTERNS:
        assert re.search(pattern, source) is None, pattern


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_noncanonical_order_rejected() -> None:
    alpha = _registration(
        kind="alpha",
        stage=PassedStage(
            kind=validate_verification_stage_kind("alpha"),
            execution_class=VerificationStageExecutionClass.DETERMINISTIC,
        ),
    )
    beta = _registration(
        kind="beta",
        stage=PassedStage(
            kind=validate_verification_stage_kind("beta"),
            execution_class=VerificationStageExecutionClass.PROBABILISTIC,
        ),
    )
    with pytest.raises(ValueError, match="canonical order"):
        VerificationStageRegistry(registrations=(beta, alpha))
