# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
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
    VerificationDisposition,
    VerificationStageOutcome,
    validate_verification_finding_code,
    validate_verification_requirement_code,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    VerificationStageUnavailableError,
    verification_stage_registry,
)
from intergrax.contracts.domain_verification import (
    DomainVerificationOutcome,
    DomainVerifierId,
    DomainVerificationIndependenceConfig,
    domain_verification_failed,
    domain_verification_passed,
)
from intergrax.contracts.semantic_verification import VerifierIndependenceMode
from intergrax.runtime.execution.inference_profile import InferenceProfileId
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_stages.domain import (
    DOMAIN_VERIFICATION_STAGE_KIND,
    IndependentDomainVerificationStage,
)
from intergrax.runtime.decision_verification_stages.structural import (
    STRUCTURAL_VERIFICATION_STAGE_KIND,
    NonEmptyTextStructuralValidator,
    StructuralVerificationStage,
)

_MODULE_PATHS = (
    Path("intergrax/contracts/domain_verification.py"),
    Path("intergrax/runtime/decision_verification_stages/domain.py"),
)
_FORBIDDEN_FRAGMENTS = (
    "runtime.critic",
    "runtime.nexus",
    "runtime.human",
    "runtime.policy",
    "runtime.governance",
    "DecisionLifecycleHost",
    "CriticOrchestrator",
    "Any",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr",
    "setattr",
    "hasattr",
    "inspect",
    "exec(",
    "eval(",
    "object.__setattr__",
    "dict[str, Any]",
    "str(self.verifier.verifier_id",
)

_DOMAIN_REQUIREMENT = validate_verification_requirement_code(
    "verification.domain.requirement_failed",
)
_DOMAIN_FINDING = validate_verification_finding_code(
    "verification.domain.requirement_failed",
)


@dataclass(frozen=True, slots=True)
class DomainPayload:
    value: str


@dataclass(frozen=True, slots=True)
class StubDomainVerifier:
    verifier_id_value: DomainVerifierId
    available: bool = True
    outcome: DomainVerificationOutcome = domain_verification_passed()
    verify_calls: list[CandidateDecision[DomainPayload]] | None = None

    @property
    def verifier_id(self) -> DomainVerifierId:
        return self.verifier_id_value

    def is_available(self) -> bool:
        return self.available

    def verify(self, candidate: CandidateDecision[DomainPayload]) -> DomainVerificationOutcome:
        if self.verify_calls is not None:
            self.verify_calls.append(candidate)
        return self.outcome


def _candidate(value: str = "ok") -> CandidateDecision[DomainPayload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="domain", subject="subject-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("domain_payload"),
        content=DomainPayload(value=value),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)


def _stage(
    *,
    verifier: StubDomainVerifier,
    execution_class: VerificationStageExecutionClass,
    independence: DomainVerificationIndependenceConfig | None = None,
) -> IndependentDomainVerificationStage[DomainPayload]:
    return IndependentDomainVerificationStage(
        verifier=verifier,
        execution_class=execution_class,
        independence=independence,
    )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_deterministic_domain_pass() -> None:
    stage = _stage(
        verifier=StubDomainVerifier(verifier_id_value=DomainVerifierId("domain.legal")),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    record = await stage.verify(_candidate())
    assert record.outcome is VerificationStageOutcome.PASSED


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_deterministic_domain_challenge() -> None:
    verifier = StubDomainVerifier(
        verifier_id_value=DomainVerifierId("domain.legal"),
        outcome=domain_verification_failed(
            requirement_code=_DOMAIN_REQUIREMENT,
            finding_code=_DOMAIN_FINDING,
            message="domain requirement failed",
        ),
    )
    stage = _stage(
        verifier=verifier,
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    record = await stage.verify(_candidate())
    assert record.outcome is VerificationStageOutcome.CHALLENGED
    assert record.challenge is not None
    assert record.challenge.finding.code == "verification.domain.requirement_failed"


@pytest.mark.unit
@pytest.mark.gate
def test_probabilistic_domain_execution_class() -> None:
    stage = _stage(
        verifier=StubDomainVerifier(verifier_id_value=DomainVerifierId("domain.parts")),
        execution_class=VerificationStageExecutionClass.PROBABILISTIC,
    )
    assert stage.execution_class is VerificationStageExecutionClass.PROBABILISTIC


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_exact_proposal_identity() -> None:
    candidate = _candidate()
    stage = _stage(
        verifier=StubDomainVerifier(verifier_id_value=DomainVerifierId("domain.legal")),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    record = await stage.verify(candidate)
    assert record.proposal_ref == candidate_decision_ref(candidate)


@pytest.mark.unit
@pytest.mark.gate
def test_stable_verifier_identity() -> None:
    verifier = StubDomainVerifier(verifier_id_value=DomainVerifierId("domain.security"))
    stage = _stage(
        verifier=verifier,
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    assert stage.verifier.verifier_id == DomainVerifierId("domain.security")


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unavailable_domain_verifier_raises() -> None:
    stage = _stage(
        verifier=StubDomainVerifier(
            verifier_id_value=DomainVerifierId("domain.legal"),
            available=False,
        ),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    with pytest.raises(VerificationStageUnavailableError):
        await stage.verify(_candidate())


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_required_unavailable_integrates_with_pipeline() -> None:
    stage = _stage(
        verifier=StubDomainVerifier(
            verifier_id_value=DomainVerifierId("domain.legal"),
            available=False,
        ),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    registration = VerificationStageRegistration(
        kind=DOMAIN_VERIFICATION_STAGE_KIND,
        stage=stage,
        required=True,
    )
    pipeline = VerificationPipeline(registry=verification_stage_registry((registration,)))
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.CHALLENGED


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_optional_unavailable_integrates_with_pipeline_skip() -> None:
    unavailable = _stage(
        verifier=StubDomainVerifier(
            verifier_id_value=DomainVerifierId("domain.legal"),
            available=False,
        ),
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )

    @dataclass(frozen=True, slots=True)
    class TextExtractor:
        def extract(self, content: DomainPayload) -> str:
            return content.value

    structural = StructuralVerificationStage(
        validators=(
            NonEmptyTextStructuralValidator(
                extractor=TextExtractor(),
                field_label="value",
            ),
        ),
    )
    registrations = (
        VerificationStageRegistration(
            kind=DOMAIN_VERIFICATION_STAGE_KIND,
            stage=unavailable,
            required=False,
        ),
        VerificationStageRegistration(
            kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
            stage=structural,
            required=True,
        ),
    )
    pipeline = VerificationPipeline(registry=verification_stage_registry(registrations))
    result = await pipeline.verify(_candidate())
    assert result.disposition is VerificationDisposition.PASSED
    assert len(result.stage_records) == 1


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invalid_verifier_identity_propagates_before_verify() -> None:
    verifier = StubDomainVerifier(
        verifier_id_value=DomainVerifierId("   "),
        verify_calls=[],
    )
    stage = _stage(
        verifier=verifier,
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    with pytest.raises(ValueError):
        await stage.verify(_candidate())
    assert verifier.verify_calls == []


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_domain_verifier_does_not_mutate_candidate() -> None:
    candidate = _candidate("original")
    verifier = StubDomainVerifier(
        verifier_id_value=DomainVerifierId("domain.legal"),
        verify_calls=[],
    )
    stage = _stage(
        verifier=verifier,
        execution_class=VerificationStageExecutionClass.DETERMINISTIC,
    )
    _ = await stage.verify(candidate)
    assert candidate.artifact.content.value == "original"
    assert verifier.verify_calls is not None
    assert verifier.verify_calls[0] is candidate


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_independent_distinct_profiles_allowed() -> None:
    stage = _stage(
        verifier=StubDomainVerifier(verifier_id_value=DomainVerifierId("domain.legal")),
        execution_class=VerificationStageExecutionClass.PROBABILISTIC,
        independence=DomainVerificationIndependenceConfig(
            mode=VerifierIndependenceMode.INDEPENDENT,
            producer_profile_id=InferenceProfileId("profile-a"),
            verifier_profile_id=InferenceProfileId("profile-b"),
        ),
    )
    record = await stage.verify(_candidate())
    assert record.outcome is VerificationStageOutcome.PASSED


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_independent_same_profile_challenged_before_verifier() -> None:
    verifier = StubDomainVerifier(
        verifier_id_value=DomainVerifierId("domain.legal"),
        verify_calls=[],
    )
    stage = _stage(
        verifier=verifier,
        execution_class=VerificationStageExecutionClass.PROBABILISTIC,
        independence=DomainVerificationIndependenceConfig(
            mode=VerifierIndependenceMode.INDEPENDENT,
            producer_profile_id=InferenceProfileId("profile-a"),
            verifier_profile_id=InferenceProfileId("profile-a"),
        ),
    )
    record = await stage.verify(_candidate())
    assert record.outcome is VerificationStageOutcome.CHALLENGED
    assert record.challenge is not None
    assert record.challenge.finding.code == "verification.domain.profile_not_independent"
    assert verifier.verify_calls == []


@pytest.mark.unit
@pytest.mark.gate
def test_shared_profile_same_profiles_valid_config() -> None:
    config = DomainVerificationIndependenceConfig(
        mode=VerifierIndependenceMode.SHARED_PROFILE,
        producer_profile_id=InferenceProfileId("profile-a"),
        verifier_profile_id=InferenceProfileId("profile-a"),
    )
    assert config.mode is VerifierIndependenceMode.SHARED_PROFILE


@pytest.mark.unit
@pytest.mark.gate
def test_shared_profile_different_profiles_rejected_at_config() -> None:
    with pytest.raises(ValueError, match="SHARED_PROFILE"):
        DomainVerificationIndependenceConfig(
            mode=VerifierIndependenceMode.SHARED_PROFILE,
            producer_profile_id=InferenceProfileId("profile-a"),
            verifier_profile_id=InferenceProfileId("profile-b"),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_no_critic_dependency() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "runtime.critic" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_forbidden_audit_domain_modules() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source
