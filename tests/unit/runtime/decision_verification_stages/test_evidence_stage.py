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
from intergrax.contracts.decision_verification import VerificationDisposition, VerificationStageOutcome
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    VerificationStageUnavailableError,
    verification_stage_registry,
)
from intergrax.contracts.evidence_claims import (
    EvidenceBackedClaim,
    EvidenceClaimSet,
    EvidenceReferenceId,
    mint_evidence_claim_id,
    validate_claim_kind,
    validate_evidence_reference_id,
)
from intergrax.contracts.evidence_verification import EvidenceClaimsProvider, EvidenceReferenceResolver
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_stages.evidence import (
    EVIDENCE_VERIFICATION_STAGE_KIND,
    EvidenceVerificationStage,
    evidence_verification_stage_config,
)

_MODULE_PATHS = (
    Path("intergrax/contracts/evidence_verification.py"),
    Path("intergrax/runtime/decision_verification_stages/evidence.py"),
)
_FORBIDDEN_FRAGMENTS = (
    "runtime.critic",
    "runtime.nexus",
    "L0Gateway",
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
)


@dataclass(frozen=True, slots=True)
class EvidenceCarrierPayload:
    claim_set: EvidenceClaimSet | None


@dataclass(frozen=True, slots=True)
class EvidenceClaimsExtractor:
    def extract(self, candidate: CandidateDecision[EvidenceCarrierPayload]) -> EvidenceClaimSet | None:
        return candidate.artifact.content.claim_set


@dataclass(frozen=True, slots=True)
class InMemoryEvidenceResolver:
    known_ids: frozenset[EvidenceReferenceId]
    available: bool = True

    def is_available(self) -> bool:
        return self.available

    def evidence_exists(self, evidence_id: EvidenceReferenceId) -> bool:
        return evidence_id in self.known_ids


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _candidate(
    claim_set: EvidenceClaimSet | None,
) -> CandidateDecision[EvidenceCarrierPayload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="evidence", subject="subject-1"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("evidence_carrier"),
        content=EvidenceCarrierPayload(claim_set=claim_set),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )


def _complete_claim_set() -> EvidenceClaimSet:
    evidence_id = validate_evidence_reference_id("evidence.ref.1")
    claim = EvidenceBackedClaim(
        claim_id=mint_evidence_claim_id(),
        statement="Bounded claim.",
        claim_kind=validate_claim_kind("generic.claim"),
        supporting_evidence_ids=(evidence_id,),
    )
    return EvidenceClaimSet(claims=(claim,))


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_complete_evidence_passes() -> None:
    claim_set = _complete_claim_set()
    evidence_id = claim_set.claims[0].supporting_evidence_ids[0]
    stage = EvidenceVerificationStage(
        claims_provider=EvidenceClaimsExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset({evidence_id})),
    )
    record = await stage.verify(_candidate(claim_set))
    assert record.outcome is VerificationStageOutcome.PASSED


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_missing_required_evidence_challenged() -> None:
    stage = EvidenceVerificationStage(
        claims_provider=EvidenceClaimsExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset()),
        config=evidence_verification_stage_config(require_claims=True),
    )
    record = await stage.verify(_candidate(None))
    assert record.outcome is VerificationStageOutcome.CHALLENGED
    assert record.challenge is not None
    assert record.challenge.finding.code == "verification.evidence.missing"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_missing_supporting_evidence_challenged() -> None:
    claim = EvidenceBackedClaim(
        claim_id=mint_evidence_claim_id(),
        statement="Bounded claim.",
        claim_kind=validate_claim_kind("generic.claim"),
        supporting_evidence_ids=(),
    )
    claim_set = EvidenceClaimSet(claims=(claim,))
    stage = EvidenceVerificationStage(
        claims_provider=EvidenceClaimsExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset()),
        config=evidence_verification_stage_config(require_supporting_evidence=True),
    )
    record = await stage.verify(_candidate(claim_set))
    assert record.outcome is VerificationStageOutcome.CHALLENGED
    assert record.challenge is not None
    assert record.challenge.finding.code == "verification.evidence.supporting_missing"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unresolved_reference_challenged() -> None:
    evidence_id = validate_evidence_reference_id("evidence.ref.missing")
    claim = EvidenceBackedClaim(
        claim_id=mint_evidence_claim_id(),
        statement="Bounded claim.",
        claim_kind=validate_claim_kind("generic.claim"),
        supporting_evidence_ids=(evidence_id,),
    )
    claim_set = EvidenceClaimSet(claims=(claim,))
    stage = EvidenceVerificationStage(
        claims_provider=EvidenceClaimsExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset()),
    )
    record = await stage.verify(_candidate(claim_set))
    assert record.outcome is VerificationStageOutcome.CHALLENGED
    assert record.challenge is not None
    assert record.challenge.finding.code == "verification.evidence.reference_invalid"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unavailable_resolver_raises() -> None:
    stage = EvidenceVerificationStage(
        claims_provider=EvidenceClaimsExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset(), available=False),
    )
    with pytest.raises(VerificationStageUnavailableError):
        await stage.verify(_candidate(_complete_claim_set()))


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_required_unavailable_integrates_with_pipeline_challenged() -> None:
    stage = EvidenceVerificationStage(
        claims_provider=EvidenceClaimsExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset(), available=False),
    )
    registration = VerificationStageRegistration(
        kind=EVIDENCE_VERIFICATION_STAGE_KIND,
        stage=stage,
        required=True,
    )
    pipeline = VerificationPipeline(registry=verification_stage_registry((registration,)))
    result = await pipeline.verify(_candidate(_complete_claim_set()))
    assert result.disposition is VerificationDisposition.CHALLENGED


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_optional_unavailable_integrates_with_pipeline_skip() -> None:
    unavailable_stage = EvidenceVerificationStage(
        claims_provider=EvidenceClaimsExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset(), available=False),
    )
    from intergrax.runtime.decision_verification_stages.structural import (
        NonEmptyTextStructuralValidator,
        StructuralVerificationStage,
        STRUCTURAL_VERIFICATION_STAGE_KIND,
    )

    @dataclass(frozen=True, slots=True)
    class CombinedTextExtractor:
        def extract(self, content: CombinedPayload) -> str:
            return content.text

    structural = StructuralVerificationStage(
        validators=(
            NonEmptyTextStructuralValidator(
                extractor=CombinedTextExtractor(),
                field_label="text",
            ),
        ),
    )

    @dataclass(frozen=True, slots=True)
    class CombinedPayload:
        text: str
        claim_set: EvidenceClaimSet | None

    @dataclass(frozen=True, slots=True)
    class CombinedEvidenceExtractor:
        def extract(self, candidate: CandidateDecision[CombinedPayload]) -> EvidenceClaimSet | None:
            return candidate.artifact.content.claim_set

    combined_evidence_stage = EvidenceVerificationStage(
        claims_provider=CombinedEvidenceExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset(), available=False),
    )

    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="evidence", subject="subject-2"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("combined"),
        content=CombinedPayload(text="ok", claim_set=None),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    candidate = CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)

    registrations = (
        VerificationStageRegistration(
            kind=EVIDENCE_VERIFICATION_STAGE_KIND,
            stage=combined_evidence_stage,
            required=False,
        ),
        VerificationStageRegistration(
            kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
            stage=structural,
            required=True,
        ),
    )
    pipeline = VerificationPipeline(registry=verification_stage_registry(registrations))
    result = await pipeline.verify(candidate)
    assert result.disposition is VerificationDisposition.PASSED
    assert len(result.stage_records) == 1


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_exact_proposal_identity() -> None:
    claim_set = _complete_claim_set()
    evidence_id = claim_set.claims[0].supporting_evidence_ids[0]
    candidate = _candidate(claim_set)
    stage = EvidenceVerificationStage(
        claims_provider=EvidenceClaimsExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset({evidence_id})),
    )
    record = await stage.verify(candidate)
    assert record.proposal_ref == candidate_decision_ref(candidate)


@pytest.mark.unit
@pytest.mark.gate
def test_execution_class_deterministic() -> None:
    stage = EvidenceVerificationStage(
        claims_provider=EvidenceClaimsExtractor(),
        resolver=InMemoryEvidenceResolver(known_ids=frozenset()),
    )
    assert stage.execution_class is VerificationStageExecutionClass.DETERMINISTIC


@pytest.mark.unit
@pytest.mark.gate
def test_no_critic_dependency() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "runtime.critic" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_forbidden_audit_evidence_modules() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source
