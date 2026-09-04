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
from intergrax.contracts.decision_verification import VerificationStageOutcome
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.guardrail_verification import assess_guardrail_scan
from intergrax.integrations.contracts.llm_guardrail import GuardrailScanResult
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_stages.guardrail import (
    GUARDRAIL_VERIFICATION_STAGE_KIND,
    GuardrailScanProvider,
    GuardrailVerificationStage,
)

_MODULE_PATHS = (
    Path("intergrax/contracts/guardrail_verification.py"),
    Path("intergrax/runtime/decision_verification_stages/guardrail.py"),
)
_FORBIDDEN_FRAGMENTS = (
    "runtime.critic",
    "runtime.policy",
    "runtime.governance",
    "policy_bridge",
    "PolicyBridge",
    "L0Gateway",
    "CriticOrchestrator",
    "LayerVerdict",
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
class GuardrailCarrierPayload:
    scan: GuardrailScanResult | None


@dataclass(frozen=True, slots=True)
class GuardrailScanExtractor:
    def extract(self, candidate: CandidateDecision[GuardrailCarrierPayload]) -> GuardrailScanResult | None:
        return candidate.artifact.content.scan


def _execution_lineage() -> DecisionExecutionLineage:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )

    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _candidate(
    scan: GuardrailScanResult | None,
) -> CandidateDecision[GuardrailCarrierPayload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="guardrail", subject="subject-1"),
        tenant_id="tenant-a",
        execution=_execution_lineage(),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("guardrail_carrier"),
        content=GuardrailCarrierPayload(scan=scan),
    )
    lineage = DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return CandidateDecision(
        identity=identity,
        artifact=artifact,
        lineage=lineage,
    )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_quality_guard_failure_challenged() -> None:
    stage = GuardrailVerificationStage(
        scan_provider=GuardrailScanExtractor(),
    )
    scan = GuardrailScanResult(allowed=False, detail="output blocked")
    record = await stage.verify(_candidate(scan))
    assert record.outcome is VerificationStageOutcome.CHALLENGED
    assert record.challenge is not None
    assert record.challenge.requirement_code == "verification.guardrail.output_blocked"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_quality_guard_pass_passed() -> None:
    stage = GuardrailVerificationStage(
        scan_provider=GuardrailScanExtractor(),
    )
    scan = GuardrailScanResult(allowed=True, categories=("pii_email",))
    record = await stage.verify(_candidate(scan))
    assert record.outcome is VerificationStageOutcome.PASSED


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_missing_scan_passes() -> None:
    stage = GuardrailVerificationStage(
        scan_provider=GuardrailScanExtractor(),
    )
    record = await stage.verify(_candidate(None))
    assert record.outcome is VerificationStageOutcome.PASSED


@pytest.mark.unit
@pytest.mark.gate
def test_no_policy_imports_in_stage_modules() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in (
            "runtime.policy",
            "runtime.governance",
            "policy_bridge",
            "PolicyBridge",
        ):
            assert fragment not in source


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_stable_deterministic_ordering() -> None:
    stage = GuardrailVerificationStage(
        scan_provider=GuardrailScanExtractor(),
    )
    scan = GuardrailScanResult(allowed=False, detail="blocked")
    candidate = _candidate(scan)
    first = await stage.verify(candidate)
    second = await stage.verify(candidate)
    assert first == second


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_exact_proposal_ref() -> None:
    candidate = _candidate(GuardrailScanResult(allowed=True))
    stage = GuardrailVerificationStage(
        scan_provider=GuardrailScanExtractor(),
    )
    record = await stage.verify(candidate)
    assert record.proposal_ref == candidate_decision_ref(candidate)


@pytest.mark.unit
@pytest.mark.gate
def test_execution_class_deterministic() -> None:
    stage = GuardrailVerificationStage(
        scan_provider=GuardrailScanExtractor(),
    )
    assert stage.execution_class is VerificationStageExecutionClass.DETERMINISTIC


@pytest.mark.unit
@pytest.mark.gate
def test_migration_parity_blocked_guardrail_scan() -> None:
    scan = GuardrailScanResult(allowed=False, detail="output blocked")
    assessment = assess_guardrail_scan(scan)
    assert assessment.passed is False


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_pipeline_integration() -> None:
    stage = GuardrailVerificationStage(
        scan_provider=GuardrailScanExtractor(),
    )
    registration = VerificationStageRegistration(
        kind=GUARDRAIL_VERIFICATION_STAGE_KIND,
        stage=stage,
        required=True,
    )
    pipeline = VerificationPipeline(registry=verification_stage_registry((registration,)))
    result = await pipeline.verify(
        _candidate(GuardrailScanResult(allowed=False, detail="blocked")),
    )
    assert result.disposition.value == "challenged"


@pytest.mark.unit
@pytest.mark.gate
def test_forbidden_audit_guardrail_modules() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source
