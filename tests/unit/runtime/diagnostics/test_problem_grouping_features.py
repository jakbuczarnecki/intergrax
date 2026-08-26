# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticCertainty,
    DiagnosticFinding,
    DiagnosticFindingKind,
    DiagnosticLimitation,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnomalyKind,
    LifecycleAnomalyScope,
)
from intergrax.runtime.diagnostics.problem_grouping_features import (
    MAX_TEXT_EVIDENCE_CHARS,
    ProblemGroupingFeatureIntegrityError,
    ProblemGroupingTextEvidenceSourceKind,
    project_assessment_features,
    semantic_input_from_assessment,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"


def _finding(*, claim: str = "A lifecycle event was recorded after canonical run closure.") -> DiagnosticFinding:
    event_id = mint_event_id()
    return DiagnosticFinding(
        kind=DiagnosticFindingKind.EVENT_AFTER_TERMINAL,
        scope=LifecycleAnomalyScope.EXECUTION,
        attempt_id=None,
        certainty=DiagnosticCertainty.PROVEN,
        claim=claim,
        source_anomaly_kind=LifecycleAnomalyKind.EVENT_AFTER_TERMINAL,
        supporting_event_ids=(event_id,),
        supporting_evidence_ids=(),
        supporting_positions=(),
    )


def _limitation() -> DiagnosticLimitation:
    event_id = mint_event_id()
    return DiagnosticLimitation(
        kind=DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED,
        factual_message="Runtime history is truncated; conclusions requiring the unseen tail cannot be proven.",
        source_anomaly_kind=LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED,
        supporting_event_ids=(event_id,),
        supporting_evidence_ids=(),
        supporting_positions=(),
    )


def _assessment(
    *,
    findings: tuple[DiagnosticFinding, ...] = (),
    limitations: tuple[DiagnosticLimitation, ...] = (),
) -> DiagnosticAssessment:
    return DiagnosticAssessment(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        findings=findings,
        limitations=limitations,
    )


def test_project_assessment_features_links_structural_signature_and_text() -> None:
    assessment = _assessment(findings=(_finding(),), limitations=(_limitation(),))
    features = project_assessment_features(assessment)

    assert features.representation_version == "1"
    assert len(features.text_evidence) == 2
    assert features.text_evidence[0].source_kind is ProblemGroupingTextEvidenceSourceKind.OPERATOR_CLAIM
    assert features.text_evidence[1].source_kind is ProblemGroupingTextEvidenceSourceKind.FACTUAL_LIMITATION
    assert features.structural_signature.findings
    assert features.structural_signature.limitations


def test_semantic_input_preserves_subject_identity() -> None:
    assessment = _assessment(findings=(_finding(),))
    semantic_input = semantic_input_from_assessment(assessment)

    assert semantic_input.subject.tenant_id == assessment.tenant_id
    assert semantic_input.subject.task_id == assessment.task_id
    assert semantic_input.subject.run_id == assessment.run_id
    assert semantic_input.features is not None


def test_project_rejects_oversized_text() -> None:
    oversized = "x" * (MAX_TEXT_EVIDENCE_CHARS + 1)
    assessment = _assessment(findings=(_finding(claim=oversized),))
    with pytest.raises(ProblemGroupingFeatureIntegrityError):
        project_assessment_features(assessment)
