# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""C1 / LKW RAG functional diagnostic specification (DIAG-FUNCTIONAL-Q1)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import EventId
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.runtime.diagnostics.functional_diagnostic_specification import (
    FunctionalDiagnosticCheck,
    FunctionalDiagnosticRequirement,
    FunctionalDiagnosticRequirementKind,
    FunctionalDiagnosticSpecification,
    CandidateExistsRequirement,
    OperationOutcomeStatusRequirement,
    OutputRelationExistsRequirement,
    SelectionArtifactMatchRequirement,
    ValidationOutcomeRequirement,
    validate_functional_diagnostic_specification,
)
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.observability.functional_validation_evidence import FunctionalValidationOutcome

C1_RAG_SPECIFICATION_ID = FunctionalDiagnosticSpecificationId(
    "fdspec_000000000000000000000000c1aa0001",
)
C1_RAG_SPECIFICATION_VERSION = 1
C1_RAG_QUERY_ID = "c1-rag-retrieval-1"
C1_RAG_RETRIEVE_OPERATION_ID = "rag.retrieve"
C1_RAG_SYNTHESIZE_OPERATION_ID = "rag.synthesize"
C1_RAG_EXPECTED_SELECTION_ARTIFACT = "chunk:incident-report"

CHECK_C1_RETRIEVAL_OPERATION = FunctionalDiagnosticCheckId(
    "fdcheck_00000000000000000000000000000001",
)
CHECK_C1_CANDIDATES = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000002")
CHECK_C1_SELECTION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000003")
CHECK_C1_OUTPUT_RELATION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000004")
CHECK_C1_VALIDATION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000005")


def build_c1_rag_functional_diagnostic_specification(
    *,
    validation_id: EventId | None = None,
    include_output_relation: bool = False,
    include_validation: bool = True,
    expected_selection_artifact_ref: str = C1_RAG_EXPECTED_SELECTION_ARTIFACT,
) -> FunctionalDiagnosticSpecification:
    checks: list[FunctionalDiagnosticCheck] = [
        FunctionalDiagnosticCheck(
            check_id=CHECK_C1_RETRIEVAL_OPERATION,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                operation_outcome_status=OperationOutcomeStatusRequirement(
                    operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
                    expected_status=PipelineOperationStatus.SUCCEEDED,
                ),
            ),
            pass_claim="RAG retrieval operation succeeded.",
            fail_claim="RAG retrieval operation failed.",
            insufficient_claim="No retrieval operation outcome evidence.",
        ),
        FunctionalDiagnosticCheck(
            check_id=CHECK_C1_CANDIDATES,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                candidate_exists=CandidateExistsRequirement(query_id=C1_RAG_QUERY_ID),
            ),
            pass_claim="Retrieval produced ranked candidates.",
            fail_claim="No retrieval candidates were recorded.",
            insufficient_claim="No candidate rank evidence.",
        ),
        FunctionalDiagnosticCheck(
            check_id=CHECK_C1_SELECTION,
            requirement=FunctionalDiagnosticRequirement(
                kind=FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH,
                selection_artifact_match=SelectionArtifactMatchRequirement(
                    query_id=C1_RAG_QUERY_ID,
                    expected_artifact_ref=expected_selection_artifact_ref,
                ),
            ),
            pass_claim="Incident report chunk was selected.",
            fail_claim="Wrong retrieval selection was recorded.",
            insufficient_claim="No selection evidence.",
        ),
    ]
    if include_output_relation:
        checks.append(
            FunctionalDiagnosticCheck(
                check_id=CHECK_C1_OUTPUT_RELATION,
                dependencies=(CHECK_C1_SELECTION,),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OUTPUT_RELATION_EXISTS,
                    output_relation_exists=OutputRelationExistsRequirement(
                        operation_id=C1_RAG_SYNTHESIZE_OPERATION_ID,
                    ),
                ),
                pass_claim="Synthesis output relation was recorded.",
                fail_claim="Synthesis output relation missing.",
                insufficient_claim="No synthesis output relation evidence.",
            ),
        )
    if include_validation:
        if validation_id is None:
            raise ValueError("validation_id is required when include_validation is True")
        checks.append(
            FunctionalDiagnosticCheck(
                check_id=CHECK_C1_VALIDATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
                    validation_outcome=ValidationOutcomeRequirement(
                        validation_id=validation_id,
                        expected_outcome=FunctionalValidationOutcome.PASSED,
                    ),
                ),
                pass_claim="External oracle validation passed.",
                fail_claim="External oracle validation failed.",
                insufficient_claim="No linked validation evidence.",
            ),
        )
    return validate_functional_diagnostic_specification(
        FunctionalDiagnosticSpecification(
            specification_id=C1_RAG_SPECIFICATION_ID,
            version=C1_RAG_SPECIFICATION_VERSION,
            checks=tuple(checks),
        ),
    )


__all__ = [
    "C1_RAG_EXPECTED_SELECTION_ARTIFACT",
    "C1_RAG_QUERY_ID",
    "C1_RAG_RETRIEVE_OPERATION_ID",
    "C1_RAG_SPECIFICATION_ID",
    "C1_RAG_SPECIFICATION_VERSION",
    "C1_RAG_SYNTHESIZE_OPERATION_ID",
    "CHECK_C1_CANDIDATES",
    "CHECK_C1_OUTPUT_RELATION",
    "CHECK_C1_RETRIEVAL_OPERATION",
    "CHECK_C1_SELECTION",
    "CHECK_C1_VALIDATION",
    "build_c1_rag_functional_diagnostic_specification",
]
