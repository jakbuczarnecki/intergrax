# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Q3 web-search functional diagnostic specification (DIAG-FUNCTIONAL-Q3)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import EventId
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.runtime.diagnostics.functional_diagnostic_specification import (
    CandidateExistsRequirement,
    FunctionalDiagnosticCheck,
    FunctionalDiagnosticRequirement,
    FunctionalDiagnosticRequirementKind,
    FunctionalDiagnosticSpecification,
    OperationOutcomeStatusRequirement,
    OutputRelationExistsRequirement,
    SelectionArtifactMatchRequirement,
    ValidationOutcomeRequirement,
    validate_functional_diagnostic_specification,
)
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.observability.functional_validation_evidence import FunctionalValidationOutcome

Q3_WEB_SPECIFICATION_ID = FunctionalDiagnosticSpecificationId(
    "fdspec_000000000000000000000000a3b30001",
)
Q3_WEB_SPECIFICATION_VERSION = 1
Q3_WEB_QUERY_ID = "q3-web-search-1"
Q3_WEB_QUERY_CONSTRUCT_OPERATION_ID = "web.query.construct"
Q3_WEB_SEARCH_OPERATION_ID = "web.search.query"
Q3_WEB_EXTRACT_OPERATION_ID = "web.extract"
Q3_WEB_SYNTHESIZE_OPERATION_ID = "web.synthesize"
Q3_EXPECTED_OFFICIAL_SOURCE_ARTIFACT = (
    "url:https://www.python.org/downloads/release/python-3120"
)

CHECK_Q3_QUERY = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000021")
CHECK_Q3_SEARCH = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000022")
CHECK_Q3_CANDIDATES = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000023")
CHECK_Q3_SELECTION = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000024")
CHECK_Q3_EXTRACTION_RELATION = FunctionalDiagnosticCheckId(
    "fdcheck_00000000000000000000000000000025",
)
CHECK_Q3_EXTRACTION_VALIDATION = FunctionalDiagnosticCheckId(
    "fdcheck_00000000000000000000000000000027",
)
CHECK_Q3_FINAL = FunctionalDiagnosticCheckId("fdcheck_00000000000000000000000000000026")


def build_q3_web_search_functional_diagnostic_specification(
    *,
    query_validation_id: EventId | None = None,
    extraction_validation_id: EventId | None = None,
    final_validation_id: EventId | None = None,
    include_query_validation: bool = True,
    include_extraction_validation: bool = True,
    include_final_validation: bool = True,
    expected_selection_artifact_ref: str = Q3_EXPECTED_OFFICIAL_SOURCE_ARTIFACT,
) -> FunctionalDiagnosticSpecification:
    checks: list[FunctionalDiagnosticCheck] = []
    if include_query_validation:
        if query_validation_id is None:
            raise ValueError("query_validation_id is required when include_query_validation is True")
        checks.append(
            FunctionalDiagnosticCheck(
                check_id=CHECK_Q3_QUERY,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
                    validation_outcome=ValidationOutcomeRequirement(
                        validation_id=query_validation_id,
                        expected_outcome=FunctionalValidationOutcome.PASSED,
                    ),
                ),
                pass_claim="Constructed query matched expected search intent.",
                fail_claim="Constructed query did not match expected search intent.",
                insufficient_claim="No linked query validation evidence.",
            ),
        )
    checks.extend(
        [
            FunctionalDiagnosticCheck(
                check_id=CHECK_Q3_SEARCH,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id=Q3_WEB_SEARCH_OPERATION_ID,
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Web search provider invocation succeeded.",
                fail_claim="Web search provider invocation failed.",
                insufficient_claim="No search operation outcome evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=CHECK_Q3_CANDIDATES,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id=Q3_WEB_QUERY_ID),
                ),
                pass_claim="Search provider returned ranked source candidates.",
                fail_claim="No search source candidates were recorded.",
                insufficient_claim="No candidate rank evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=CHECK_Q3_SELECTION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH,
                    selection_artifact_match=SelectionArtifactMatchRequirement(
                        query_id=Q3_WEB_QUERY_ID,
                        expected_artifact_ref=expected_selection_artifact_ref,
                    ),
                ),
                pass_claim="Official authoritative source was selected.",
                fail_claim="Wrong web source was selected.",
                insufficient_claim="No source selection evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=CHECK_Q3_EXTRACTION_RELATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OUTPUT_RELATION_EXISTS,
                    output_relation_exists=OutputRelationExistsRequirement(
                        operation_id=Q3_WEB_EXTRACT_OPERATION_ID,
                    ),
                ),
                pass_claim="Extraction output relation was recorded.",
                fail_claim="Extraction output relation missing.",
                insufficient_claim="No extraction output relation evidence.",
            ),
        ],
    )
    if include_extraction_validation:
        if extraction_validation_id is None:
            raise ValueError(
                "extraction_validation_id is required when include_extraction_validation is True",
            )
        checks.append(
            FunctionalDiagnosticCheck(
                check_id=CHECK_Q3_EXTRACTION_VALIDATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
                    validation_outcome=ValidationOutcomeRequirement(
                        validation_id=extraction_validation_id,
                        expected_outcome=FunctionalValidationOutcome.PASSED,
                    ),
                ),
                pass_claim="Extracted fact matched oracle expectation.",
                fail_claim="Extracted fact did not match oracle expectation.",
                insufficient_claim="No linked extraction validation evidence.",
            ),
        )
    if include_final_validation:
        if final_validation_id is None:
            raise ValueError("final_validation_id is required when include_final_validation is True")
        checks.append(
            FunctionalDiagnosticCheck(
                check_id=CHECK_Q3_FINAL,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
                    validation_outcome=ValidationOutcomeRequirement(
                        validation_id=final_validation_id,
                        expected_outcome=FunctionalValidationOutcome.PASSED,
                    ),
                ),
                pass_claim="Final answer oracle validation passed.",
                fail_claim="Final answer oracle validation failed.",
                insufficient_claim="No linked final validation evidence.",
            ),
        )
    return validate_functional_diagnostic_specification(
        FunctionalDiagnosticSpecification(
            specification_id=Q3_WEB_SPECIFICATION_ID,
            version=Q3_WEB_SPECIFICATION_VERSION,
            checks=tuple(checks),
        ),
    )


__all__ = [
    "CHECK_Q3_CANDIDATES",
    "CHECK_Q3_EXTRACTION_RELATION",
    "CHECK_Q3_EXTRACTION_VALIDATION",
    "CHECK_Q3_FINAL",
    "CHECK_Q3_QUERY",
    "CHECK_Q3_SEARCH",
    "CHECK_Q3_SELECTION",
    "Q3_EXPECTED_OFFICIAL_SOURCE_ARTIFACT",
    "Q3_WEB_EXTRACT_OPERATION_ID",
    "Q3_WEB_QUERY_CONSTRUCT_OPERATION_ID",
    "Q3_WEB_QUERY_ID",
    "Q3_WEB_SEARCH_OPERATION_ID",
    "Q3_WEB_SPECIFICATION_ID",
    "Q3_WEB_SPECIFICATION_VERSION",
    "Q3_WEB_SYNTHESIZE_OPERATION_ID",
    "build_q3_web_search_functional_diagnostic_specification",
]
