"""Orchestrates raw request → authoritative ProductIdentificationQuery."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.product_identification_query import (
    ProductIdentificationQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationInputOrigin,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.contracts import (
    ProductIdentificationQueryInterpreter,
    ProductIdentificationQueryUnderstandingResult,
    QueryUnderstandingExtractionBundle,
    QueryUnderstandingIssue,
    QueryUnderstandingIssueCode,
    QueryUnderstandingObservedPayload,
    RawProductIdentificationRequest,
    raw_input_fingerprint,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.extractors import (
    ProductIdentifierExtractor,
    StructuredConstraintExtractor,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.normalization import (
    normalize_search_text,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.policies import (
    QueryUnderstandingMergePolicy,
)


def _blocking(issues: tuple[QueryUnderstandingIssue, ...]) -> bool:
    return any(
        issue.code
        in (
            QueryUnderstandingIssueCode.CONFLICTING_USER_CONSTRAINT,
            QueryUnderstandingIssueCode.AMBIGUOUS_IDENTIFIER_TYPE,
            QueryUnderstandingIssueCode.NO_ACTIONABLE_SEMANTICS,
        )
        for issue in issues
    )


@dataclass(frozen=True, slots=True)
class ProductIdentificationQueryUnderstandingService:
    identifier_extractor: ProductIdentifierExtractor
    structured_extractor: StructuredConstraintExtractor
    merge_policy: QueryUnderstandingMergePolicy
    interpreter: ProductIdentificationQueryInterpreter | None = None

    def understand(
        self,
        request: RawProductIdentificationRequest,
    ) -> ProductIdentificationQueryUnderstandingResult:
        raw_text = request.raw_text
        search_text = normalize_search_text(raw_text)

        id_records, id_issues = self.identifier_extractor.extract(raw_text)
        (
            required,
            negative,
            soft,
            missing,
            constraint_issues,
        ) = self.structured_extractor.extract(raw_text)

        deterministic_bundle = QueryUnderstandingExtractionBundle(
            identifiers=id_records,
            required_constraints=required,
            negative_constraints=negative,
            soft_preferences=soft,
            missing_requirements=missing,
        )

        interpreter_candidate = None
        if self.interpreter is not None:
            interpreter_candidate = self.interpreter.interpret(request)

        merged, merge_issues = self.merge_policy.merge(
            deterministic_bundle,
            interpreter_candidate,
        )

        issues = _sorted_issues((*id_issues, *constraint_issues, *merge_issues))

        if _only_punctuation(search_text):
            issues = _sorted_issues(
                issues
                + (
                    QueryUnderstandingIssue(
                        code=QueryUnderstandingIssueCode.NO_ACTIONABLE_SEMANTICS,
                        detail="no actionable semantics in raw input",
                    ),
                )
            )

        observation = QueryUnderstandingObservedPayload(
            raw_input_sha256_prefix=raw_input_fingerprint(raw_text),
            extracted_identifiers=merged.identifiers,
            required_constraints=merged.required_constraints,
            negative_constraints=merged.negative_constraints,
            soft_preferences=merged.soft_preferences,
            issues=issues,
            search_text=search_text,
        )

        if _blocking(issues):
            return ProductIdentificationQueryUnderstandingResult(
                query=None,
                issues=issues,
                extraction=merged,
                observation=observation,
                pipeline_input_origin=ProductIdentificationInputOrigin.RAW_QUERY,
            )

        context = ProductIdentificationQueryContext(
            requested_identifiers=tuple(item.identifier for item in merged.identifiers),
            required_constraints=tuple(
                item.constraint for item in merged.required_constraints
            ),
            negative_constraints=tuple(
                item.constraint for item in merged.negative_constraints
            ),
            soft_preferences=tuple(item.preference for item in merged.soft_preferences),
            missing_user_distinguishing_requirements=merged.missing_requirements,
        )

        query = ProductIdentificationQuery(
            verification_context=context,
            search_text=search_text,
        )

        return ProductIdentificationQueryUnderstandingResult(
            query=query,
            issues=issues,
            extraction=merged,
            observation=observation,
            pipeline_input_origin=ProductIdentificationInputOrigin.RAW_QUERY,
        )


def _sorted_issues(issues: tuple[QueryUnderstandingIssue, ...]) -> tuple[QueryUnderstandingIssue, ...]:
    return tuple(sorted(issues, key=lambda i: (i.code.value, i.detail or "")))


def _only_punctuation(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return all(not character.isalnum() for character in stripped)

