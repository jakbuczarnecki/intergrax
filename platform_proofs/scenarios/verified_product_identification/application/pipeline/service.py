"""Canonical VPI production pipeline orchestration (5C12)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationSelectionRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    OfferCandidateFusionRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityHypothesisConfiguration,
    ProductIdentityHypothesisRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.errors import (
    IdentityEvidenceUnavailableError,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    IdentityHypothesisEvaluationRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationRunId,
    ProductIdentificationStage,
    StageFailureObservedPayload,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.ports import (
    MonotonicClockPort,
    ObservationSinkError,
    ProductIdentificationObservationSink,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.recorder import (
    ProductIdentificationObservationRecorder,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.contracts import (
    ClarificationRequirementSelectionPort,
    IdentityHypothesisEvaluationPort,
    MultiChannelRetrievalPort,
    OfferCandidateFusionPort,
    ProductIdentificationPipelineConfiguration,
    ProductIdentificationPipelineRequest,
    ProductIdentificationPipelineResult,
    ProductIdentificationPipelineStageFailure,
    ProductIdentificationVerificationPort,
    ProductIdentityHypothesisPort,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.observation_mapping import (
    build_clarification_payload,
    build_fusion_payload,
    build_identity_evaluation_payload,
    build_identity_hypotheses_payload,
    build_query_context_payload,
    build_retrieval_channel_payloads,
    build_terminal_payload,
    build_verification_payload,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.stage_timing import (
    execute_timed_stage,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
    MultiChannelRetrievalResult,
    RetrievalChannelExecutionStatus,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationVerificationRequest,
)


def _allocate_run_id(request: ProductIdentificationPipelineRequest) -> ProductIdentificationRunId:
    if request.run_id is not None:
        return request.run_id
    return ProductIdentificationRunId(value=str(uuid.uuid4()))


def _first_retrieval_failure(
    retrieval: MultiChannelRetrievalResult,
) -> CatalogSearchFailure | None:
    for outcome in (
        retrieval.exact,
        retrieval.lexical,
        retrieval.structured,
        retrieval.vector,
    ):
        if outcome.status is RetrievalChannelExecutionStatus.FAILED and outcome.failure is not None:
            return outcome.failure
    return None


def _retrieval_stage_failed(retrieval: MultiChannelRetrievalResult) -> bool:
    summary = retrieval.execution_summary
    return summary.channels_failed > 0 and summary.channels_succeeded == 0


@dataclass(frozen=True, slots=True)
class ProductIdentificationPipelineService:
    retrieval_service: MultiChannelRetrievalPort
    fusion_service: OfferCandidateFusionPort
    identity_service: ProductIdentityHypothesisPort
    identity_evaluation_service: IdentityHypothesisEvaluationPort
    verification_service: ProductIdentificationVerificationPort
    clarification_service: ClarificationRequirementSelectionPort
    observation_sink: ProductIdentificationObservationSink
    clock: MonotonicClockPort
    configuration: ProductIdentificationPipelineConfiguration = (
        ProductIdentificationPipelineConfiguration()
    )

    def run(self, request: ProductIdentificationPipelineRequest) -> ProductIdentificationPipelineResult:
        run_id = _allocate_run_id(request)
        recorder = ProductIdentificationObservationRecorder(
            run_id=run_id,
            sink=self.observation_sink,
            sink_mode=self.configuration.observation_sink_mode,
        )
        try:
            return self._run_business_path(request, run_id=run_id, recorder=recorder)
        except ObservationSinkError:
            return ProductIdentificationPipelineResult(
                run_id=run_id,
                decision=None,
                clarification=None,
                stage_failure=ProductIdentificationPipelineStageFailure(
                    stage=ProductIdentificationStage.TERMINAL,
                    observation_sink_failed=True,
                ),
            )

    def _run_business_path(
        self,
        request: ProductIdentificationPipelineRequest,
        *,
        run_id: ProductIdentificationRunId,
        recorder: ProductIdentificationObservationRecorder,
    ) -> ProductIdentificationPipelineResult:
        query_payload = build_query_context_payload(
            input_origin=request.input_origin,
            query_context=request.query_context,
            catalog_content_identity=request.catalog_content_identity,
        )
        recorder.record_payload(
            stage=ProductIdentificationStage.QUERY_CONTEXT,
            payload=query_payload,
        )
        execute_timed_stage(
            recorder=recorder,
            clock=self.clock,
            stage=ProductIdentificationStage.QUERY_CONTEXT,
            operation=lambda: None,
        )

        retrieval, _ = execute_timed_stage(
            recorder=recorder,
            clock=self.clock,
            stage=ProductIdentificationStage.RETRIEVAL,
            operation=lambda: self.retrieval_service.retrieve(request.retrieval_request),
        )
        channel_payloads = build_retrieval_channel_payloads(
            retrieval,
            configuration=self.configuration,
            channel_durations_ns={},
        )
        for channel_payload in channel_payloads:
            recorder.record_payload(
                stage=ProductIdentificationStage.RETRIEVAL,
                payload=channel_payload,
            )

        if _retrieval_stage_failed(retrieval):
            failure = _first_retrieval_failure(retrieval)
            return self._fail_stage(
                recorder=recorder,
                run_id=run_id,
                stage=ProductIdentificationStage.RETRIEVAL,
                catalog_failure=failure,
            )

        fused, _ = execute_timed_stage(
            recorder=recorder,
            clock=self.clock,
            stage=ProductIdentificationStage.FUSION,
            operation=lambda: self.fusion_service.fuse(
                OfferCandidateFusionRequest(
                    candidates=retrieval.candidates,
                    limit=self.configuration.fusion_candidate_limit,
                )
            ),
        )
        recorder.record_payload(
            stage=ProductIdentificationStage.FUSION,
            payload=build_fusion_payload(retrieval, fused, configuration=self.configuration),
        )

        try:
            identity, _ = execute_timed_stage(
                recorder=recorder,
                clock=self.clock,
                stage=ProductIdentificationStage.IDENTITY_HYPOTHESIS,
                operation=lambda: self.identity_service.form_hypotheses(
                    ProductIdentityHypothesisRequest(
                        fused_candidates=fused,
                        configuration=IdentityHypothesisConfiguration(
                            max_candidates=self.configuration.identity_max_candidates,
                        ),
                    )
                ),
            )
        except IdentityEvidenceUnavailableError as error:
            catalog_failure = CatalogSearchFailure(
                kind=CatalogSearchFailureKind.UNAVAILABLE,
                message=str(error),
            )
            return self._fail_stage(
                recorder=recorder,
                run_id=run_id,
                stage=ProductIdentificationStage.IDENTITY_HYPOTHESIS,
                catalog_failure=catalog_failure,
            )

        recorder.record_payload(
            stage=ProductIdentificationStage.IDENTITY_HYPOTHESIS,
            payload=build_identity_hypotheses_payload(identity),
        )

        ranked, _ = execute_timed_stage(
            recorder=recorder,
            clock=self.clock,
            stage=ProductIdentificationStage.IDENTITY_EVALUATION,
            operation=lambda: self.identity_evaluation_service.evaluate(
                IdentityHypothesisEvaluationRequest(hypotheses=identity),
            ),
        )
        recorder.record_payload(
            stage=ProductIdentificationStage.IDENTITY_EVALUATION,
            payload=build_identity_evaluation_payload(ranked),
        )

        verification_outcome, _ = execute_timed_stage(
            recorder=recorder,
            clock=self.clock,
            stage=ProductIdentificationStage.VERIFICATION,
            operation=lambda: self.verification_service.run(
                ProductIdentificationVerificationRequest(
                    ranked_hypotheses=ranked,
                    query_context=request.query_context,
                )
            ),
        )
        if verification_outcome.failure is not None:
            return self._fail_stage(
                recorder=recorder,
                run_id=run_id,
                stage=ProductIdentificationStage.VERIFICATION,
                catalog_failure=verification_outcome.failure,
            )
        decision = verification_outcome.decision
        if decision is None:
            raise RuntimeError("verification outcome missing decision on success path")

        recorder.record_payload(
            stage=ProductIdentificationStage.VERIFICATION,
            payload=build_verification_payload(decision),
        )

        clarification, _ = execute_timed_stage(
            recorder=recorder,
            clock=self.clock,
            stage=ProductIdentificationStage.CLARIFICATION,
            operation=lambda: self.clarification_service.select(
                ClarificationSelectionRequest(
                    decision=decision,
                    query_context=request.query_context,
                )
            ),
        )
        recorder.record_payload(
            stage=ProductIdentificationStage.CLARIFICATION,
            payload=build_clarification_payload(clarification),
        )

        recorder.record_payload(
            stage=ProductIdentificationStage.TERMINAL,
            payload=build_terminal_payload(decision, clarification),
        )

        return ProductIdentificationPipelineResult(
            run_id=run_id,
            decision=decision,
            clarification=clarification,
            stage_failure=None,
        )

    def _fail_stage(
        self,
        *,
        recorder: ProductIdentificationObservationRecorder,
        run_id: ProductIdentificationRunId,
        stage: ProductIdentificationStage,
        catalog_failure: CatalogSearchFailure | None,
    ) -> ProductIdentificationPipelineResult:
        recorder.record_payload(
            stage=ProductIdentificationStage.TERMINAL,
            payload=StageFailureObservedPayload(
                failed_stage=stage,
                catalog_failure=catalog_failure,
            ),
        )
        return ProductIdentificationPipelineResult(
            run_id=run_id,
            decision=None,
            clarification=None,
            stage_failure=ProductIdentificationPipelineStageFailure(
                stage=stage,
                catalog_failure=catalog_failure,
            ),
        )
