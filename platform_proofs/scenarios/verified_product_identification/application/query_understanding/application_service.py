"""Thin orchestration: raw request → query understanding → production pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.pipeline.contracts import (
    ProductIdentificationPipelineRequest,
    ProductIdentificationPipelineResult,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.service import (
    ProductIdentificationPipelineService,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.contracts import (
    ProductIdentificationQueryUnderstandingResult,
    RawProductIdentificationRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.service import (
    ProductIdentificationQueryUnderstandingService,
)


@dataclass(frozen=True, slots=True)
class RawToPipelineOutcome:
    understanding: ProductIdentificationQueryUnderstandingResult
    pipeline: ProductIdentificationPipelineResult | None


@dataclass(frozen=True, slots=True)
class ProductIdentificationApplicationService:
    query_understanding: ProductIdentificationQueryUnderstandingService
    pipeline: ProductIdentificationPipelineService

    def identify_from_raw(
        self,
        raw: RawProductIdentificationRequest,
        pipeline_request: ProductIdentificationPipelineRequest,
    ) -> RawToPipelineOutcome:
        understanding = self.query_understanding.understand(raw)
        if understanding.query is None:
            return RawToPipelineOutcome(understanding=understanding, pipeline=None)
        typed_request = ProductIdentificationPipelineRequest(
            run_id=pipeline_request.run_id,
            query=understanding.query,
            catalog_content_identity=pipeline_request.catalog_content_identity,
        )
        pipeline_result = self.pipeline.run(typed_request)
        return RawToPipelineOutcome(understanding=understanding, pipeline=pipeline_result)
