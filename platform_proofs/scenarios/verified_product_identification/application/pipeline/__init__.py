"""VPI production identification pipeline (5C12)."""

from platform_proofs.scenarios.verified_product_identification.application.pipeline.composition import (
    build_product_identification_pipeline,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.contracts import (
    ProductIdentificationPipelineConfiguration,
    ProductIdentificationPipelineRequest,
    ProductIdentificationPipelineResult,
    ProductIdentificationPipelineStageFailure,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.service import (
    ProductIdentificationPipelineService,
)

__all__ = (
    "ProductIdentificationPipelineConfiguration",
    "ProductIdentificationPipelineRequest",
    "ProductIdentificationPipelineResult",
    "ProductIdentificationPipelineService",
    "ProductIdentificationPipelineStageFailure",
    "build_product_identification_pipeline",
)
