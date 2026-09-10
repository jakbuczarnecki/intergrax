"""VPI query understanding — raw user input to ProductIdentificationQuery."""

from platform_proofs.scenarios.verified_product_identification.application.query_understanding.composition import (
    build_product_identification_query_understanding_service,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.contracts import (
    MAX_RAW_QUERY_CHARS,
    ProductIdentificationQueryUnderstandingResult,
    QueryUnderstandingIssue,
    QueryUnderstandingIssueCode,
    RawProductIdentificationRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.query_understanding.service import (
    ProductIdentificationQueryUnderstandingService,
)

__all__ = [
    "MAX_RAW_QUERY_CHARS",
    "ProductIdentificationQueryUnderstandingResult",
    "ProductIdentificationQueryUnderstandingService",
    "QueryUnderstandingIssue",
    "QueryUnderstandingIssueCode",
    "RawProductIdentificationRequest",
    "build_product_identification_query_understanding_service",
]
