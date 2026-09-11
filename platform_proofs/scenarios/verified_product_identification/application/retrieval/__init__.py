"""Scenario-owned multi-channel retrieval orchestration."""

from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
    ExactChannelRetrievalOutcome,
    LexicalChannelRetrievalOutcome,
    MultiChannelRetrievalRequest,
    MultiChannelRetrievalResult,
    RetrievalChannelExecutionStatus,
    RetrievalExecutionPolicy,
    RetrievalExecutionSummary,
    StructuredChannelRetrievalOutcome,
    VectorChannelRetrievalOutcome,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.service import (
    MultiChannelRetrievalService,
)

__all__ = (
    "ExactChannelRetrievalOutcome",
    "LexicalChannelRetrievalOutcome",
    "MultiChannelRetrievalRequest",
    "MultiChannelRetrievalResult",
    "MultiChannelRetrievalService",
    "RetrievalChannelExecutionStatus",
    "RetrievalExecutionPolicy",
    "RetrievalExecutionSummary",
    "StructuredChannelRetrievalOutcome",
    "VectorChannelRetrievalOutcome",
)
