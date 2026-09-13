# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generic multi-channel retrieval coordination (platform mechanism)."""

from intergrax.rag.retrieval.multichannel.contracts import (
    MultiChannelRetrievalCoordinator,
    MultiChannelRetrievalResult,
    RetrievalChannelFailure,
    RetrievalChannelKey,
    RetrievalChannelOperation,
    RetrievalChannelOutcome,
    RetrievalChannelStatus,
)
from intergrax.rag.retrieval.multichannel.coordinator import (
    SequentialMultiChannelRetrievalCoordinator,
)
from intergrax.rag.retrieval.multichannel.errors import MultiChannelRetrievalContractError

__all__ = [
    "MultiChannelRetrievalContractError",
    "MultiChannelRetrievalCoordinator",
    "MultiChannelRetrievalResult",
    "RetrievalChannelFailure",
    "RetrievalChannelKey",
    "RetrievalChannelOperation",
    "RetrievalChannelOutcome",
    "RetrievalChannelStatus",
    "SequentialMultiChannelRetrievalCoordinator",
]
