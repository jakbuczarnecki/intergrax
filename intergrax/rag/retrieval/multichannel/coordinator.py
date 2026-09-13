# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic sequential multi-channel retrieval coordinator."""

from __future__ import annotations

from typing import Generic

from intergrax.rag.retrieval.multichannel.contracts import (
    MultiChannelRetrievalResult,
    RetrievalChannelKey,
    RetrievalChannelOperation,
    RetrievalChannelOutcome,
    TResult,
)
from intergrax.rag.retrieval.multichannel.errors import MultiChannelRetrievalContractError


class SequentialMultiChannelRetrievalCoordinator(Generic[TResult]):
    """Execute channel operations in declaration order; no fusion, retry, or abort-on-failure."""

    def execute(
        self,
        operations: tuple[RetrievalChannelOperation[TResult], ...],
    ) -> MultiChannelRetrievalResult[TResult]:
        if not isinstance(operations, tuple):
            raise TypeError("operations must be a tuple")

        _reject_duplicate_channel_keys(operations)

        collected: list[RetrievalChannelOutcome[TResult]] = []
        for operation in operations:
            outcome = operation.execute()
            collected.append(outcome)

        return MultiChannelRetrievalResult(outcomes=tuple(collected))


def _reject_duplicate_channel_keys(
    operations: tuple[RetrievalChannelOperation[TResult], ...],
) -> None:
    seen: set[RetrievalChannelKey] = set()
    for operation in operations:
        key = operation.channel_key
        if not isinstance(key, RetrievalChannelKey):
            raise TypeError("operation.channel_key must be RetrievalChannelKey")
        if key in seen:
            raise MultiChannelRetrievalContractError(
                f"duplicate RetrievalChannelKey in execution plan: {key.value!r}"
            )
        seen.add(key)
