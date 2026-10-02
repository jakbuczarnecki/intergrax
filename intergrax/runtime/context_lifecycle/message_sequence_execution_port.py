# © Artur Czarnecki. All rights reserved.

"""Message-sequence artifact executor port (UCL / Token Optimization contract surface)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.runtime.token_optimization.message_sequence_artifact import (
        MessageSequenceArtifactExecutionRequest,
        MessageSequenceArtifactExecutionResult,
    )


@runtime_checkable
class MessageSequenceArtifactExecutionPort(Protocol):
    """Minimal execution surface used by UCL artifact materialization."""

    def execute(
        self,
        request: MessageSequenceArtifactExecutionRequest,
    ) -> MessageSequenceArtifactExecutionResult: ...


__all__ = [
    "MessageSequenceArtifactExecutionPort",
]
