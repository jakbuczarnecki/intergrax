# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed errors for generic multi-channel retrieval coordination."""

from __future__ import annotations


class MultiChannelRetrievalContractError(ValueError):
    """Invalid coordinator input or channel outcome invariants violated."""

    def __init__(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        super().__init__(message)
