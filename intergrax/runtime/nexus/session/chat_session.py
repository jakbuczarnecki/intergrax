# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deprecated import path — use :mod:`intergrax.memory.contracts.chat_session`."""

from __future__ import annotations

from intergrax.memory.contracts.chat_session import (
    ChatSession,
    SessionCloseReason,
    SessionStatus,
)

__all__ = [
    "ChatSession",
    "SessionCloseReason",
    "SessionStatus",
]
