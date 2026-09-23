# © Artur Czarnecki. All rights reserved.

"""Provider-neutral suspended operation entity keys (UCA-6C-R6)."""

from __future__ import annotations

from uuid import uuid4


def mint_suspended_operation_id() -> str:
    """Mint a durable entity key — not an execution or continuation identity."""
    return f"sop_{uuid4().hex}"


__all__ = ["mint_suspended_operation_id"]
