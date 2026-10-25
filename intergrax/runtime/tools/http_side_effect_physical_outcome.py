# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""HTTP side-effect physical outcome classification for durable external-operation state."""

from __future__ import annotations


def http_side_effect_outcome_uncertain_after_submit(exc: BaseException) -> bool:
    """True when a side-effectful HTTP call may have committed despite transport failure."""
    if isinstance(exc, RuntimeError):
        message = str(exc)
        return message.startswith("http_transport_error:")
    return False


__all__ = ["http_side_effect_outcome_uncertain_after_submit"]
