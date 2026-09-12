# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Execution spine admission failure when autonomy guard denies (SELF-HEALING R6.3)."""

from __future__ import annotations

from intergrax.contracts.self_healing.autonomy.execution_authorization import AutonomyExecutionAuthorization


class AutonomyExecutionDeniedError(RuntimeError):
    """Raised by spine admission hooks — does not replace execution authority."""

    def __init__(self, authorization: AutonomyExecutionAuthorization) -> None:
        self.authorization = authorization
        reason = authorization.reasons[0] if authorization.reasons else "autonomy execution denied"
        super().__init__(reason)


__all__ = ["AutonomyExecutionDeniedError"]
