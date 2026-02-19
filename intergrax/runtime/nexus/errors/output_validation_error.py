# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputValidationError(RuntimeError):
    """
    Raised when RuntimeAnswer violates hard production output constraints.

    This is an ENGINE-level invariant breach.
    It must be treated as non-retryable and non-policy error.
    """

    run_id: str
    reason_code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.reason_code}] {self.message} (run_id={self.run_id})"
