# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared interaction intake errors."""

from __future__ import annotations


class HostNotAcceptingWorkError(Exception):
    """Raised when the application host cannot accept new execution work."""

    def __init__(self, error_id: str, detail: str = "") -> None:
        self.error_id = error_id
        self.detail = detail or error_id
        super().__init__(self.detail)
