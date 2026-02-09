# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import Protocol


class RunRecordStore(Protocol):
    """Read-only access to persisted run records."""

    def get(self, run_id: str):
        ...
