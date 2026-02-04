# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPolicy:
    """
    Defines execution governance rules for a run.
    """
    max_retries: int
    timeout_seconds: int

    @classmethod
    def default(cls) -> ExecutionPolicy:
        return cls(
            max_retries=2,
            timeout_seconds=300,
        )
