# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionSlot:
    """
    Represents an acquired execution slot.

    slot_id must uniquely identify the acquisition instance.
    Used for safe release.
    """
    slot_id: str


class DistributedExecutionSemaphore(ABC):
    """
    Distributed coordination primitive controlling
    the number of concurrent executions across nodes.

    This is NOT a rate limiter.
    This is a strict concurrency governor.

    Requirements:
    - Multi-node safe
    - Atomic acquire
    - Explicit release
    - Lease/TTL safety in implementation
    """

    @abstractmethod
    def acquire(
        self,
        *,
        tenant_id: str,
        max_parallel: int,
    ) -> ExecutionSlot | None:
        """
        Attempt to acquire execution slot.

        Returns:
            ExecutionSlot if granted
            None if limit reached
        """

    @abstractmethod
    def release(
        self,
        *,
        tenant_id: str,
        slot: ExecutionSlot,
    ) -> None:
        """
        Release previously acquired slot.

        Must be idempotent and safe under retry.
        """