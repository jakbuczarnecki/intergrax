# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionResult


class InvocationStatus(str, Enum):
    """
    Persistent invocation state.

    STARTED   — execution began but not completed.
    COMPLETED — execution finished and result is stored.
    """

    STARTED = "started"
    COMPLETED = "completed"


class IdempotencyStore(ABC):
    """
    Ledger-based idempotency port for tool invocations.

    Guarantees:
    - tenant isolation
    - crash safety (STARTED vs COMPLETED)
    - prevention of duplicate side-effects
    - deterministic retry semantics

    Exactly-once semantics for side-effect tools.
    """

    @abstractmethod
    def get_status(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[InvocationStatus]:
        """
        Returns current invocation status or None if not recorded.
        """
        ...

    @abstractmethod
    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: Optional[int] = None,
    ) -> None:
        """
        Records STARTED state atomically.

        If lease_seconds is provided, implementation must guarantee that
        STARTED state expires after given number of seconds to allow takeover.
        """

    @abstractmethod
    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Transitions STARTED -> COMPLETED atomically.

        If completed_ttl_seconds is provided, implementation may expire
        COMPLETED entry after given number of seconds.
        """

    @abstractmethod
    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        """
        Returns previously completed execution result if exists.
        """
        ...