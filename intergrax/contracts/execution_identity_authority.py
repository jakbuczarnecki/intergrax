# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution identity authority port (EE-A2).

Canonical **runtime implementation** lives in
``intergrax.runtime.execution.identity_authority`` and is invoked only from
``ExecutionRuntime``, ``ChildExecutionRunner``, and ``AttemptLifecycleService``
(retry AttemptId). This module documents the enterprise contract surface;
it does not mint identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId

CANONICAL_IDENTITY_AUTHORITY_MODULE = "intergrax.runtime.execution.identity_authority"
CANONICAL_LIFECYCLE_OWNER_MODULE = "intergrax.runtime.execution.runtime"
CANONICAL_ATTEMPT_LIFECYCLE_MODULE = "intergrax.runtime.execution.attempt_lifecycle.service"


@dataclass(frozen=True, slots=True)
class MintedExecutionIdentity:
    """Immutable root execution identity tuple admitted into lifecycle."""

    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


class ExecutionIdentityAuthorityPort(Protocol):
    """Port over runtime-owned identity minting (no parallel mint system)."""

    def mint_execution_identity(
        self,
        *,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
        execution_id: ExecutionId | None = None,
    ) -> MintedExecutionIdentity:
        """Mint or complete the root Run / Attempt / Execution triple for one admission."""
        ...

    def mint_attempt_identity(self) -> AttemptId:
        """Mint the next AttemptId for a durable retry transition (AttemptLifecycleService)."""
        ...

    def mint_run_identity(self) -> RunId:
        """Mint a standalone RunId when admission supplies no run (background transport)."""
        ...

    def mint_child_execution_identity(self) -> ExecutionId:
        """Mint a child ExecutionId under an active parent execution tree."""
        ...

    def close_identity(self) -> None:
        """Lifecycle close is owned by ExecutionRuntime; identity values remain immutable."""
        ...


__all__ = [
    "CANONICAL_ATTEMPT_LIFECYCLE_MODULE",
    "CANONICAL_IDENTITY_AUTHORITY_MODULE",
    "CANONICAL_LIFECYCLE_OWNER_MODULE",
    "ExecutionIdentityAuthorityPort",
    "MintedExecutionIdentity",
]
