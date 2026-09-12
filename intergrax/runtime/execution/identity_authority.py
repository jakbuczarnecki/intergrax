# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ExecutionRuntime-owned identity minting authority (NPSC-3C-F-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_identity_authority import (
    ExecutionIdentityAuthorityPort,
    MintedExecutionIdentity,
)


@dataclass(frozen=True, slots=True)
class RootTaskIdentity:
    """Resolved root Run and Attempt identifiers plus minted root ExecutionId."""

    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class BackgroundTransportIdentity:
    """Canonical TaskId/RunId/AttemptId minted for one background transport execution."""

    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId


class DefaultExecutionIdentityAuthority:
    """Default port implementation — sole production mint surface for execution lifecycle."""

    def mint_execution_identity(
        self,
        *,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
        execution_id: ExecutionId | None = None,
    ) -> MintedExecutionIdentity:
        if execution_id is None:
            root = mint_root_execution_identity(
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=None,
            )
            return MintedExecutionIdentity(
                run_id=root.run_id,
                attempt_id=root.attempt_id,
                execution_id=root.execution_id,
            )
        return MintedExecutionIdentity(
            run_id=run_id or mint_run_id(),
            attempt_id=attempt_id or mint_attempt_id(),
            execution_id=execution_id,
        )

    def mint_attempt_identity(self) -> AttemptId:
        return mint_attempt_id()

    def mint_run_identity(self) -> RunId:
        return mint_run_id()

    def mint_child_execution_identity(self) -> ExecutionId:
        return mint_child_execution_id()

    def close_identity(self) -> None:
        """Identity values are immutable; lifecycle close is owned by ExecutionRuntime."""
        return None


default_execution_identity_authority: ExecutionIdentityAuthorityPort = (
    DefaultExecutionIdentityAuthority()
)


def mint_root_execution_identity(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> RootTaskIdentity:
    """Mint canonical generic root identity for one root execution invocation."""
    return RootTaskIdentity(
        run_id=run_id or default_execution_identity_authority.mint_run_identity(),
        attempt_id=attempt_id or default_execution_identity_authority.mint_attempt_identity(),
        execution_id=execution_id or mint_execution_id(),
    )


def mint_background_transport_identity() -> BackgroundTransportIdentity:
    """Mint canonical background transport identity before durable persistence."""
    return BackgroundTransportIdentity(
        task_id=mint_task_id(),
        run_id=default_execution_identity_authority.mint_run_identity(),
        attempt_id=default_execution_identity_authority.mint_attempt_identity(),
    )


def mint_child_execution_id() -> ExecutionId:
    """Mint a child ExecutionId under the active parent execution tree."""
    return mint_execution_id()


def mint_retry_attempt_id() -> AttemptId:
    """Mint the next canonical AttemptId for a durable retry transition."""
    return default_execution_identity_authority.mint_attempt_identity()


__all__ = [
    "BackgroundTransportIdentity",
    "DefaultExecutionIdentityAuthority",
    "RootTaskIdentity",
    "default_execution_identity_authority",
    "mint_background_transport_identity",
    "mint_child_execution_id",
    "mint_retry_attempt_id",
    "mint_root_execution_identity",
]
