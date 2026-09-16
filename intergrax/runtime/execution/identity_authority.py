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
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.resume_planner import (
    execution_identity_from_checkpoint,
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
    """Canonical TaskId/RunId/AttemptId/ExecutionId minted for one transport execution."""

    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


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
        attempt_id=attempt_id
        or default_execution_identity_authority.mint_attempt_identity(),
        execution_id=execution_id or mint_execution_id(),
    )


def resolve_root_task_identity(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
    resume_checkpoint: TaskCheckpoint | None = None,
) -> RootTaskIdentity:
    """Resolve root identity, honoring durable resume checkpoint four-ID inputs when present."""
    if resume_checkpoint is not None and resume_checkpoint.runtime is not None:
        checkpoint_run_id, checkpoint_attempt_id = execution_identity_from_checkpoint(
            resume_checkpoint,
        )
        checkpoint_tree = resume_checkpoint.runtime.execution_tree
        checkpoint_root_execution_id = next(
            entry.execution_id
            for entry in checkpoint_tree.entries
            if entry.parent_execution_id is None
        )
        if run_id is not None and run_id != checkpoint_run_id:
            raise ValueError(
                "explicit run_id conflicts with resume checkpoint identity: "
                f"{run_id!r} != {checkpoint_run_id!r}"
            )
        if attempt_id is not None and attempt_id != checkpoint_attempt_id:
            raise ValueError(
                "explicit attempt_id conflicts with resume checkpoint identity: "
                f"{attempt_id!r} != {checkpoint_attempt_id!r}"
            )
        if execution_id is not None and execution_id != checkpoint_root_execution_id:
            raise ValueError(
                "explicit execution_id conflicts with resume checkpoint identity: "
                f"{execution_id!r} != {checkpoint_root_execution_id!r}"
            )
        return mint_root_execution_identity(
            run_id=checkpoint_run_id,
            attempt_id=checkpoint_attempt_id,
            execution_id=execution_id,
        )
    return mint_root_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )


def mint_background_transport_identity() -> BackgroundTransportIdentity:
    """Mint canonical background transport identity before durable persistence."""
    root = mint_root_execution_identity()
    return BackgroundTransportIdentity(
        task_id=mint_task_id(),
        run_id=root.run_id,
        attempt_id=root.attempt_id,
        execution_id=root.execution_id,
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
    "resolve_root_task_identity",
]
