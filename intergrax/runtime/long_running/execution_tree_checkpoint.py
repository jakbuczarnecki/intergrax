# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical Execution Tree checkpoint models and resume planning (UE-9C)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)


class ExecutionCheckpointStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    PENDING = "pending"
    RUNNING = "running"


class ExecutionPriorOutput(BaseModel):
    agent_id: str
    summary: str
    status: str
    graph_node_id: str | None = None


class ExecutionCheckpointEntry(BaseModel):
    execution_id: ExecutionId
    parent_execution_id: ExecutionId | None = None
    status: ExecutionCheckpointStatus
    graph_node_id: str | None = None
    resumed_from_execution_id: ExecutionId | None = None
    prior_output: ExecutionPriorOutput | None = None

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("parent_execution_id", mode="before")
    @classmethod
    def _validate_parent_execution_id(cls, value: object) -> ExecutionId | None:
        if value is None:
            return None
        return validate_execution_id(value)

    @field_validator("resumed_from_execution_id", mode="before")
    @classmethod
    def _validate_resumed_from_execution_id(cls, value: object) -> ExecutionId | None:
        if value is None:
            return None
        return validate_execution_id(value)


class ExecutionTreeSnapshot(BaseModel):
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    entries: list[ExecutionCheckpointEntry] = Field(default_factory=list)

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object) -> TaskId:
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    def validate_tree(self) -> None:
        if not self.entries:
            raise ValueError("execution tree must contain at least one entry")
        roots = [entry for entry in self.entries if entry.parent_execution_id is None]
        if len(roots) != 1:
            raise ValueError("execution tree must contain exactly one root entry")
        known_ids = {entry.execution_id for entry in self.entries}
        for entry in self.entries:
            if entry.parent_execution_id is not None:
                if entry.parent_execution_id not in known_ids:
                    raise ValueError(
                        "execution tree parent missing: "
                        f"{entry.parent_execution_id!r} for {entry.execution_id!r}"
                    )
        self._assert_acyclic()

    def validate_for_task(self, *, task_id: TaskId, run_id: RunId) -> None:
        if self.task_id != task_id:
            raise ValueError(
                f"execution tree task_id mismatch: {self.task_id!r} != {task_id!r}"
            )
        if self.run_id != run_id:
            raise ValueError(
                f"execution tree run_id mismatch: {self.run_id!r} != {run_id!r}"
            )
        self.validate_tree()

    def entry_by_execution_id(self, execution_id: ExecutionId) -> ExecutionCheckpointEntry | None:
        for entry in self.entries:
            if entry.execution_id == execution_id:
                return entry
        return None

    def entry_by_graph_node_id(self, graph_node_id: str) -> ExecutionCheckpointEntry | None:
        for entry in self.entries:
            if entry.graph_node_id == graph_node_id:
                return entry
        return None

    def completed_graph_node_ids(self) -> frozenset[str]:
        completed: set[str] = set()
        for entry in self.entries:
            if entry.graph_node_id is None:
                continue
            if entry.status is not ExecutionCheckpointStatus.COMPLETED:
                continue
            if entry.prior_output is None:
                continue
            completed.add(entry.graph_node_id)
        return frozenset(completed)

    def _assert_acyclic(self) -> None:
        children: dict[ExecutionId, list[ExecutionId]] = {}
        for entry in self.entries:
            if entry.parent_execution_id is None:
                continue
            children.setdefault(entry.parent_execution_id, []).append(entry.execution_id)

        def visit(execution_id: ExecutionId, stack: set[ExecutionId]) -> None:
            if execution_id in stack:
                raise ValueError("execution tree contains a cycle")
            stack.add(execution_id)
            for child_id in children.get(execution_id, ()):
                visit(child_id, stack)
            stack.remove(execution_id)

        root = next(entry.execution_id for entry in self.entries if entry.parent_execution_id is None)
        visit(root, set())


class ExecutionTreeRecorder:
    """In-memory recorder for the active attempt execution tree."""

    __slots__ = ("_snapshot",)

    def __init__(self, snapshot: ExecutionTreeSnapshot) -> None:
        self._snapshot = snapshot

    @classmethod
    def start_root(
        cls,
        *,
        task_id: TaskId,
        run_id: RunId,
        attempt_id: AttemptId,
        root_execution_id: ExecutionId,
    ) -> ExecutionTreeRecorder:
        snapshot = ExecutionTreeSnapshot(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            entries=[
                ExecutionCheckpointEntry(
                    execution_id=root_execution_id,
                    parent_execution_id=None,
                    status=ExecutionCheckpointStatus.RUNNING,
                )
            ],
        )
        return cls(snapshot)

    @classmethod
    def from_snapshot(cls, snapshot: ExecutionTreeSnapshot) -> ExecutionTreeRecorder:
        snapshot.validate_tree()
        return cls(snapshot.model_copy(deep=True))

    @property
    def snapshot(self) -> ExecutionTreeSnapshot:
        return self._snapshot.model_copy(deep=True)

    def record_child_started(
        self,
        *,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId,
        graph_node_id: str,
        resumed_from_execution_id: ExecutionId | None = None,
    ) -> None:
        if self._snapshot.entry_by_execution_id(execution_id) is not None:
            raise ValueError(f"duplicate execution_id in active tree: {execution_id!r}")
        self._snapshot.entries.append(
            ExecutionCheckpointEntry(
                execution_id=execution_id,
                parent_execution_id=parent_execution_id,
                status=ExecutionCheckpointStatus.RUNNING,
                graph_node_id=graph_node_id,
                resumed_from_execution_id=resumed_from_execution_id,
            )
        )

    def record_completed(
        self,
        execution_id: ExecutionId,
        *,
        prior_output: ExecutionPriorOutput,
    ) -> None:
        entry = self._require_entry(execution_id)
        index = self._snapshot.entries.index(entry)
        self._snapshot.entries[index] = entry.model_copy(
            update={
                "status": ExecutionCheckpointStatus.COMPLETED,
                "prior_output": prior_output,
            }
        )

    def record_failed(
        self,
        execution_id: ExecutionId,
        *,
        prior_output: ExecutionPriorOutput | None = None,
    ) -> None:
        entry = self._require_entry(execution_id)
        index = self._snapshot.entries.index(entry)
        self._snapshot.entries[index] = entry.model_copy(
            update={
                "status": ExecutionCheckpointStatus.FAILED,
                "prior_output": prior_output,
            }
        )

    def mark_interrupted(self, execution_id: ExecutionId) -> None:
        entry = self._require_entry(execution_id)
        if entry.status is ExecutionCheckpointStatus.COMPLETED:
            raise ValueError(
                f"cannot mark completed execution as interrupted: {execution_id!r}"
            )
        index = self._snapshot.entries.index(entry)
        self._snapshot.entries[index] = entry.model_copy(
            update={"status": ExecutionCheckpointStatus.INTERRUPTED}
        )

    def adopt_historical_entry(self, entry: ExecutionCheckpointEntry) -> None:
        if entry.status is not ExecutionCheckpointStatus.COMPLETED:
            raise ValueError("only completed historical entries can be adopted")
        if self._snapshot.entry_by_execution_id(entry.execution_id) is not None:
            return
        self._snapshot.entries.append(entry.model_copy(deep=True))

    def mark_running_interrupted(self) -> None:
        for entry in list(self._snapshot.entries):
            if entry.status is ExecutionCheckpointStatus.RUNNING:
                self.mark_interrupted(entry.execution_id)

    def _require_entry(self, execution_id: ExecutionId) -> ExecutionCheckpointEntry:
        entry = self._snapshot.entry_by_execution_id(execution_id)
        if entry is None:
            raise ValueError(f"unknown execution_id in active tree: {execution_id!r}")
        return entry


class ExecutionTreeResumePlan(BaseModel):
    historical_snapshot: ExecutionTreeSnapshot
    active_snapshot: ExecutionTreeSnapshot
    skip_graph_node_ids: frozenset[str] = Field(default_factory=frozenset)
    resume_graph_node_ids: frozenset[str] = Field(default_factory=frozenset)
    historical_by_graph_node_id: dict[str, ExecutionCheckpointEntry] = Field(
        default_factory=dict
    )


def minimal_execution_tree_snapshot(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    root_execution_id: ExecutionId,
) -> ExecutionTreeSnapshot:
    return ExecutionTreeSnapshot(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        entries=[
            ExecutionCheckpointEntry(
                execution_id=root_execution_id,
                parent_execution_id=None,
                status=ExecutionCheckpointStatus.RUNNING,
            )
        ],
    )


def minimal_runtime_checkpoint(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    root_execution_id: ExecutionId,
) -> "RuntimeCheckpoint":
    from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint

    return RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=minimal_execution_tree_snapshot(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=root_execution_id,
        ),
    )


def build_execution_tree_resume_plan(
    checkpoint_tree: ExecutionTreeSnapshot,
    *,
    task_id: TaskId,
    run_id: RunId,
    new_attempt_id: AttemptId,
    new_root_execution_id: ExecutionId,
) -> ExecutionTreeResumePlan:
    checkpoint_tree.validate_for_task(task_id=task_id, run_id=run_id)
    historical_entries = [entry.model_copy(deep=True) for entry in checkpoint_tree.entries]
    for entry in historical_entries:
        if entry.status is ExecutionCheckpointStatus.RUNNING:
            entry.status = ExecutionCheckpointStatus.INTERRUPTED

    active_recorder = ExecutionTreeRecorder.start_root(
        task_id=task_id,
        run_id=run_id,
        attempt_id=new_attempt_id,
        root_execution_id=new_root_execution_id,
    )
    skip_nodes: set[str] = set()
    resume_nodes: set[str] = set()
    historical_by_node: dict[str, ExecutionCheckpointEntry] = {}
    for entry in historical_entries:
        if entry.graph_node_id is None:
            continue
        historical_by_node[entry.graph_node_id] = entry
        if (
            entry.status is ExecutionCheckpointStatus.COMPLETED
            and entry.prior_output is not None
        ):
            skip_nodes.add(entry.graph_node_id)
            active_recorder.adopt_historical_entry(entry)
        elif entry.status in (
            ExecutionCheckpointStatus.INTERRUPTED,
            ExecutionCheckpointStatus.RUNNING,
            ExecutionCheckpointStatus.PENDING,
            ExecutionCheckpointStatus.FAILED,
        ):
            resume_nodes.add(entry.graph_node_id)

    active_snapshot = active_recorder.snapshot
    active_snapshot.validate_tree()
    return ExecutionTreeResumePlan(
        historical_snapshot=ExecutionTreeSnapshot(
            task_id=checkpoint_tree.task_id,
            run_id=checkpoint_tree.run_id,
            attempt_id=checkpoint_tree.attempt_id,
            entries=historical_entries,
        ),
        active_snapshot=active_snapshot,
        skip_graph_node_ids=frozenset(skip_nodes),
        resume_graph_node_ids=frozenset(resume_nodes),
        historical_by_graph_node_id=historical_by_node,
    )
