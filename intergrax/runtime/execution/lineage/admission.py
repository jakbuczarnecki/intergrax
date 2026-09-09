# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution lineage admission hooks (DG-001 R1)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptScope,
    ExecutionLineageIntegrityError,
    ExecutionLineagePersistence,
    ExecutionLineageUnavailableError,
)
from intergrax.runtime.execution.lineage.active_lineage import (
    bind_attempt_lineage_degradation,
)


class ExecutionLineageRootAdmissionHook:
    """Durable root admission executed before delegate."""

    __slots__ = (
        "_persistence",
        "_scope",
        "_segment_root_execution_id",
        "_execution_id",
        "_graph_node_id",
    )

    def __init__(
        self,
        *,
        persistence: ExecutionLineagePersistence,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        graph_node_id: str | None = None,
    ) -> None:
        self._persistence = persistence
        self._scope = scope
        self._segment_root_execution_id = segment_root_execution_id
        self._execution_id = execution_id
        self._graph_node_id = graph_node_id

    async def admit(self, request: object) -> None:
        del request
        self._persistence.admit_root(
            self._scope,
            self._segment_root_execution_id,
            self._execution_id,
            graph_node_id=self._graph_node_id,
        )


class ExecutionLineageChildAdmissionHook:
    """Durable child admission executed before delegate."""

    __slots__ = (
        "_persistence",
        "_scope",
        "_segment_root_execution_id",
        "_execution_id",
        "_parent_execution_id",
        "_graph_node_id",
    )

    def __init__(
        self,
        *,
        persistence: ExecutionLineagePersistence,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId,
        graph_node_id: str | None = None,
    ) -> None:
        self._persistence = persistence
        self._scope = scope
        self._segment_root_execution_id = segment_root_execution_id
        self._execution_id = execution_id
        self._parent_execution_id = parent_execution_id
        self._graph_node_id = graph_node_id

    async def admit(self, request: object) -> None:
        del request
        try:
            self._persistence.admit_child(
                self._scope,
                self._segment_root_execution_id,
                self._execution_id,
                self._parent_execution_id,
                graph_node_id=self._graph_node_id,
            )
        except ExecutionLineageUnavailableError:
            try:
                self._persistence.mark_degraded(
                    self._scope, "child_admission_unavailable"
                )
            except ExecutionLineageUnavailableError:
                raise
            bind_attempt_lineage_degradation(True)
        except ExecutionLineageIntegrityError:
            raise


def build_child_lineage_admission_hook(
    *,
    persistence: ExecutionLineagePersistence,
    scope: ExecutionLineageAttemptScope,
    segment_root_execution_id: ExecutionId,
    execution_id: ExecutionId,
    parent_execution_id: ExecutionId,
) -> ExecutionLineageChildAdmissionHook:
    return ExecutionLineageChildAdmissionHook(
        persistence=persistence,
        scope=scope,
        segment_root_execution_id=segment_root_execution_id,
        execution_id=execution_id,
        parent_execution_id=parent_execution_id,
    )
