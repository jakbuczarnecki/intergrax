# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable execution lineage admission contracts (DG-001 R1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

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

_SCHEMA_VERSION = 1
_MAX_LINEAGE_PAGE_LIMIT = 1000
_MAX_ADMISSION_PAGE_LIMIT = _MAX_LINEAGE_PAGE_LIMIT


class ExecutionLineageSegmentLifecycle(StrEnum):
    SEGMENT_OPEN = "SEGMENT_OPEN"
    SEGMENT_CLOSED_CLEAN = "SEGMENT_CLOSED_CLEAN"
    SEGMENT_UNCLEAN = "SEGMENT_UNCLEAN"


class ExecutionLineageAttemptClosureKind(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRY_SUPERSEDED = "RETRY_SUPERSEDED"


class ExecutionLineagePersistenceProvider(StrEnum):
    """Composition-time selector for durable :class:`ExecutionLineagePersistence` wiring."""

    DOCUMENT_STORE = "document_store"


class ExecutionLineageError(RuntimeError):
    """Base error for execution lineage persistence."""


class ExecutionLineageIntegrityError(ExecutionLineageError):
    """Structural lineage conflict — fail closed."""


class ExecutionLineageUnavailableError(ExecutionLineageError):
    """Durable backend unavailable or write failed."""


class ExecutionLineageConfigurationError(ExecutionLineageError):
    """Lineage capability misconfigured — fail closed."""


class ExecutionLineageAttemptScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if self.tenant_id != self.tenant_id.strip():
            raise ValueError("tenant_id must be trimmed")


def build_execution_lineage_attempt_scope(
    *,
    tenant_id: str,
    task_id: TaskId | str,
    run_id: RunId | str,
    attempt_id: AttemptId | str,
) -> ExecutionLineageAttemptScope:
    normalized_tenant = tenant_id.strip()
    if not normalized_tenant:
        raise ValueError("tenant_id must be non-empty")
    return ExecutionLineageAttemptScope(
        tenant_id=normalized_tenant,
        task_id=validate_task_id(task_id),
        run_id=validate_run_id(run_id),
        attempt_id=validate_attempt_id(attempt_id),
    )


class ExecutionLineageAdmissionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scope: ExecutionLineageAttemptScope
    segment_root_execution_id: ExecutionId
    execution_id: ExecutionId
    parent_execution_id: ExecutionId | None
    admission_position: int = Field(ge=1)
    graph_node_id: str | None = None

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        segment_root = validate_execution_id(self.segment_root_execution_id)
        execution = validate_execution_id(self.execution_id)
        parent = (
            validate_execution_id(self.parent_execution_id)
            if self.parent_execution_id is not None
            else None
        )
        if parent is None:
            if execution != segment_root:
                raise ValueError(
                    "root admission requires execution_id == segment_root_execution_id",
                )
        else:
            if execution == parent:
                raise ValueError(
                    "child admission requires execution_id != parent_execution_id"
                )


class ExecutionLineageSegmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scope: ExecutionLineageAttemptScope
    root_execution_id: ExecutionId
    predecessor_root_execution_id: ExecutionId | None
    lifecycle: ExecutionLineageSegmentLifecycle


class ExecutionLineageAttemptState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scope: ExecutionLineageAttemptScope
    generation: int = Field(ge=1)
    next_admission_position: int = Field(ge=1)
    active_segment_root_execution_id: ExecutionId | None = None
    degraded: bool = False
    sealed: bool = False
    closure_kind: ExecutionLineageAttemptClosureKind | None = None
    discovery_contract_version: int | None = None

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        marker = self.discovery_contract_version
        if marker is not None and marker != 1:
            raise ValueError(
                "discovery_contract_version must be None or 1",
            )


class ExecutionLineageRunScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str
    task_id: TaskId
    run_id: RunId

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if self.tenant_id != self.tenant_id.strip():
            raise ValueError("tenant_id must be trimmed")


def build_execution_lineage_run_scope(
    *,
    tenant_id: str,
    task_id: TaskId | str,
    run_id: RunId | str,
) -> ExecutionLineageRunScope:
    normalized_tenant = tenant_id.strip()
    if not normalized_tenant:
        raise ValueError("tenant_id must be non-empty")
    return ExecutionLineageRunScope(
        tenant_id=normalized_tenant,
        task_id=validate_task_id(task_id),
        run_id=validate_run_id(run_id),
    )


class ExecutionLineageDiscoveryCoverageOrigin(StrEnum):
    FROM_RUN_START = "FROM_RUN_START"


class ExecutionLineageAttemptDiscoveryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_scope: ExecutionLineageRunScope
    attempt_id: AttemptId
    discovery_position: int = Field(ge=1)


class ExecutionLineageDiscoveryRunState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_scope: ExecutionLineageRunScope
    generation: int = Field(ge=0)
    next_discovery_position: int = Field(ge=1)
    coverage_contract_version: int | None = None
    coverage_origin: ExecutionLineageDiscoveryCoverageOrigin | None = None

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if (
            self.coverage_origin
            is ExecutionLineageDiscoveryCoverageOrigin.FROM_RUN_START
            and self.coverage_contract_version != 1
        ):
            raise ValueError(
                "coverage_origin FROM_RUN_START requires coverage_contract_version == 1",
            )


class ExecutionLineageAttemptDiscoveryPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempts: tuple[ExecutionLineageAttemptDiscoveryRecord, ...]
    next_cursor: str | None = None


class ExecutionLineageSealRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scope: ExecutionLineageAttemptScope
    closure_kind: ExecutionLineageAttemptClosureKind
    degraded: bool


class ExecutionLineageAdmissionPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    admissions: tuple[ExecutionLineageAdmissionRecord, ...]
    next_cursor: str | None = None


class ExecutionLineageSegmentPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    segments: tuple[ExecutionLineageSegmentRecord, ...]
    next_cursor: str | None = None


def validate_lineage_page_limit(
    limit: int, *, hard_maximum: int = _MAX_LINEAGE_PAGE_LIMIT
) -> int:
    if type(limit) is not int or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive int")
    if limit > hard_maximum:
        raise ValueError(f"limit must be <= {hard_maximum}")
    return limit


def validate_admission_page_limit(
    limit: int, *, hard_maximum: int = _MAX_ADMISSION_PAGE_LIMIT
) -> int:
    return validate_lineage_page_limit(limit, hard_maximum=hard_maximum)


class ExecutionLineageReader(ABC):
    """Read-only execution lineage port for diagnostics and derived projections."""

    @abstractmethod
    def list_admissions_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAdmissionPage:
        """Bounded admission history for one attempt."""

    @abstractmethod
    def list_segments_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageSegmentPage:
        """Bounded segment history for one attempt."""

    @abstractmethod
    def read_attempt_lineage_state(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState | None:
        """Read durable attempt coordination state."""

    @abstractmethod
    def read_seal(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageSealRecord | None:
        """Read durable attempt seal when present."""

    @abstractmethod
    def read_discovery_run_state(
        self,
        run_scope: ExecutionLineageRunScope,
    ) -> ExecutionLineageDiscoveryRunState | None:
        """Read durable run-scoped discovery coordination state."""

    @abstractmethod
    def list_attempts_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAttemptDiscoveryPage:
        """Bounded paginated attempt discovery for one run."""

    @abstractmethod
    def read_attempt_discovery_record(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord | None:
        """Point read for one attempt discovery record within a run."""


class ExecutionLineagePersistence(ExecutionLineageReader, ABC):
    """Provider-neutral durable execution lineage write authority."""

    @property
    @abstractmethod
    def is_durable(self) -> bool:
        """Whether state survives process restart."""

    @abstractmethod
    def open_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        *,
        discovery_contract_version: int | None = None,
    ) -> ExecutionLineageAttemptState:
        """Open attempt coordination state idempotently."""

    @abstractmethod
    def register_attempt_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord:
        """Register one attempt in the run discovery projection."""

    @abstractmethod
    def open_segment(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
        predecessor_root_execution_id: ExecutionId | None = None,
    ) -> ExecutionLineageSegmentRecord:
        """Open one execution segment within an attempt."""

    @abstractmethod
    def admit_root(
        self,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        *,
        graph_node_id: str | None = None,
    ) -> ExecutionLineageAdmissionRecord:
        """Durable root admission for the active segment."""

    @abstractmethod
    def admit_child(
        self,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId,
        *,
        graph_node_id: str | None = None,
    ) -> ExecutionLineageAdmissionRecord:
        """Durable child admission under the active segment."""

    @abstractmethod
    def close_segment_for_resume(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
    ) -> ExecutionLineageSegmentRecord:
        """Mark the current segment clean-closed for pause/resume continuity."""

    @abstractmethod
    def mark_degraded(
        self,
        scope: ExecutionLineageAttemptScope,
        reason_code: str,
    ) -> ExecutionLineageAttemptState:
        """Persist monotonic attempt degradation."""

    @abstractmethod
    def seal_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        closure_kind: ExecutionLineageAttemptClosureKind,
    ) -> ExecutionLineageSealRecord:
        """Seal attempt and close active segment when required."""


__all__ = [
    "ExecutionLineageAdmissionPage",
    "ExecutionLineageAdmissionRecord",
    "ExecutionLineageAttemptClosureKind",
    "ExecutionLineageAttemptDiscoveryPage",
    "ExecutionLineageAttemptDiscoveryRecord",
    "ExecutionLineageAttemptScope",
    "ExecutionLineageAttemptState",
    "ExecutionLineageConfigurationError",
    "ExecutionLineageDiscoveryCoverageOrigin",
    "ExecutionLineageDiscoveryRunState",
    "ExecutionLineageError",
    "ExecutionLineageIntegrityError",
    "ExecutionLineagePersistence",
    "ExecutionLineagePersistenceProvider",
    "ExecutionLineageReader",
    "ExecutionLineageRunScope",
    "ExecutionLineageSealRecord",
    "ExecutionLineageSegmentLifecycle",
    "ExecutionLineageSegmentPage",
    "ExecutionLineageSegmentRecord",
    "ExecutionLineageUnavailableError",
    "build_execution_lineage_attempt_scope",
    "build_execution_lineage_run_scope",
    "validate_admission_page_limit",
    "validate_lineage_page_limit",
]
