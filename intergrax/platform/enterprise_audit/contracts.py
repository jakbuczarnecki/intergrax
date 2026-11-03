# © Artur Czarnecki. All rights reserved.

"""Enterprise audit contracts (ENT-AUDIT-X)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Literal


class AuditSeverity(StrEnum):
    BLOCKER = "BLOCKER"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    TRACKED_DEBT = "TRACKED_DEBT"
    ENVIRONMENT = "ENVIRONMENT"


class AuditCategory(StrEnum):
    ARCHITECTURE = "architecture"
    TENANT = "tenant"
    VBN = "vbn"
    QUALIFICATION = "qualification"
    INTERNAL = "internal"


class DriftStatus(StrEnum):
    NEW = "NEW"
    RESOLVED = "RESOLVED"
    UNCHANGED = "UNCHANGED"
    REGRESSED = "REGRESSED"


class AggregateAuditStatus(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED = "BLOCKED"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


class AuditMode(StrEnum):
    FAST = "FAST"
    FULL = "FULL"
    FREEZE = "FREEZE"
    CHANGED_SINCE = "CHANGED-SINCE"


class EnterpriseAuditExitCode(IntEnum):
    CLEAN = 0
    VIOLATION = 1
    INCOMPLETE = 2
    INTERNAL = 3


EntrypointKind = Literal["pytest_nodeid", "python_script", "pytest_file", "internal_pending"]


@dataclass(frozen=True, slots=True)
class ManifestCheck:
    check_id: str
    entrypoint_kind: EntrypointKind
    entrypoint: str
    mutating: bool = False
    modes: frozenset[AuditMode] = frozenset({AuditMode.FULL, AuditMode.FREEZE})
    severity_default: AuditSeverity = AuditSeverity.ERROR
    category: AuditCategory = AuditCategory.ARCHITECTURE
    pg_id: str = ""
    depends_on: frozenset[str] = frozenset()
    fast_mandatory: bool = False
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class AuditFinding:
    finding_id: str
    check_id: str
    severity: AuditSeverity
    category: AuditCategory
    message: str
    source_path: str = ""
    frz_mapping: tuple[str, ...] = ()
    drift: DriftStatus | None = None


@dataclass(frozen=True, slots=True)
class CheckRunOutcome:
    check_id: str
    executed: bool
    skipped: bool
    skip_reason: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    findings: tuple[AuditFinding, ...] = ()
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass(frozen=True, slots=True)
class AuditExecutionManifest:
    schema_version: Literal[1] = 1
    audit_mode: AuditMode = AuditMode.FAST
    qualification_sha: str = ""
    audit_execution_sha: str = ""
    repo_remote: str = ""
    branch: str = ""
    checks_planned: int = 0
    checks_executed: int = 0
    checks_skipped: int = 0
    coverage_label: str = "PARTIAL"
    certifying_run: bool = True
    run_status: Literal["completed", "interrupted"] = "completed"
    check_manifest_digest: str = ""
    outcomes: tuple[CheckRunOutcome, ...] = ()


@dataclass(frozen=True, slots=True)
class AuditReport:
    status: AggregateAuditStatus
    exit_code: EnterpriseAuditExitCode
    audit_mode: AuditMode
    current_sha: str
    baseline_sha: str | None
    findings: tuple[AuditFinding, ...] = ()
    limitations: tuple[str, ...] = field(default_factory=tuple)
    coverage_matrix_summary: dict[str, int] = field(default_factory=dict)
    certifying_run: bool = True
