# © Artur Czarnecki. All rights reserved.

"""Typed contracts for canonical qualification DAG definitions and compiled plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from testing_support.execution_qualification.contracts import (
    QualificationRunManifest,
    QualificationSuite,
)

_SAFE_NODE_ID_RE = re.compile(r"^[\w.-]+$")


class QualificationNodeKind(StrEnum):
    LEAF_SUITE = "LEAF_SUITE"
    AGGREGATE_GATE = "AGGREGATE_GATE"
    STATIC_GATE = "STATIC_GATE"


@dataclass(frozen=True, slots=True)
class QualificationGateDefinition:
    """Aggregate or static gate — no pytest/subprocess configuration."""

    gate_id: str
    requires: tuple[str, ...]
    mandatory: bool = True
    declaration_index: int = 0
    description: str | None = None
    gate_kind: QualificationNodeKind = QualificationNodeKind.AGGREGATE_GATE

    def __post_init__(self) -> None:
        if not self.gate_id or not _SAFE_NODE_ID_RE.fullmatch(self.gate_id):
            raise ValueError(f"invalid gate_id: {self.gate_id!r}")
        if self.declaration_index < 0:
            raise ValueError("declaration_index must be non-negative")
        if self.gate_kind not in (
            QualificationNodeKind.AGGREGATE_GATE,
            QualificationNodeKind.STATIC_GATE,
        ):
            raise ValueError("gate_kind must be AGGREGATE_GATE or STATIC_GATE")


@dataclass(frozen=True, slots=True)
class QualificationProfile:
    profile_id: str
    root_gate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.profile_id or not _SAFE_NODE_ID_RE.fullmatch(self.profile_id):
            raise ValueError(f"invalid profile_id: {self.profile_id!r}")


@dataclass(frozen=True, slots=True)
class QualificationGraphDefinition:
    """Explicit graph inputs: manifest suites + gates + profile catalog."""

    run_manifest: QualificationRunManifest
    gates: tuple[QualificationGateDefinition, ...]
    profiles: tuple[QualificationProfile, ...] = ()


@dataclass(frozen=True, slots=True)
class QualificationExecutionNode:
    node_id: str
    kind: QualificationNodeKind
    dependencies: tuple[str, ...]
    declaration_index: int
    suite: QualificationSuite | None = None
    gate: QualificationGateDefinition | None = None

    def __post_init__(self) -> None:
        if self.kind is QualificationNodeKind.LEAF_SUITE:
            if self.suite is None or self.gate is not None:
                raise ValueError("LEAF_SUITE requires suite and no gate")
        else:
            if self.gate is None or self.suite is not None:
                raise ValueError("gate nodes require gate and no suite")


@dataclass(frozen=True, slots=True)
class QualificationExecutionPlan:
    profile_id: str
    ordered_nodes: tuple[QualificationExecutionNode, ...]
    leaf_suite_ids: tuple[str, ...]
    root_gate_ids: tuple[str, ...]
    dependency_edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")


class QualificationGraphError(ValueError):
    """Base class for graph definition / compilation failures."""


class QualificationManifestConflictError(QualificationGraphError):
    """Duplicate or conflicting manifest node definitions."""


class ConflictingQualificationSuiteDefinitionError(QualificationManifestConflictError):
    """Same suite_id with differing executable definitions."""


class MissingQualificationDependencyError(QualificationGraphError):
    """Required node id does not exist in the manifest or gate catalog."""


class QualificationDependencyCycleError(QualificationGraphError):
    """Cycle detected in the qualification dependency graph."""

    def __init__(self, cycle_path: tuple[str, ...]) -> None:
        self.cycle_path = cycle_path
        path = " -> ".join(cycle_path)
        super().__init__(f"qualification dependency cycle: {path}")


class QualificationProfileError(QualificationGraphError):
    """Invalid or inconsistent qualification profile."""
