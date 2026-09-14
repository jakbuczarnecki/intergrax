# © Artur Czarnecki. All rights reserved.

"""Immutable catalog profile compilation results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from testing_support.execution_qualification.contracts import QualificationSuite
from testing_support.execution_qualification.graph_contracts import (
    QualificationExecutionPlan,
    QualificationGraphDefinition,
)


@dataclass(frozen=True, slots=True)
class CompiledCatalogProfile:
    graph: QualificationGraphDefinition
    plan: QualificationExecutionPlan
    suite_by_id: Mapping[str, QualificationSuite]
    shared_suite_ids: tuple[str, ...] = ()
