# © Artur Czarnecki. All rights reserved.

"""Immutable canonical qualification catalog."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from testing_support.execution_qualification.catalog.contracts import (
    CompiledCatalogProfile,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    PROFILE_BUILDERS,
)
from testing_support.execution_qualification.catalog.validation import (
    validate_compiled_profile,
)
from testing_support.execution_qualification.graph_contracts import (
    QualificationExecutionPlan,
)


@dataclass(frozen=True, slots=True)
class QualificationCatalog:
    """Explicit composition of canonical qualification profiles (no dynamic registry)."""

    profile_ids: tuple[str, ...]

    def compile_profile(self, profile_id: str) -> CompiledCatalogProfile:
        if profile_id not in self.profile_ids:
            raise KeyError(f"unknown profile_id: {profile_id!r}")
        builder: Callable[[], CompiledCatalogProfile] = PROFILE_BUILDERS[profile_id]
        compiled = builder()
        validate_compiled_profile(compiled)
        return compiled

    def compile_execution_plan(self, profile_id: str) -> QualificationExecutionPlan:
        return self.compile_profile(profile_id).plan
