# © Artur Czarnecki. All rights reserved.

"""Typed coordinator port for qualification plan execution."""

from __future__ import annotations

from typing import Protocol

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationRunResult,
    QualificationRunConfig,
    QualificationRunManifest,
)


class QualificationCoordinatorPort(Protocol):
    def run(
        self,
        manifest: QualificationRunManifest,
        config: QualificationRunConfig,
    ) -> ExecutionQualificationRunResult:
        """Execute manifest suites and return typed run receipts."""
        ...
