# © Artur Czarnecki. All rights reserved.

"""Harness for DIAG R5 extension SPI qualification (R5-A1–A7)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.runtime.diagnostics.diagnostic_extension_evidence_store import (
    InMemoryDiagnosticExtensionEvidenceStore,
)
from intergrax.runtime.diagnostics.diagnostic_extension_registry import (
    DiagnosticExtensionRegistry,
)
from intergrax.runtime.diagnostics.diagnostic_extension_service import (
    DiagnosticExtensionService,
)
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from testing_support.runtime.execution_failure_evidence_r2_closure_harness import (
    ExecutionFailureEvidenceClosureHarness,
    build_execution_failure_evidence_r2_closure_harness,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    read_service_for_tests,
)

_DEFAULT_NAMESPACE = "qualification.r5"


@dataclass(slots=True)
class DiagnosticExtensionSpiR5Harness:
    execution: ExecutionFailureEvidenceClosureHarness
    evidence_store: InMemoryDiagnosticExtensionEvidenceStore
    registry: DiagnosticExtensionRegistry
    extension_service: DiagnosticExtensionService

    @property
    def tenant_id(self) -> str:
        return self.execution.tenant_id

    @property
    def read_service(self) -> DiagnosticReadService:
        return self.execution.read_service


def build_diagnostic_extension_spi_r5_harness(
    *,
    tenant_id: str | None = None,
    registry: DiagnosticExtensionRegistry | None = None,
) -> DiagnosticExtensionSpiR5Harness:
    execution = build_execution_failure_evidence_r2_closure_harness(tenant_id=tenant_id)
    store = InMemoryDiagnosticExtensionEvidenceStore()
    resolved_registry = registry or DiagnosticExtensionRegistry.empty()
    service = DiagnosticExtensionService(
        registry=resolved_registry,
        evidence_store=store,
    )
    execution.read_service = read_service_for_tests(
        execution.problem_persistence,
        execution.execution_reconstructor,
        occurrence_persistence=execution.occurrence_persistence,
        extension_service=service,
    )
    return DiagnosticExtensionSpiR5Harness(
        execution=execution,
        evidence_store=store,
        registry=resolved_registry,
        extension_service=service,
    )


__all__ = [
    "DiagnosticExtensionSpiR5Harness",
    "_DEFAULT_NAMESPACE",
    "build_diagnostic_extension_spi_r5_harness",
]
