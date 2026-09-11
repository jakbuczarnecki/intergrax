# © Artur Czarnecki. All rights reserved.

"""Contract tests for diagnostic extension SPI (R5)."""

from __future__ import annotations

import pytest

from intergrax.contracts.diagnostic_extension_evidence import (
    DiagnosticEvidenceScope,
    DiagnosticExtensionEvidence,
    validate_extension_evidence_tenant_scope,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.diagnostic_extension_registry import (
    DiagnosticExtensionConfigurationError,
    DiagnosticExtensionRegistry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _scope(tenant_id: str = "tenant-a") -> DiagnosticEvidenceScope:
    return DiagnosticEvidenceScope(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_extension_evidence_requires_namespaced_kind() -> None:
    with pytest.raises(ValueError):
        DiagnosticExtensionEvidence.mint(
            scope=_scope(),
            evidence_namespace="qualification.r5",
            kind="not_namespaced",
            summary="x",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_validate_extension_evidence_tenant_scope_rejects_cross_tenant() -> None:
    evidence = DiagnosticExtensionEvidence.mint(
        scope=_scope(tenant_id="tenant-a"),
        evidence_namespace="qualification.r5",
        kind="qualification.r5.fact",
        summary="ok",
    )
    with pytest.raises(ValueError):
        validate_extension_evidence_tenant_scope(evidence, tenant_id="tenant-b")


@pytest.mark.unit
@pytest.mark.gate
def test_registry_orders_by_priority_namespace_then_id() -> None:
    class _A:
        contributor_id = "z-id"
        evidence_namespace = "ns.b"
        priority = 1

        def collect(self, context: object) -> tuple[()]:
            return ()

    class _B:
        contributor_id = "a-id"
        evidence_namespace = "ns.a"
        priority = 0

        def collect(self, context: object) -> tuple[()]:
            return ()

    registry = DiagnosticExtensionRegistry(evidence_contributors=(_A(), _B()))
    ordered_ids = tuple(c.contributor_id for c in registry.evidence_contributors)
    assert ordered_ids == ("a-id", "z-id")


@pytest.mark.unit
@pytest.mark.gate
def test_registry_rejects_duplicate_stable_id() -> None:
    class _One:
        contributor_id = "dup"
        evidence_namespace = "ns.a"
        priority = 0

        def collect(self, context: object) -> tuple[()]:
            return ()

    class _Two:
        contributor_id = "dup"
        evidence_namespace = "ns.b"
        priority = 1

        def collect(self, context: object) -> tuple[()]:
            return ()

    with pytest.raises(DiagnosticExtensionConfigurationError):
        DiagnosticExtensionRegistry(evidence_contributors=(_One(), _Two()))
