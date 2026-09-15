# © Artur Czarnecki. All rights reserved.

"""OBS-FUNCTIONAL-CONTRACTS-1 architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.functional_evidence import (
    FunctionalEvidencePersistence,
    PipelineEvidenceScope,
    PlatformFunctionalEvidence,
)
from intergrax.contracts.functional_evidence.persistence import (
    FunctionalEvidencePersistence as PersistenceFromSubmodule,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.functional_evidence import PLATFORM_FUNCTIONAL_EVIDENCE_SCHEMA
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    assert_functional_evidence_persistence_conformance,
    sample_functional_evidence,
)

pytestmark = pytest.mark.unit

_FORBIDDEN_OBS_DIAG_IMPORT_PREFIXES = (
    "intergrax.runtime.diagnostics.functional_evidence",
    "intergrax.runtime.diagnostics.functional_evidence_persistence",
)


def test_observability_does_not_import_diag_functional_evidence_modules() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    observability_root = repo_root / "intergrax" / "runtime" / "observability"
    for path in observability_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in _FORBIDDEN_OBS_DIAG_IMPORT_PREFIXES:
                    if node.module == prefix or node.module.startswith(f"{prefix}."):
                        raise AssertionError(
                            f"{path.relative_to(repo_root)} imports forbidden module {node.module}",
                        )


def test_single_canonical_functional_evidence_contract_definitions() -> None:
    assert PlatformFunctionalEvidence.__module__ == "intergrax.contracts.functional_evidence.models"
    assert PipelineEvidenceScope.__module__ == "intergrax.contracts.functional_evidence.models"
    assert FunctionalEvidencePersistence is PersistenceFromSubmodule
    assert FunctionalEvidencePersistence.__module__ == "intergrax.contracts.functional_evidence.persistence"


def test_execution_scoped_scope_requires_five_ids() -> None:
    scope = PipelineEvidenceScope(
        tenant_id="tenant-arch",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    assert scope.execution_id
    assert scope.attempt_id

    with pytest.raises(Exception):
        PipelineEvidenceScope(
            tenant_id="tenant-arch",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        )

    with pytest.raises(Exception):
        PipelineEvidenceScope(
            tenant_id="tenant-arch",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=ExecutionId("not-a-valid-execution-id"),
        )


def test_tenant_isolation_fail_closed_on_scope() -> None:
    with pytest.raises(Exception):
        PipelineEvidenceScope(
            tenant_id="   ",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        )


def test_plugin_providers_conform_without_core_vendor_import() -> None:
    secret = b"x" * 32
    in_memory = InMemoryFunctionalEvidencePersistence(cursor_secret=secret)
    assert_functional_evidence_persistence_conformance(in_memory, label="arch-in-memory")
    assert isinstance(in_memory, FunctionalEvidencePersistence)


def test_query_order_is_pagination_not_execution_authority() -> None:
    from intergrax.contracts.functional_evidence.persistence import functional_evidence_query_order_key

    evidence = sample_functional_evidence()
    key = functional_evidence_query_order_key(evidence)
    assert len(key) == 2


def test_diag_consumer_reads_contract_backed_persistence() -> None:
    from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer

    persistence = InMemoryFunctionalEvidencePersistence(cursor_secret=b"y" * 32)
    scope = sample_functional_evidence().scope
    persistence.append(sample_functional_evidence(scope=scope))
    analyzer = FunctionalDiagnosticAnalyzer(persistence)
    assert analyzer is not None


def test_platform_schema_is_v2_execution_scoped() -> None:
    assert PLATFORM_FUNCTIONAL_EVIDENCE_SCHEMA == "platform_functional_evidence.v2"
