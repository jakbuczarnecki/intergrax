# © Artur Czarnecki. All rights reserved.

"""OBS-FUNCTIONAL-CONTRACTS-1 / R1 architecture gates."""

from __future__ import annotations

import ast
import inspect
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
from intergrax.runtime.observability.functional_evidence.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.observability.functional_evidence.functional_evidence_persistence_conformance import (
    assert_functional_evidence_persistence_conformance,
    sample_functional_evidence,
)
from intergrax.runtime.observability.functional_evidence.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OBSERVABILITY_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "observability"
_FE_SUBSYSTEM_ROOT = _OBSERVABILITY_ROOT / "functional_evidence"
_CONTRACTS_FE_ROOT = _REPO_ROOT / "intergrax" / "contracts" / "functional_evidence"
_DIAGNOSTICS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _import_modules_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def test_observability_functional_evidence_subsystem_does_not_import_runtime_diagnostics() -> None:
    forbidden_prefix = "intergrax.runtime.diagnostics"
    targets = [
        _OBSERVABILITY_ROOT / "functional_evidence_runtime_wiring.py",
        _OBSERVABILITY_ROOT / "functional_evidence_recorder.py",
        *_iter_python_files(_FE_SUBSYSTEM_ROOT),
    ]
    for path in targets:
        for module in _import_modules_in_file(path):
            if module == forbidden_prefix or module.startswith(f"{forbidden_prefix}."):
                raise AssertionError(
                    f"{path.relative_to(_REPO_ROOT)} imports forbidden module {module}",
                )


def test_contracts_functional_evidence_do_not_import_runtime() -> None:
    forbidden_prefix = "intergrax.runtime"
    for path in _iter_python_files(_CONTRACTS_FE_ROOT):
        for module in _import_modules_in_file(path):
            if module == forbidden_prefix or module.startswith(f"{forbidden_prefix}."):
                raise AssertionError(
                    f"{path.relative_to(_REPO_ROOT)} imports forbidden module {module}",
                )


def test_canonical_functional_evidence_providers_not_defined_under_diagnostics() -> None:
    forbidden_class_names = {
        "InMemoryFunctionalEvidencePersistence",
        "DocumentStoreFunctionalEvidencePersistence",
    }
    for path in _iter_python_files(_DIAGNOSTICS_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in forbidden_class_names:
                raise AssertionError(
                    f"{path.relative_to(_REPO_ROOT)} defines forbidden provider {node.name}",
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


def test_document_store_provider_conformance() -> None:
    from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
    from intergrax.integrations._shared.conformance import assert_conditional_document_store

    store = assert_conditional_document_store(InMemoryDocumentStore())
    persistence = DocumentStoreFunctionalEvidencePersistence(store, cursor_secret=b"z" * 32)
    assert_functional_evidence_persistence_conformance(persistence, label="arch-document-store")


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


def test_functional_evidence_runtime_wiring_has_no_getattr_discovery() -> None:
    from intergrax.runtime.observability import functional_evidence_runtime_wiring as wiring_mod

    source = inspect.getsource(wiring_mod)
    assert "getattr(" not in source
    assert "hasattr(" not in source
    assert "setattr(" not in source
