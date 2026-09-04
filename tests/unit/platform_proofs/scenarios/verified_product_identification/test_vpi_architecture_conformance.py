"""VPI dependency-direction architecture conformance gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_VPI_ROOT = _REPO_ROOT / "platform_proofs/scenarios/verified_product_identification"
_EMBEDDING_MATERIALIZATION_ROOT = _VPI_ROOT / "embedding_materialization"


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_application_domain_does_not_import_integrations() -> None:
    domain_root = _VPI_ROOT / "application/domain"
    violations: list[str] = []
    for module_path in sorted(domain_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported.startswith("platform_proofs.scenarios.verified_product_identification.integrations"):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_orchestration_depends_only_on_ports() -> None:
    orchestrator_paths = [
        _VPI_ROOT / "storage_bootstrap/orchestration/orchestrator.py",
        _VPI_ROOT / "embedding_materialization/orchestration/orchestrator.py",
    ]
    for orchestrator_path in orchestrator_paths:
        imports = _module_imports(orchestrator_path)
        forbidden = sorted(
            imported
            for imported in imports
            if imported.startswith(
                "platform_proofs.scenarios.verified_product_identification.integrations"
            )
        )
        assert forbidden == [], f"{orchestrator_path.name} imports integrations"


def test_storage_bootstrap_contracts_have_no_provider_paths() -> None:
    contracts_root = _VPI_ROOT / "storage_bootstrap/contracts"
    violations: list[str] = []
    for module_path in sorted(contracts_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if ".integrations." in imported:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def _iter_production_python_files(root: Path):
    for path in root.rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        yield path


def test_storage_orchestrator_has_no_embedding_execution_dependency() -> None:
    orchestrator_path = _VPI_ROOT / "storage_bootstrap/orchestration/orchestrator.py"
    source = orchestrator_path.read_text(encoding="utf-8")
    assert "EmbeddingExecutionPort" not in source
    assert "embed_batch" not in source
    assert "IntergraxEmbeddingBootstrapAdapter" not in source


def test_storage_composition_has_no_embedding_provider() -> None:
    composition_path = _VPI_ROOT / "composition/bootstrap_runtime.py"
    source = composition_path.read_text(encoding="utf-8")
    assert "IntergraxEmbeddingBootstrapAdapter" not in source
    assert "ParquetFilesystemArtifactReader" in source


def test_storage_orchestrator_has_no_parquet_or_vendor_imports() -> None:
    orchestrator_path = _VPI_ROOT / "storage_bootstrap/orchestration/orchestrator.py"
    imports = _module_imports(orchestrator_path)
    forbidden = sorted(
        imported
        for imported in imports
        if any(
            fragment in imported
            for fragment in (
                "pyarrow",
                "qdrant",
                "psycopg",
                "sentence_transformers",
                "torch",
                "integrations.embedding",
                "stores.parquet",
            )
        )
    )
    assert forbidden == []


def test_no_reflection_in_embedding_materialization_production_code() -> None:
    forbidden_names = {"getattr", "setattr", "hasattr", "inspect"}
    for path in _iter_production_python_files(_EMBEDDING_MATERIALIZATION_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert forbidden_names.isdisjoint(names), f"forbidden reflection in {path}"


def test_no_torch_in_embedding_materialization_orchestrator() -> None:
    orchestrator_path = _EMBEDDING_MATERIALIZATION_ROOT / "orchestration/orchestrator.py"
    imports = _module_imports(orchestrator_path)
    forbidden = sorted(
        imported
        for imported in imports
        if imported in {"torch", "sentence_transformers"}
    )
    assert forbidden == []


def test_qualification_contracts_have_no_torch_imports() -> None:
    contracts_root = _VPI_ROOT / "qualification/contracts"
    violations: list[str] = []
    for module_path in sorted(contracts_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported in {"torch", "sentence_transformers", "qdrant_client", "psycopg"}:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_qualification_core_has_no_torch_imports() -> None:
    core_modules = (
        _VPI_ROOT / "qualification/batch_selection.py",
        _VPI_ROOT / "qualification/bottleneck.py",
        _VPI_ROOT / "qualification/duration_estimate.py",
        _VPI_ROOT / "qualification/reporting.py",
        _VPI_ROOT / "qualification/runner.py",
        _VPI_ROOT / "qualification/text_length_profile.py",
    )
    violations: list[str] = []
    for module_path in core_modules:
        for imported in _module_imports(module_path):
            if imported in {"torch", "sentence_transformers"}:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_production_code_has_no_concrete_hf_embedding_provider_imports() -> None:
    forbidden = "intergrax.rag.embedding.providers.hf_embedding_provider"
    violations: list[str] = []
    for path in _iter_production_python_files(_VPI_ROOT):
        for imported in _module_imports(path):
            if imported == forbidden:
                violations.append(str(path.relative_to(_REPO_ROOT)))
    assert violations == []


def test_no_weak_contracts_in_embedding_materialization_production_code() -> None:
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "Mapping[str, object]",
        ": object",
        "type: ignore",
    )
    for path in _iter_production_python_files(_EMBEDDING_MATERIALIZATION_ROOT):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {path}"
