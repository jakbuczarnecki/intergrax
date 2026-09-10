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


def test_data_pack_load_adapters_do_not_import_vector_or_model_providers() -> None:
    adapter_root = _VPI_ROOT / "storage_bootstrap/adapters/postgresql"
    forbidden = frozenset(
        {
            "qdrant",
            "pgvector",
            "torch",
            "sentence_transformers",
            "transformers",
        }
    )
    violations: list[str] = []
    for module_path in sorted(adapter_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            root = imported.split(".")[0]
            if root in forbidden:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_pgvector_adapter_does_not_import_model_providers() -> None:
    adapter_root = _VPI_ROOT / "storage_bootstrap/adapters/pgvector"
    forbidden = frozenset(
        {
            "qdrant",
            "qdrant_client",
            "torch",
            "sentence_transformers",
            "transformers",
        }
    )
    violations: list[str] = []
    for module_path in sorted(adapter_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            root = imported.split(".")[0]
            if root in forbidden:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_pgvector_adapter_does_not_import_qdrant() -> None:
    adapter_root = _VPI_ROOT / "storage_bootstrap/adapters/pgvector"
    violations: list[str] = []
    for module_path in sorted(adapter_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported.split(".")[0] in {"qdrant", "qdrant_client"}:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_qdrant_adapter_does_not_import_pgvector() -> None:
    adapter_root = _VPI_ROOT / "storage_bootstrap/adapters/qdrant"
    violations: list[str] = []
    for module_path in sorted(adapter_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported.split(".")[0] == "pgvector":
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_data_pack_load_core_has_no_provider_imports() -> None:
    core_root = _VPI_ROOT / "storage_bootstrap/data_pack_load"
    forbidden = frozenset(
        {
            "psycopg",
            "asyncpg",
            "sqlalchemy",
            "mysql",
            "qdrant",
            "pgvector",
            "torch",
            "sentence_transformers",
            "transformers",
        }
    )
    violations: list[str] = []
    for module_path in sorted(core_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            root = imported.split(".")[0]
            if root in forbidden:
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
        if "qualification" in path.parts:
            continue
        for imported in _module_imports(path):
            if imported == forbidden:
                violations.append(str(path.relative_to(_REPO_ROOT)))
    assert violations == []


def test_retrieval_orchestration_has_no_provider_imports() -> None:
    retrieval_root = _VPI_ROOT / "application/retrieval"
    forbidden = frozenset(
        {
            "psycopg",
            "asyncpg",
            "sqlalchemy",
            "mysql",
            "qdrant",
            "qdrant_client",
            "pgvector",
            "torch",
            "sentence_transformers",
            "transformers",
        }
    )
    violations: list[str] = []
    for module_path in sorted(retrieval_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            root = imported.split(".")[0]
            if root in forbidden:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_retrieval_orchestration_depends_only_on_application_ports() -> None:
    retrieval_root = _VPI_ROOT / "application/retrieval"
    violations: list[str] = []
    for module_path in sorted(retrieval_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported.startswith(
                "platform_proofs.scenarios.verified_product_identification.integrations"
            ):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_no_reflection_in_retrieval_orchestration_production_code() -> None:
    retrieval_root = _VPI_ROOT / "application/retrieval"
    forbidden_names = {"getattr", "setattr", "hasattr", "inspect"}
    for path in _iter_production_python_files(retrieval_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert forbidden_names.isdisjoint(names), f"forbidden reflection in {path}"


def test_no_weak_contracts_in_retrieval_orchestration_production_code() -> None:
    retrieval_root = _VPI_ROOT / "application/retrieval"
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "Mapping[str, object]",
        ": object",
        "type: ignore",
    )
    for path in _iter_production_python_files(retrieval_root):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {path}"


def test_fusion_layer_has_no_dataset_or_infrastructure_imports() -> None:
    fusion_root = _VPI_ROOT / "application/fusion"
    forbidden_fragments = (
        ".dataset.",
        ".data_pack.",
        ".storage_bootstrap.",
        ".integrations.providers.",
    )
    violations: list[str] = []
    for module_path in sorted(fusion_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if any(fragment in imported for fragment in forbidden_fragments):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_fusion_layer_has_no_provider_imports() -> None:
    fusion_root = _VPI_ROOT / "application/fusion"
    forbidden = frozenset(
        {
            "psycopg",
            "asyncpg",
            "sqlalchemy",
            "mysql",
            "qdrant",
            "qdrant_client",
            "pgvector",
            "torch",
            "sentence_transformers",
            "transformers",
        }
    )
    violations: list[str] = []
    for module_path in sorted(fusion_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            root = imported.split(".")[0]
            if root in forbidden:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_fusion_layer_has_no_integrations_imports() -> None:
    fusion_root = _VPI_ROOT / "application/fusion"
    violations: list[str] = []
    for module_path in sorted(fusion_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported.startswith(
                "platform_proofs.scenarios.verified_product_identification.integrations"
            ):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_fusion_layer_has_no_cluster_id_usage() -> None:
    fusion_root = _VPI_ROOT / "application/fusion"
    for path in _iter_production_python_files(fusion_root):
        source = path.read_text(encoding="utf-8")
        assert "cluster_id" not in source, f"cluster_id found in {path}"


def test_fusion_layer_has_no_reflection() -> None:
    fusion_root = _VPI_ROOT / "application/fusion"
    forbidden_names = {"getattr", "setattr", "hasattr", "inspect"}
    for path in _iter_production_python_files(fusion_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert forbidden_names.isdisjoint(names), f"forbidden reflection in {path}"


def test_fusion_layer_has_no_weak_contracts() -> None:
    fusion_root = _VPI_ROOT / "application/fusion"
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "Mapping[str, object]",
        ": object",
        "type: ignore",
    )
    for path in _iter_production_python_files(fusion_root):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {path}"


def test_identity_layer_has_no_dataset_or_infrastructure_imports() -> None:
    identity_root = _VPI_ROOT / "application/identity"
    forbidden_fragments = (
        ".dataset.",
        ".data_pack.",
        ".storage_bootstrap.",
        ".integrations.providers.",
    )
    violations: list[str] = []
    for module_path in sorted(identity_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if any(fragment in imported for fragment in forbidden_fragments):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_identity_layer_has_no_provider_imports() -> None:
    identity_root = _VPI_ROOT / "application/identity"
    forbidden = frozenset(
        {
            "psycopg",
            "asyncpg",
            "sqlalchemy",
            "mysql",
            "qdrant",
            "qdrant_client",
            "pgvector",
            "torch",
            "sentence_transformers",
            "transformers",
        }
    )
    violations: list[str] = []
    for module_path in sorted(identity_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            root = imported.split(".")[0]
            if root in forbidden:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_identity_layer_has_no_integrations_imports() -> None:
    identity_root = _VPI_ROOT / "application/identity"
    violations: list[str] = []
    for module_path in sorted(identity_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported.startswith(
                "platform_proofs.scenarios.verified_product_identification.integrations"
            ):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_identity_layer_has_no_cluster_id_usage() -> None:
    identity_root = _VPI_ROOT / "application/identity"
    for path in _iter_production_python_files(identity_root):
        source = path.read_text(encoding="utf-8")
        assert "cluster_id" not in source, f"cluster_id found in {path}"


def test_identity_layer_has_no_reflection() -> None:
    identity_root = _VPI_ROOT / "application/identity"
    forbidden_names = {"getattr", "setattr", "hasattr", "inspect"}
    for path in _iter_production_python_files(identity_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert forbidden_names.isdisjoint(names), f"forbidden reflection in {path}"


def test_identity_layer_has_no_weak_contracts() -> None:
    identity_root = _VPI_ROOT / "application/identity"
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "Mapping[str, object]",
        ": object",
        "type: ignore",
    )
    for path in _iter_production_python_files(identity_root):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {path}"


def test_identity_evaluation_layer_has_no_dataset_or_infrastructure_imports() -> None:
    evaluation_root = _VPI_ROOT / "application/identity_evaluation"
    forbidden_fragments = (
        ".dataset.",
        ".data_pack.",
        ".storage_bootstrap.",
        ".integrations.providers.",
    )
    violations: list[str] = []
    for module_path in sorted(evaluation_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if any(fragment in imported for fragment in forbidden_fragments):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_identity_evaluation_layer_has_no_provider_imports() -> None:
    evaluation_root = _VPI_ROOT / "application/identity_evaluation"
    forbidden = frozenset(
        {
            "psycopg",
            "asyncpg",
            "sqlalchemy",
            "mysql",
            "qdrant",
            "qdrant_client",
            "pgvector",
            "torch",
            "sentence_transformers",
            "transformers",
        }
    )
    violations: list[str] = []
    for module_path in sorted(evaluation_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            root = imported.split(".")[0]
            if root in forbidden:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_identity_evaluation_layer_has_no_integrations_imports() -> None:
    evaluation_root = _VPI_ROOT / "application/identity_evaluation"
    violations: list[str] = []
    for module_path in sorted(evaluation_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported.startswith(
                "platform_proofs.scenarios.verified_product_identification.integrations"
            ):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_identity_evaluation_layer_has_no_cluster_id_usage() -> None:
    evaluation_root = _VPI_ROOT / "application/identity_evaluation"
    for path in _iter_production_python_files(evaluation_root):
        source = path.read_text(encoding="utf-8")
        assert "cluster_id" not in source, f"cluster_id found in {path}"


def test_identity_evaluation_layer_has_no_reflection() -> None:
    evaluation_root = _VPI_ROOT / "application/identity_evaluation"
    forbidden_names = {"getattr", "setattr", "hasattr", "inspect"}
    for path in _iter_production_python_files(evaluation_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert forbidden_names.isdisjoint(names), f"forbidden reflection in {path}"


def test_identity_evaluation_layer_has_no_weak_contracts() -> None:
    evaluation_root = _VPI_ROOT / "application/identity_evaluation"
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "Mapping[str, object]",
        ": object",
        "type: ignore",
    )
    for path in _iter_production_python_files(evaluation_root):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {path}"


def test_identity_evaluation_layer_has_no_terminal_verdict_strings() -> None:
    evaluation_root = _VPI_ROOT / "application/identity_evaluation"
    forbidden = (
        "VERIFIED",
        "AMBIGUOUS",
        "INSUFFICIENT_INFORMATION",
        "NO_MATCH",
    )
    for path in _iter_production_python_files(evaluation_root):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{token} found in {path}"


def test_verification_layer_has_no_dataset_or_infrastructure_imports() -> None:
    verification_root = _VPI_ROOT / "application/verification"
    forbidden_fragments = (
        ".dataset.",
        ".data_pack.",
        ".storage_bootstrap.",
        ".integrations.providers.",
    )
    violations: list[str] = []
    for module_path in sorted(verification_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if any(fragment in imported for fragment in forbidden_fragments):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_verification_layer_has_no_provider_imports() -> None:
    verification_root = _VPI_ROOT / "application/verification"
    forbidden = frozenset(
        {
            "psycopg",
            "asyncpg",
            "sqlalchemy",
            "mysql",
            "qdrant",
            "qdrant_client",
            "pgvector",
            "torch",
            "sentence_transformers",
            "transformers",
        }
    )
    violations: list[str] = []
    for module_path in sorted(verification_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            root = imported.split(".")[0]
            if root in forbidden:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_verification_layer_has_no_integrations_imports() -> None:
    verification_root = _VPI_ROOT / "application/verification"
    violations: list[str] = []
    for module_path in sorted(verification_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported.startswith(
                "platform_proofs.scenarios.verified_product_identification.integrations"
            ):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_verification_layer_has_no_cluster_id_usage() -> None:
    verification_root = _VPI_ROOT / "application/verification"
    for path in _iter_production_python_files(verification_root):
        source = path.read_text(encoding="utf-8")
        assert "cluster_id" not in source, f"cluster_id found in {path}"


def test_verification_layer_has_no_reflection() -> None:
    verification_root = _VPI_ROOT / "application/verification"
    forbidden_names = {"getattr", "setattr", "hasattr", "inspect"}
    for path in _iter_production_python_files(verification_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert forbidden_names.isdisjoint(names), f"forbidden reflection in {path}"


def test_verification_layer_has_no_weak_contracts() -> None:
    verification_root = _VPI_ROOT / "application/verification"
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "Mapping[str, object]",
        ": object",
        "type: ignore",
    )
    for path in _iter_production_python_files(verification_root):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {path}"


def test_fusion_and_identity_layers_have_no_terminal_verdict_strings() -> None:
    for layer in ("fusion", "identity"):
        layer_root = _VPI_ROOT / "application" / layer
        forbidden = (
            "ProductIdentificationOutcome",
            "INSUFFICIENT_INFORMATION",
        )
        for path in _iter_production_python_files(layer_root):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in source, f"{token} found in {path}"


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
