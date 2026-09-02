"""Permanent VPI production vendor-import architecture gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_VPI_ROOT = _REPO_ROOT / "platform_proofs/scenarios/verified_product_identification"

_PRODUCTION_PATHS = (
    _VPI_ROOT / "application",
    _VPI_ROOT / "storage_bootstrap",
    _VPI_ROOT / "integrations",
    _VPI_ROOT / "composition",
    _VPI_ROOT / "ingest",
    _VPI_ROOT / "bootstrap.py",
)

_FORBIDDEN_VENDOR_ROOTS = frozenset(
    {
        "qdrant_client",
        "psycopg",
        "asyncpg",
        "openai",
        "sentence_transformers",
        "ollama",
        "weaviate",
        "chromadb",
        "pgvector",
        "torch",
        "vllm",
        "llama_cpp",
    }
)

_FORBIDDEN_VECTOR_PROVIDER_PREFIX = "intergrax.integrations.providers.vector_store.qdrant"

_PROVIDER_IMPORT_ALLOWED_PREFIXES = (
    f"{_VPI_ROOT / 'composition'}".replace("\\", "/"),
    f"{_VPI_ROOT / 'integrations/catalog_store/postgresql'}".replace("\\", "/"),
)


def _iter_production_python_files() -> list[Path]:
    files: list[Path] = []
    for path in _PRODUCTION_PATHS:
        if path.is_file():
            files.append(path)
            continue
        if not path.exists():
            continue
        files.extend(sorted(path.rglob("*.py")))
    return files


def _imported_modules(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _imported_roots(module_path: Path) -> set[str]:
    return {module.split(".", 1)[0].casefold() for module in _imported_modules(module_path)}


def _provider_import_allowed(module_path: Path) -> bool:
    normalized = module_path.as_posix()
    return any(
        normalized.startswith(prefix)
        for prefix in _PROVIDER_IMPORT_ALLOWED_PREFIXES
    )


def test_vpi_production_has_no_direct_vendor_sdk_imports() -> None:
    violations: list[str] = []
    for module_path in _iter_production_python_files():
        forbidden = sorted(_FORBIDDEN_VENDOR_ROOTS & _imported_roots(module_path))
        if forbidden:
            rel = module_path.relative_to(_REPO_ROOT).as_posix()
            violations.append(f"{rel}: {', '.join(forbidden)}")
    assert violations == []


def test_vpi_production_outside_composition_has_no_qdrant_provider_imports() -> None:
    violations: list[str] = []
    for module_path in _iter_production_python_files():
        if _provider_import_allowed(module_path):
            continue
        for imported in _imported_modules(module_path):
            if imported.startswith(_FORBIDDEN_VECTOR_PROVIDER_PREFIX):
                rel = module_path.relative_to(_REPO_ROOT).as_posix()
                violations.append(f"{rel}: {imported}")
    assert violations == []


def test_composition_may_import_public_qdrant_plugin_api_only() -> None:
    composition_path = _VPI_ROOT / "composition" / "bootstrap_runtime.py"
    modules = _imported_modules(composition_path)
    qdrant_imports = sorted(
        module for module in modules if module.startswith(_FORBIDDEN_VECTOR_PROVIDER_PREFIX)
    )
    assert qdrant_imports == [
        "intergrax.integrations.providers.vector_store.qdrant.config",
        "intergrax.integrations.providers.vector_store.qdrant.opens",
    ]


def test_platform_search_adapter_has_no_qdrant_provider_import() -> None:
    adapter_path = _VPI_ROOT / "integrations/search_store/platform_bootstrap_adapter.py"
    source = adapter_path.read_text(encoding="utf-8")
    assert "qdrant_client" not in source
    assert "QdrantClient" not in source
    assert "PointStruct" not in source
    assert "integrations.providers.vector_store.qdrant" not in source
