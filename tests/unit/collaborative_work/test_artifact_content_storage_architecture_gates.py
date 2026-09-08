# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for MP-3F artifact content storage."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

from intergrax.collaborative_work.content_storage import (
    ArtifactContentStore,
    ObjectStorageArtifactContentStore,
)
from intergrax.integrations.contracts.object_storage import ConditionalObjectStorage

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTENT_STORAGE_PATH = _REPO_ROOT / "intergrax" / "collaborative_work" / "content_storage.py"
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.integrations.providers.object_storage.s3",
    "intergrax.integrations.providers.object_storage.gcs",
    "intergrax.integrations.providers.object_storage.azure",
    "intergrax.integrations.providers.object_storage.minio",
    "intergrax.integrations.providers.object_storage.filesystem",
    "intergrax.collaborative_work.postgresql_repository",
    "intergrax.collaborative_work.sqlite_repository",
    "intergrax.runtime.nexus",
    "intergrax.contracts.proof_receipt",
    "intergrax.contracts.decision",
    "applications.lkw",
)
_FORBIDDEN_AST_NAMES = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "vars",
        "Any",
    },
)
_FORBIDDEN_ATTRIBUTE_NAMES = frozenset({"__dict__"})


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _collect_forbidden_ast_usage(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_AST_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden name {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTE_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden name {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "__setattr__" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "object":
                    violations.append(
                        f"{path.name}:{node.lineno} references forbidden object.__setattr__",
                    )
    return violations


def test_mp3f_content_storage_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(_CONTENT_STORAGE_PATH):
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORT_PREFIXES):
            violations.append(f"{_CONTENT_STORAGE_PATH.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp3f_content_storage_has_no_forbidden_dynamic_patterns() -> None:
    violations = _collect_forbidden_ast_usage(_CONTENT_STORAGE_PATH)
    assert not violations, "\n".join(violations)


def test_mp3f_port_has_no_delete_or_presign() -> None:
    source = inspect.getsource(ArtifactContentStore)
    lowered = source.lower()
    assert "delete" not in lowered
    assert "presign" not in lowered
    assert "list" not in lowered


def test_mp3f_adapter_has_no_delete_or_presign_surface() -> None:
    public_names = {
        name
        for name in dir(ObjectStorageArtifactContentStore)
        if not name.startswith("_")
    }
    assert "delete" not in public_names
    assert "presigned_url" not in public_names
    assert "close" not in public_names


def test_mp3f_module_does_not_import_metadata_repository() -> None:
    source = _CONTENT_STORAGE_PATH.read_text(encoding="utf-8")
    assert "artifact_service" not in source
    assert "repository" not in source
    assert "persistence" not in source
    assert "ExecutionProvenanceRef" not in source
    assert "ProofReceipt" not in source


def test_mp3f_adapter_requires_conditional_object_storage() -> None:
    hints = get_type_hints(ObjectStorageArtifactContentStore.__init__)
    assert hints["object_storage"] is ConditionalObjectStorage


def test_mp3f_adapter_put_does_not_call_overwrite_capable_put() -> None:
    source = inspect.getsource(ObjectStorageArtifactContentStore.put)
    assert "put_if_absent" in source
    assert "self._object_storage.put(" not in source.replace("put_if_absent", "")


def test_mp3f_adapter_put_does_not_use_threading_lock() -> None:
    source = _CONTENT_STORAGE_PATH.read_text(encoding="utf-8")
    assert "threading" not in source
    assert "Lock" not in source


def test_mp3f_module_does_not_define_plugin_registry() -> None:
    source = _CONTENT_STORAGE_PATH.read_text(encoding="utf-8")
    assert "ArtifactStoragePluginRegistry" not in source
    assert "PluginRegistry" not in source
