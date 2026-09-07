# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for MP-3G execution lineage integration."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CW_PACKAGE = _REPO_ROOT / "intergrax" / "collaborative_work"
_ARTIFACT_SERVICE_PATH = _CW_PACKAGE / "artifact_service.py"
_FORBIDDEN_IMPORTS = (
    "intergrax.proofs.receipts",
    "intergrax.runtime.execution_evidence",
    "intergrax.contracts.execution_evidence.receipt",
    "intergrax.runtime.nexus",
)
_FORBIDDEN_MUTATION_PATTERNS = (
    re.compile(r"def\s+attach_execution\s*\("),
    re.compile(r"def\s+set_execution\s*\("),
    re.compile(r"def\s+update_version_provenance\s*\("),
    re.compile(r"def\s+patch_version\s*\("),
    re.compile(r"def\s+link_execution_to_version\s*\("),
)
_PRODUCTION_FILES = tuple(
    sorted(
        path
        for path in _CW_PACKAGE.glob("*.py")
        if path.name != "__init__.py"
    )
)


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


def test_mp3g_collaborative_work_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_FILES:
        for lineno, module in _collect_imports(path):
            if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORTS):
                violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp3g_artifact_service_has_no_post_publication_mutation_api() -> None:
    source = _ARTIFACT_SERVICE_PATH.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_MUTATION_PATTERNS:
        assert not pattern.search(source), f"forbidden pattern {pattern.pattern}"


def test_mp3g_artifact_service_exposes_execution_publication_methods() -> None:
    source = _ARTIFACT_SERVICE_PATH.read_text(encoding="utf-8")
    assert "def create_artifact_from_execution" in source
    assert "def publish_version_from_execution" in source
    assert "def create_artifact(" in source
    assert "def publish_version(" in source


def test_mp3g_artifact_service_human_path_passes_execution_none() -> None:
    source = _ARTIFACT_SERVICE_PATH.read_text(encoding="utf-8")
    assert "return self._create_artifact(request, execution=None)" in source
    assert "return self._publish_version(request, execution=None)" in source


def test_mp3g_artifact_service_does_not_auto_create_work_item_execution_link() -> None:
    source = _ARTIFACT_SERVICE_PATH.read_text(encoding="utf-8")
    assert "WorkItemExecutionLink" not in source
    assert "execution_link" not in source


def test_mp3g_artifact_service_has_no_mp3f_content_storage_coupling() -> None:
    source = _ARTIFACT_SERVICE_PATH.read_text(encoding="utf-8")
    assert "ObjectStorageArtifactContentStore" not in source
    assert "content_storage" not in source
