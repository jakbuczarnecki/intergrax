# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.maintenance.check_knowledge_document_conformance import (
    REPO_ROOT,
    audit_repository,
    main,
    scan_ast_boundary,
)

pytestmark = pytest.mark.gate

CHECKER = REPO_ROOT / "scripts" / "maintenance" / "check_knowledge_document_conformance.py"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_knowledge_tree(tmp_path: Path) -> Path:
    (tmp_path / "intergrax" / "knowledge").mkdir(parents=True)
    return tmp_path


def test_ast_passes_clean_file(tmp_path: Path) -> None:
    repo = _make_knowledge_tree(tmp_path)
    _write(
        repo / "intergrax" / "knowledge" / "clean.py",
        "from intergrax.knowledge.contracts import KnowledgeDocument\n",
    )
    assert scan_ast_boundary(repo) == []


def test_ast_detects_direct_langchain_core_import(tmp_path: Path) -> None:
    repo = _make_knowledge_tree(tmp_path)
    _write(
        repo / "intergrax" / "knowledge" / "bad.py",
        "import langchain_core\n",
    )
    diagnostics = scan_ast_boundary(repo)
    assert any("KNOWLEDGE_DOCUMENT_AST_VIOLATION" in item for item in diagnostics)
    assert any("langchain_core" in item for item in diagnostics)


def test_ast_detects_from_langchain_core_import(tmp_path: Path) -> None:
    repo = _make_knowledge_tree(tmp_path)
    _write(
        repo / "intergrax" / "knowledge" / "bad.py",
        "from langchain_core.documents import Document\n",
    )
    diagnostics = scan_ast_boundary(repo)
    assert any(
        "from-import 'langchain_core.documents'" in item for item in diagnostics
    )


def test_ast_detects_intergrax_compat_import(tmp_path: Path) -> None:
    repo = _make_knowledge_tree(tmp_path)
    _write(
        repo / "intergrax" / "knowledge" / "bad.py",
        "from intergrax.compat.langchain import to_langchain_document\n",
    )
    diagnostics = scan_ast_boundary(repo)
    assert any("intergrax.compat.langchain" in item for item in diagnostics)


def test_ast_detects_literal_importlib_import_module(tmp_path: Path) -> None:
    repo = _make_knowledge_tree(tmp_path)
    _write(
        repo / "intergrax" / "knowledge" / "bad.py",
        "import importlib\nimportlib.import_module('langchain_core.documents')\n",
    )
    diagnostics = scan_ast_boundary(repo)
    assert any("importlib.import_module('langchain_core.documents')" in item for item in diagnostics)


def test_ast_detects_literal_dunder_import(tmp_path: Path) -> None:
    repo = _make_knowledge_tree(tmp_path)
    _write(
        repo / "intergrax" / "knowledge" / "bad.py",
        '__import__("langchain_core.documents")\n',
    )
    diagnostics = scan_ast_boundary(repo)
    assert any("__import__('langchain_core.documents')" in item for item in diagnostics)


def test_ast_ignores_plain_string_and_comment(tmp_path: Path) -> None:
    repo = _make_knowledge_tree(tmp_path)
    _write(
        repo / "intergrax" / "knowledge" / "clean.py",
        '# from langchain_core.documents import Document\n'
        'message = "from langchain_core.documents import Document"\n',
    )
    assert scan_ast_boundary(repo) == []


def test_isolated_runtime_proof_passes() -> None:
    violations = audit_repository(REPO_ROOT)
    isolated = [
        item
        for item in violations
        if item.startswith("KNOWLEDGE_DOCUMENT_ISOLATED_IMPORT_FAILURE")
    ]
    assert isolated == []


def test_main_passes_for_current_repo() -> None:
    proc = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Knowledge document conformance: OK" in proc.stdout
    assert main() == 0


def test_negative_proof_ast_rejects_langchain_import(tmp_path: Path) -> None:
    repo = _make_knowledge_tree(tmp_path)
    _write(
        repo / "intergrax" / "knowledge" / "bad.py",
        "from langchain_core.documents import Document\n",
    )
    diagnostics = scan_ast_boundary(repo)
    assert any(item.startswith("KNOWLEDGE_DOCUMENT_AST_VIOLATION:") for item in diagnostics)
