# © Artur Czarnecki. All rights reserved.

"""EBH-2G-R1 — RAG retrieval composition and candidate ABI mechanical gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RESOLVE = _REPO_ROOT / "intergrax" / "rag" / "retrieval" / "resolve.py"
_RETRIEVAL_SERVICE = _REPO_ROOT / "intergrax" / "rag" / "retrieval" / "retrieval_service.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path))


def test_resolve_retrieval_service_has_no_semantic_any_composition_params() -> None:
    tree = _parse(_RESOLVE)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_retrieval_service"
    )
    for arg in fn.args.args + fn.args.kwonlyargs:
        if arg.annotation is None:
            pytest.fail(f"resolve_retrieval_service parameter {arg.arg} must be annotated")
        ann = ast.unparse(arg.annotation)
        if ann == "Any":
            pytest.fail(f"resolve_retrieval_service parameter {arg.arg} must not use Any")
    source = _read(_RESOLVE)
    assert "vectorstore_manager: Any" not in source
    assert "embedding_manager: Any" not in source
    assert "retriever_manager: Any" not in source
    assert "reranker_manager: Any" not in source


def test_retrieval_service_has_no_implicit_candidate_abi() -> None:
    source = _read(_RETRIEVAL_SERVICE)
    assert "_candidates_to_chunks" not in source
    assert "List[Any]" not in source
    assert "attribute_access.optional(c, " not in source
    assert "retriever must return RetrievalHit candidates" in source


def test_retrieval_service_has_no_vendor_sdk_imports() -> None:
    tree = _parse(_RETRIEVAL_SERVICE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("qdrant")
                assert not alias.name.startswith("openai")
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.integrations.providers")


def test_retrieval_service_imports_resolve_only_from_rag_retrieval() -> None:
    source = _read(_REPO_ROOT / "intergrax" / "tools" / "providers" / "rag" / "service.py")
    assert "resolve_retrieval_service" in source
    assert "from intergrax.rag.retrieval.resolve import resolve_retrieval_service" in source
