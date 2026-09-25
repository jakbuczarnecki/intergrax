# © Artur Czarnecki. All rights reserved.

"""EBH-2G-R1 — RAG retrieval composition and candidate ABI mechanical gates."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RESOLVE = _REPO_ROOT / "intergrax" / "rag" / "retrieval" / "resolve.py"
_RETRIEVAL_SERVICE = _REPO_ROOT / "intergrax" / "rag" / "retrieval" / "retrieval_service.py"
_BASE_RETRIEVER_MANAGER = (
    _REPO_ROOT / "intergrax" / "rag" / "retrievers" / "contracts" / "base_retriever_manager.py"
)
_RETRIEVER_MANAGER = _REPO_ROOT / "intergrax" / "rag" / "retrievers" / "retriever_manager.py"
_CANONICAL_METADATA_FILTER_ANNOTATION = "MetadataFilter | None"
_FORBIDDEN_CAPABILITY_GETATTR_ATTRS = frozenset({"supports_scoped_retrieval", "last_execution"})


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


def _forbidden_capability_getattr_calls(tree: ast.Module) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
            continue
        if len(node.args) < 2:
            continue
        second = node.args[1]
        if isinstance(second, ast.Constant) and isinstance(second.value, str):
            if second.value in _FORBIDDEN_CAPABILITY_GETATTR_ATTRS:
                violations.append(second.value)
    return violations


def _retrieve_metadata_filter_annotation(tree: ast.Module, class_name: str) -> str | None:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "retrieve":
                continue
            for arg in item.args.kwonlyargs:
                if arg.arg == "metadata_filter":
                    if arg.annotation is None:
                        return None
                    return ast.unparse(arg.annotation)
    return None


def _class_property_names(tree: ast.Module, class_name: str) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                for dec in item.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "property":
                        names.add(item.name)
    return names


def test_retrieval_service_has_no_dynamic_capability_probing() -> None:
    tree = _parse(_RETRIEVAL_SERVICE)
    violations = _forbidden_capability_getattr_calls(tree)
    assert violations == []
    source = _read(_RETRIEVAL_SERVICE)
    assert "hasattr(" not in source
    assert ".supports_scoped_retrieval" in source
    assert ".last_execution" in source


def test_retrieve_metadata_filter_is_canonical_typed_on_manager_boundaries() -> None:
    for path, class_name in (
        (_BASE_RETRIEVER_MANAGER, "BaseRetrieverManager"),
        (_RETRIEVER_MANAGER, "RetrieverManager"),
    ):
        ann = _retrieve_metadata_filter_annotation(_parse(path), class_name)
        assert ann == _CANONICAL_METADATA_FILTER_ANNOTATION, (
            f"{class_name}.retrieve.metadata_filter must be {_CANONICAL_METADATA_FILTER_ANNOTATION}, got {ann!r}"
        )


def test_base_retriever_manager_defines_capability_contract_properties() -> None:
    tree = _parse(_BASE_RETRIEVER_MANAGER)
    props = _class_property_names(tree, "BaseRetrieverManager")
    assert "supports_scoped_retrieval" in props
    assert "last_execution" in props


_RETRIEVAL_SECURITY = _REPO_ROOT / "intergrax" / "runtime" / "architecture" / "retrieval_security.py"
_RAG_POISONING_CALL = _REPO_ROOT / "intergrax" / "tools" / "providers" / "rag" / "service.py"


def test_retrieval_poisoning_protocol_uses_read_only_properties() -> None:
    props = _class_property_names(_parse(_RETRIEVAL_SECURITY), "RetrievalPoisoningInputChunk")
    assert props == {"id", "text", "score", "source_ref"}


def test_rag_poisoning_filter_uses_structural_wiring_without_cast() -> None:
    source = _read(_RAG_POISONING_CALL)
    assert "filter_retrieved_chunks_for_poisoning" in source
    assert "cast(" not in source
    assert "type: ignore" not in source


def test_rag_guard_gate_tests_collect_without_rag_local_embeddings() -> None:
    """RAG guard CI profile (dev-ci-rag) must not require optional HF stack at collection."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/rag/",
            "tests/unit/tools/providers/rag/",
            "-m",
            "gate",
            "--collect-only",
            "-q",
            "--tb=line",
            "-p",
            "no:xdist",
        ],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
