# © Artur Czarnecki. All rights reserved.

"""EBH-2G-R2-R2 — RAG composition typing and contract purity mechanical gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RAG_ROOT = _REPO_ROOT / "intergrax" / "rag"
_COMPAT_PROFILE_MODULE = "intergrax.integrations.registry.profile"
_CANONICAL_PROFILE_MODULE = "intergrax.integrations.contracts.integration_profile"
_INTEGRATION_VECTORSTORE = (
    _RAG_ROOT / "vectorstore" / "bootstrap" / "integration_vectorstore.py"
)
_EMBEDDING_RESOLVER = _RAG_ROOT / "embedding" / "runtime" / "resolver.py"
_QUERY_REFINER = _RAG_ROOT / "retrieval" / "query_refiner.py"
_GRAPH_INDEXER_FACTORY = _RAG_ROOT / "graph" / "indexer" / "graph_indexer_factory.py"
_RAG_PROFILE = _RAG_ROOT / "profiles" / "rag_profile.py"
_RERANK_RESOLVER = _RAG_ROOT / "rerankers" / "integration" / "resolver.py"
_API_RERANKER_BASE = _RAG_ROOT / "rerankers" / "providers" / "_api_reranker_base.py"
_COHERE_RERANKER = _RAG_ROOT / "rerankers" / "providers" / "cohere_reranker.py"
_JINA_RERANKER = _RAG_ROOT / "rerankers" / "providers" / "jina_reranker.py"
_RERANKERS_ROOT = _RAG_ROOT / "rerankers"
_CANONICAL_RERANK_PROVIDER = (
    "intergrax.integrations.contracts.rerank_provider"
)
_GATE_TARGETS = (
    _INTEGRATION_VECTORSTORE,
    _EMBEDDING_RESOLVER,
    _QUERY_REFINER,
    _GRAPH_INDEXER_FACTORY,
    _RAG_PROFILE,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path))


def _import_module_names(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _rag_production_py_files() -> list[Path]:
    return sorted(path for path in _RAG_ROOT.rglob("*.py") if path.is_file())


def test_no_compat_integration_profile_imports_under_rag() -> None:
    violations: list[str] = []
    for path in _rag_production_py_files():
        tree = _parse(path)
        if _COMPAT_PROFILE_MODULE in _import_module_names(tree):
            rel = path.relative_to(_REPO_ROOT)
            violations.append(str(rel))
    assert not violations, (
        "compatibility IntegrationProfile import forbidden under intergrax/rag: "
        + ", ".join(violations)
    )


def test_integration_vectorstore_has_no_type_ignore() -> None:
    source = _read(_INTEGRATION_VECTORSTORE)
    assert "# type: ignore" not in source


def _function_param_annotations(path: Path, name: str) -> dict[str, str]:
    tree = _parse(path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            out: dict[str, str] = {}
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg == "self" or arg.annotation is None:
                    continue
                out[arg.arg] = ast.unparse(arg.annotation)
            return out
    raise AssertionError(f"{name} not found in {path}")


def test_vectorstore_slug_params_are_str_not_object() -> None:
    for fn_name in ("_profile_tenant_id", "_scope_from_integration_config"):
        params = _function_param_annotations(_INTEGRATION_VECTORSTORE, fn_name)
        assert "slug" in params
        assert params["slug"] == "str | None"
        assert "object" not in params["slug"]


def test_embedding_contract_spec_helpers_are_typed() -> None:
    tree = _parse(_EMBEDDING_RESOLVER)
    spec_fn: ast.FunctionDef | None = None
    binder_fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_embedding_contract_spec":
            spec_fn = node
        if isinstance(node, ast.FunctionDef) and node.name == "_embedding_runtime_binder":
            binder_fn = node
    assert spec_fn is not None
    assert binder_fn is not None
    assert spec_fn.returns is not None
    assert ast.unparse(spec_fn.returns) == "IntegrationContractSpec"
    binder_params = {
        arg.arg: ast.unparse(arg.annotation) if arg.annotation else None
        for arg in binder_fn.args.args
    }
    assert binder_params.get("spec") == "IntegrationContractSpec"


def test_query_refiner_does_not_define_agentic_query_mode() -> None:
    tree = _parse(_QUERY_REFINER)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AgenticQueryMode":
                    pytest.fail("AgenticQueryMode must be owned only by rag_profile.py")


def test_graph_indexer_factory_does_not_define_graph_indexer_mode() -> None:
    tree = _parse(_GRAPH_INDEXER_FACTORY)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GraphIndexerMode":
                    pytest.fail("GraphIndexerMode must be owned only by rag_profile.py")


def test_gate_targets_have_no_type_ignore() -> None:
    for path in _GATE_TARGETS:
        assert "# type: ignore" not in _read(path), f"type: ignore forbidden in {path.name}"


def test_canonical_mode_aliases_only_in_rag_profile() -> None:
    for alias in ("AgenticQueryMode", "GraphIndexerMode"):
        owners: list[str] = []
        for path in _rag_production_py_files():
            tree = _parse(path)
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == alias:
                        owners.append(str(path.relative_to(_REPO_ROOT)))
        expected = (_RAG_ROOT / "profiles" / "rag_profile.py").resolve()
        normalized = {Path(owner).resolve() for owner in owners}
        assert normalized == {expected}, (
            f"{alias} must be defined only in rag_profile.py, found: {owners}"
        )


def test_rag_profile_env_helpers_exist() -> None:
    tree = _parse(_RAG_PROFILE)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_env_")
    }
    assert "_env_query_expansion_mode" in names
    assert "_env_graph_indexer_mode" in names
    assert "_env_agentic_query_mode" in names


def _function_return_annotation(path: Path, name: str) -> str | None:
    tree = _parse(path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            if node.returns is None:
                return None
            return ast.unparse(node.returns)
    raise AssertionError(f"{name} not found in {path}")


def _method_return_annotation(path: Path, class_name: str, method_name: str) -> str | None:
    tree = _parse(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    if item.returns is None:
                        return None
                    return ast.unparse(item.returns)
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _class_method_uses_name(path: Path, class_name: str, method_name: str, used_name: str) -> bool:
    tree = _parse(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Name) and sub.id == used_name:
                            return True
                    return False
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _module_defines_class_named(path: Path, class_name: str) -> bool:
    tree = _parse(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return True
    return False


def test_rerank_resolver_is_typed_without_dynamic_probing() -> None:
    source = _read(_RERANK_RESOLVER)
    assert "hasattr" not in source
    assert "getattr" not in source
    assert "# type: ignore" not in source
    assert _function_return_annotation(_RERANK_RESOLVER, "resolve_rerank_provider") == "RerankProvider"
    modules = _import_module_names(_parse(_RERANK_RESOLVER))
    assert _CANONICAL_RERANK_PROVIDER in modules


def test_rerank_scores_removed_from_rag_rerankers() -> None:
    violations: list[str] = []
    for path in sorted(_RERANKERS_ROOT.rglob("*.py")):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "rerank_scores":
                violations.append(str(path.relative_to(_REPO_ROOT)))
    assert not violations, "rerank_scores must not exist under intergrax/rag/rerankers: " + ", ".join(
        violations
    )


def test_api_reranker_base_delegates_to_canonical_rerank_provider() -> None:
    modules = _import_module_names(_parse(_API_RERANKER_BASE))
    assert _CANONICAL_RERANK_PROVIDER in modules
    assert (
        _method_return_annotation(_API_RERANKER_BASE, "_APIRerankerBase", "_resolve_provider")
        == "RerankProvider"
    )
    assert _class_method_uses_name(_API_RERANKER_BASE, "_APIRerankerBase", "rerank", "provider")


def test_validate_rerank_provider_results_preserves_typed_semantic_boundary() -> None:
    params = _function_param_annotations(
        _API_RERANKER_BASE,
        "_validate_rerank_provider_results",
    )
    assert params.get("results") == "Sequence[RerankerResult]"
    assert params.get("results") != "object"
    assert (
        _function_return_annotation(_API_RERANKER_BASE, "_validate_rerank_provider_results")
        == "tuple[RerankerResult, ...]"
    )


def test_cohere_and_jina_wrappers_avoid_concrete_integration_providers() -> None:
    forbidden_prefix = "intergrax.integrations.providers.rerank_provider"
    for path in (_COHERE_RERANKER, _JINA_RERANKER):
        modules = _import_module_names(_parse(path))
        violations = [module for module in modules if module.startswith(forbidden_prefix)]
        assert not violations, f"{path.name} must not import concrete integration providers: {violations}"


def test_rag_rerankers_do_not_duplicate_rerank_provider_protocol() -> None:
    owners: list[str] = []
    for path in sorted(_RERANKERS_ROOT.rglob("*.py")):
        if _module_defines_class_named(path, "RerankProvider"):
            owners.append(str(path.relative_to(_REPO_ROOT)))
    assert not owners, "RerankProvider Protocol must not be redefined under RAG rerankers: " + ", ".join(
        owners
    )
