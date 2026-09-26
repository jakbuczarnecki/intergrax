# © Artur Czarnecki. All rights reserved.

"""EBH-2H — Memory contract boundary mechanical gates."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MEMORY_ROOT = _REPO_ROOT / "intergrax" / "memory"
_COMPAT_PROFILE_MODULE = "intergrax.integrations.registry.profile"
_CANONICAL_PROFILE = "intergrax.integrations.contracts.integration_profile.IntegrationProfile"
_CANONICAL_LLM = "intergrax.llm_adapters.contracts.llm_adapter.LLMAdapter"
_BASE_EMBEDDING = "intergrax.rag.embedding.contracts.base_embedding_manager.BaseEmbeddingManager"
_BASE_VECTORSTORE = (
    "intergrax.rag.vectorstore.contracts.base_vectorstore_manager.BaseVectorstoreManager"
)

_MEMORY_COMPOSITION_FILES = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "memory_wiring.py",
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "memory_vector_wiring.py",
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "memory_control_wiring.py",
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "specialized_memory_wiring.py",
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "entity_graph_wiring.py",
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "entity_user_profile_memory_projection.py",
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "memory_provider_admission.py",
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "memory_observability_wiring.py",
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "memory_security_governance_wiring.py",
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "session_turn_index_rag_adapters.py",
    _REPO_ROOT / "intergrax" / "memory" / "resolver" / "materialization.py",
)

_MEMORY_RAG_COMPOSITION_FILES = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "memory_vector_wiring.py",
)

_MEMORY_VECTOR_WIRING = _REPO_ROOT / "intergrax" / "applications" / "_shared" / "memory_vector_wiring.py"
_USER_PROFILE_MANAGER = _REPO_ROOT / "intergrax" / "memory" / "user_profile_manager.py"
_SESSION_TURN_INDEX_SERVICE = _REPO_ROOT / "intergrax" / "memory" / "session_turn_index_service.py"
_SESSION_TURN_INDEX_CONTRACTS = (
    _REPO_ROOT / "intergrax" / "memory" / "contracts" / "session_turn_index.py"
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


def _memory_production_py_files() -> list[Path]:
    return sorted(path for path in _MEMORY_ROOT.rglob("*.py") if path.is_file())


def test_no_compat_integration_profile_imports_in_memory_scope() -> None:
    violations: list[str] = []
    for path in _MEMORY_COMPOSITION_FILES:
        tree = _parse(path)
        if _COMPAT_PROFILE_MODULE in _import_module_names(tree):
            violations.append(str(path.relative_to(_REPO_ROOT)))
    assert not violations, (
        "compatibility IntegrationProfile import forbidden in Memory EBH-2H scope: "
        + ", ".join(violations)
    )


def _function_param_annotations(path: Path, name: str) -> dict[str, str]:
    tree = _parse(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            params: dict[str, str] = {}
            for arg in node.args.args:
                if arg.arg != "self" and arg.annotation is not None:
                    params[arg.arg] = ast.unparse(arg.annotation)
            for arg in node.args.kwonlyargs:
                if arg.annotation is not None:
                    params[arg.arg] = ast.unparse(arg.annotation)
            return params
    raise AssertionError(f"function {name!r} not found in {path}")


def test_resolve_rag_stack_for_memory_wiring_typed_seam() -> None:
    params = _function_param_annotations(
        _MEMORY_VECTOR_WIRING,
        "resolve_rag_stack_for_memory_wiring",
    )
    assert params.get("integration_profile") == "IntegrationProfile | None"
    assert params.get("llm_adapter") == "LLMAdapter | None"


def test_memory_rag_composition_files_have_no_type_ignore() -> None:
    violations: list[str] = []
    for path in _MEMORY_RAG_COMPOSITION_FILES:
        if "# type: ignore" in _read(path):
            violations.append(str(path.relative_to(_REPO_ROOT)))
    assert not violations, "type: ignore forbidden on Memory↔RAG composition: " + ", ".join(
        violations
    )


def test_build_user_profile_manager_has_no_dict_object_pseudo_contract() -> None:
    tree = _parse(_MEMORY_VECTOR_WIRING)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_user_profile_manager":
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    if isinstance(child.value, ast.Dict):
                        for key in child.value.keys:
                            if key is None:
                                continue
                            if ast.unparse(key) == "kwargs" or (
                                isinstance(child.targets[0], ast.Name)
                                and child.targets[0].id == "kwargs"
                            ):
                                pytest.fail("build_user_profile_manager must not use kwargs dict")
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    if child.target.id == "kwargs" and child.annotation is not None:
                        assert ast.unparse(child.annotation) != "dict[str, object]"
            return
    pytest.fail("build_user_profile_manager not found")


def test_user_profile_manager_depends_on_manager_contracts() -> None:
    tree = _parse(_USER_PROFILE_MANAGER)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "UserProfileManager":
            init = next(
                (m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"),
                None,
            )
            assert init is not None
            param_types: dict[str, str] = {}
            for arg in init.args.kwonlyargs:
                if arg.annotation is not None:
                    param_types[arg.arg] = ast.unparse(arg.annotation)
            embedding_ann = param_types.get("embedding_manager", "")
            vectorstore_ann = param_types.get("vectorstore_manager", "")
            assert "BaseEmbeddingManager" in embedding_ann
            assert "BaseVectorstoreManager" in vectorstore_ann
            assert "rag.embedding.embedding_manager.EmbeddingManager" not in embedding_ann
            assert "rag.vectorstore.vectorstore_manager.VectorstoreManager" not in vectorstore_ann
            return
    pytest.fail("UserProfileManager.__init__ not found")


def test_canonical_memory_control_plane_remains_semantic_plane() -> None:
    from intergrax.memory.contracts.memory_control import MemoryControlPlane
    from intergrax.memory.default_memory_control_plane import DefaultMemoryControlPlane

    assert issubclass(DefaultMemoryControlPlane, MemoryControlPlane)


def test_memory_domain_does_not_import_applications() -> None:
    violations: list[str] = []
    for path in _memory_production_py_files():
        tree = _parse(path)
        for module in _import_module_names(tree):
            if module.startswith("intergrax.applications"):
                violations.append(f"{path.relative_to(_REPO_ROOT)} -> {module}")
    assert not violations, "intergrax/memory must not import applications: " + ", ".join(
        violations
    )


def test_projection_contracts_do_not_import_tier3_composition() -> None:
    contracts_root = _MEMORY_ROOT / "contracts"
    violations: list[str] = []
    for path in sorted(contracts_root.rglob("*projection*.py")):
        tree = _parse(path)
        for module in _import_module_names(tree):
            if module.startswith("intergrax.applications"):
                violations.append(f"{path.relative_to(_REPO_ROOT)} -> {module}")
    assert not violations, (
        "projection contracts must not import Tier-3 composition: " + ", ".join(violations)
    )


def test_resolve_rag_stack_runtime_annotations_match_canonical_types() -> None:
    from intergrax.applications._shared import memory_vector_wiring as mod

    sig = inspect.signature(mod.resolve_rag_stack_for_memory_wiring)
    assert sig.parameters["integration_profile"].annotation is not inspect._empty
    assert sig.parameters["llm_adapter"].annotation is not inspect._empty


def _protocol_class_property_names(tree: ast.Module, class_name: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            props: set[str] = set()
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                if len(item.decorator_list):
                    if any(
                        isinstance(dec, ast.Name) and dec.id == "property"
                        for dec in item.decorator_list
                    ):
                        props.add(item.name)
            return props
    raise AssertionError(f"class {class_name!r} not found")


def test_session_turn_index_vector_contracts_expose_readonly_properties() -> None:
    tree = _parse(_SESSION_TURN_INDEX_CONTRACTS)
    scope_props = _protocol_class_property_names(tree, "SessionTurnIndexVectorScope")
    assert scope_props == {"tenant_id", "namespace", "workspace_id"}
    filter_props = _protocol_class_property_names(tree, "SessionTurnIndexMetadataFilter")
    assert filter_props == {"conditions"}
    upsert_props = _protocol_class_property_names(tree, "SessionTurnIndexVectorUpsertRecord")
    assert upsert_props == {
        "vector_id",
        "document_content",
        "document_metadata",
        "embedding",
    }


def test_session_turn_index_service_has_no_type_ignore_or_cast_workaround() -> None:
    source = _read(_SESSION_TURN_INDEX_SERVICE)
    assert "# type: ignore" not in source
    assert "cast(" not in source


def test_session_turn_index_service_uses_validated_message_role() -> None:
    source = _read(_SESSION_TURN_INDEX_SERVICE)
    assert "_normalize_message_role" in source
    assert 'role=str(meta.get("role")' not in source.replace(" ", "")


def test_session_turn_index_service_depends_on_memory_ports_not_concrete_rag() -> None:
    tree = _parse(_SESSION_TURN_INDEX_SERVICE)
    modules = _import_module_names(tree)
    rag_concrete = (
        "intergrax.rag.embedding.embedding_manager",
        "intergrax.rag.vectorstore.vectorstore_manager",
    )
    violations = [m for m in modules if m in rag_concrete]
    assert not violations, (
        "session turn index service must not import concrete RAG managers: "
        + ", ".join(violations)
    )
    assert "intergrax.memory.contracts.session_turn_index" in modules
