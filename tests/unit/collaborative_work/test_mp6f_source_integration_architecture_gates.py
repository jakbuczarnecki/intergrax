# © Artur Czarnecki. All rights reserved.

"""MP-6F — architecture import/layer gates for source integrations."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_DIR = _REPO_ROOT / "intergrax" / "collaborative_work"

_SOURCE_GLOB = (
    "collaborative_activity_source_mapping.py",
    "collaborative_activity_source_adapters.py",
    "collaborative_activity_source_wiring.py",
    "collaborative_activity_source_ports.py",
)

_ADAPTER_AND_WIRING = (
    _SOURCE_DIR / "collaborative_activity_source_adapters.py",
    _SOURCE_DIR / "collaborative_activity_source_wiring.py",
)

_FORBIDDEN_IMPORT_SUBSTRINGS = (
    "collaborative_activity_append_store",
    "collaborative_activity_sqlite",
    "collaborative_activity_postgresql",
    "sqlite_repository",
    "postgresql_repository",
    "CollaborativeActivityAppendIntent",
    "CollaborativeActivityIngestionService",
    "CollaborativeActivityAppendStore",
)

_FORBIDDEN_CONCRETE_SERVICE_IMPORT_MODULES = (
    "intergrax.collaborative_work.service",
    "intergrax.collaborative_work.artifact_service",
    "intergrax.collaborative_work.decision_binding_service",
)

_FORBIDDEN_INNER_ANNOTATION_NAMES = frozenset(
    {
        "CollaborativeWorkService",
        "CollaborativeWorkArtifactService",
        "CollaborativeDecisionBindingService",
        "object",
        "Any",
    },
)

_PLUGINABILITY_TEST = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "collaborative_work"
    / "test_mp6f_collaborative_activity_source_adapters.py"
)


def _source_files() -> list[Path]:
    paths = [_SOURCE_DIR / name for name in _SOURCE_GLOB]
    for path in paths:
        assert path.is_file(), f"missing MP-6F module: {path}"
    return paths


def test_mp6f_source_modules_exist() -> None:
    _source_files()


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_mp6f_no_store_provider_or_append_intent_imports(path: Path) -> None:
    text = path.read_text(encoding="utf-8-sig")
    lowered = text.lower()
    for marker in _FORBIDDEN_IMPORT_SUBSTRINGS:
        assert marker.lower() not in lowered, f"{path.name} must not reference {marker}"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_mp6f_no_repository_module_import(path: Path) -> None:
    text = path.read_text(encoding="utf-8-sig")
    assert "intergrax.collaborative_work.repository" not in text, (
        f"{path.name} must not import repository layer"
    )


def test_mp6f_adapters_depend_on_publication_port_contract() -> None:
    adapters = (_SOURCE_DIR / "collaborative_activity_source_adapters.py").read_text(
        encoding="utf-8-sig",
    )
    assert "from intergrax.contracts.collaborative_activity import" in adapters
    assert "CollaborativeActivityPublicationPort" in adapters


def test_mp6f_no_getattr_hasattr_in_source_integration_modules() -> None:
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"getattr", "hasattr"}, (
                    f"{path.name} must not use reflection for authority mapping"
                )


def test_mp6f_mapping_module_has_no_publication_port_implementation_imports() -> None:
    mapping_path = _SOURCE_DIR / "collaborative_activity_source_mapping.py"
    text = mapping_path.read_text(encoding="utf-8-sig")
    assert "CollaborativeActivityPublicationPort" not in text
    assert "publish(" not in text


@pytest.mark.parametrize("path", _ADAPTER_AND_WIRING, ids=lambda p: p.name)
def test_mp6f_adapters_wiring_no_concrete_source_service_imports(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _FORBIDDEN_CONCRETE_SERVICE_IMPORT_MODULES:
            pytest.fail(f"{path.name} must not import concrete source service module {node.module}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        for arg in node.args.args:
            if arg.arg != "inner" or arg.annotation is None:
                continue
            inner_type = ast.unparse(arg.annotation)
            assert inner_type not in _FORBIDDEN_INNER_ANNOTATION_NAMES, (
                f"{path.name} inner annotation must be a contract, not {inner_type}"
            )


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_mp6f_production_modules_no_type_ignore(path: Path) -> None:
    text = path.read_text(encoding="utf-8-sig")
    assert "type: ignore" not in text, f"{path.name} must not use type: ignore"


def test_mp6f_context_view_inner_uses_context_view_composer_contract() -> None:
    adapters_path = _SOURCE_DIR / "collaborative_activity_source_adapters.py"
    tree = ast.parse(adapters_path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "ContextViewComposerWithActivityPublication":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            for arg in item.args.args:
                if arg.arg != "inner":
                    continue
                if arg.annotation is None:
                    pytest.fail("inner must be annotated")
                assert ast.unparse(arg.annotation) == "ContextViewComposer", (
                    "ContextView inner must be ContextViewComposer"
                )


def test_mp6f_no_log_and_continue_publication_semantics() -> None:
    adapters = (_SOURCE_DIR / "collaborative_activity_source_adapters.py").read_text(
        encoding="utf-8-sig",
    )
    assert "LOG_AND_CONTINUE" not in adapters
    assert "log_and_continue" not in adapters.lower()
    tree = ast.parse(adapters)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            pytest.fail("broad exception swallow forbidden in publication path")


def test_mp6f_pluginability_tests_no_type_ignore() -> None:
    text = _PLUGINABILITY_TEST.read_text(encoding="utf-8-sig")
    assert "type: ignore" not in text
