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
