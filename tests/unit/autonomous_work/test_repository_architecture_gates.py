# © Artur Czarnecki. All rights reserved.

"""AW-2A — Autonomous Work repository architecture gate tests."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerInstanceRepository,
)
from intergrax.autonomous_work.repository import WorkerInstanceRepository

pytestmark = pytest.mark.unit

_FORBIDDEN_PROVIDER_IMPORTS = (
    "sqlalchemy",
    "redis",
    "boto3",
    "psycopg",
    "asyncpg",
    "pymongo",
)


def _module_source_paths(package_name: str) -> list[Path]:
    package = importlib.import_module(package_name)
    paths: list[Path] = []
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        source_path = getattr(module, "__file__", None)
        if source_path:
            paths.append(Path(source_path))
    return paths


@pytest.mark.unit
def test_repository_ports_do_not_import_runtime_services() -> None:
    forbidden_prefixes = (
        "intergrax.runtime",
        "intergrax.applications",
        "agents.",
        "applications.",
    )
    for path in _module_source_paths("intergrax.autonomous_work"):
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("from ", "import ")):
                continue
            for prefix in forbidden_prefixes:
                assert prefix not in stripped, f"{path} imports forbidden runtime: {line}"


@pytest.mark.unit
def test_repository_ports_have_no_provider_dependencies() -> None:
    for path in _module_source_paths("intergrax.autonomous_work"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    assert root not in _FORBIDDEN_PROVIDER_IMPORTS, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                assert root not in _FORBIDDEN_PROVIDER_IMPORTS, node.module


@pytest.mark.unit
def test_in_memory_adapter_is_separate_from_contracts_package() -> None:
    contracts_module = importlib.import_module("intergrax.contracts.autonomous_work")
    adapter_module = importlib.import_module(
        "intergrax.autonomous_work.in_memory_repository"
    )
    assert contracts_module.__file__ is not None
    assert adapter_module.__file__ is not None
    contracts_root = Path(contracts_module.__file__).parent
    adapter_root = Path(adapter_module.__file__).parent
    assert contracts_root != adapter_root
    assert "contracts" in str(contracts_root).replace("\\", "/")
    assert "autonomous_work" in str(adapter_root).replace("\\", "/")
    assert "contracts" not in str(adapter_root).replace("\\", "/")


@pytest.mark.unit
def test_consumer_depends_on_port_not_adapter_implementation() -> None:
    def consumer(repo: WorkerInstanceRepository) -> None:
        assert repo.capabilities.reference_only is True

    consumer(InMemoryWorkerInstanceRepository())


@pytest.mark.unit
def test_repository_module_has_no_lifecycle_transition_logic() -> None:
    forbidden_tokens = (
        "transition_to",
        "apply_transition",
        "validate_transition",
        "lifecycle_graph",
    )
    for path in _module_source_paths("intergrax.autonomous_work"):
        source = path.read_text(encoding="utf-8").lower()
        for token in forbidden_tokens:
            assert token not in source, f"{path} contains lifecycle transition logic: {token}"
