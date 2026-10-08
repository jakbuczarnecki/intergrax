# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P2 mechanical persistence qualification gates."""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
from pathlib import Path

import pytest

from tests.qualification.state_x._state_x_closed_world_durable_state_support import (
    assert_durable_state_discovery_fully_classified,
)
from tests.qualification.state_x._state_x_final_support import (
    assert_family_inventory_closed_world,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_TRACE_X_P5_R2_P2_START_HEAD = "9351cd7ff8697e38a169afb24b14865b2206a98a"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERSISTENCE_MODULE = _REPO_ROOT / "intergrax/applications/_shared/integrations/persistence.py"
_CONTRACT_MODULES = (
    "intergrax.integrations.contracts.existing_capability_configuration_opportunity",
    "intergrax.integrations.contracts.execution_integration_configuration_pinning",
    "intergrax.contracts.execution_integration_configuration_provenance",
)


def test_txp5r2p2_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _TRACE_X_P5_R2_P2_START_HEAD, "HEAD"],
    )


def test_txp5r2p2_q02_required_store_ports_exist() -> None:
    opportunity = importlib.import_module(
        "intergrax.integrations.contracts.existing_capability_configuration_opportunity",
    )
    pinning = importlib.import_module(
        "intergrax.integrations.contracts.execution_integration_configuration_pinning",
    )
    assert hasattr(opportunity, "ExistingCapabilityConfigurationOpportunityStore")
    assert hasattr(pinning, "ExecutionIntegrationConfigurationPinningStore")


def test_txp5r2p2_q03_neutral_reader_has_no_pin() -> None:
    provenance = importlib.import_module(
        "intergrax.contracts.execution_integration_configuration_provenance",
    )
    reader = provenance.ExecutionIntegrationConfigurationProvenanceReader
    assert "pin" not in inspect.getmembers(reader, predicate=inspect.isfunction)


def test_txp5r2p2_q04_no_latest_methods_on_store_contracts() -> None:
    forbidden = ("read_latest", "get_latest", "find_current", "resolve_active")
    for module_name in _CONTRACT_MODULES:
        path = _REPO_ROOT / Path(*module_name.split("."))
        path = path.with_suffix(".py")
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{module_name} contains forbidden {token}"


def test_txp5r2p2_q05_persistence_uses_conditional_document_and_cas() -> None:
    source = _PERSISTENCE_MODULE.read_text(encoding="utf-8")
    assert "ConditionalDocumentStore" in source
    assert "put_if_absent" in source
    assert "compare_and_set" in source
    assert "schema_version" in source


def test_txp5r2p2_q06_no_pickle_or_reflection_in_persistence() -> None:
    tree = ast.parse(_PERSISTENCE_MODULE.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"getattr", "hasattr"}:
                violations.append(f"getattr/hasattr at line {node.lineno}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pickle":
                    violations.append("pickle import")
        if isinstance(node, ast.ImportFrom) and node.module == "pickle":
            violations.append("pickle import")
    assert not violations, violations


def test_txp5r2p2_q07_in_memory_adapters_marked_non_durable() -> None:
    persistence = importlib.import_module(
        "intergrax.applications._shared.integrations.persistence",
    )
    for cls_name in (
        "InMemoryExistingCapabilityConfigurationOpportunityStore",
        "InMemoryExecutionIntegrationConfigurationPinningStore",
    ):
        cls = getattr(persistence, cls_name)
        instance = cls(payload_codecs=persistence.default_integration_configuration_payload_codec_registry()) if cls_name.startswith("InMemoryExisting") else cls()
        assert instance.is_durable is False


def test_txp5r2p2_q08_state_x_current_head_delta_classification() -> None:
    assert_durable_state_discovery_fully_classified()
    assert_family_inventory_closed_world()

