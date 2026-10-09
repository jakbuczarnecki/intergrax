# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P2 mechanical persistence qualification gates."""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
from pathlib import Path

from testing_support.integration_configuration_payload_codecs import (
    qualification_integration_configuration_payload_codec_registry,
)

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
        instance = (
            cls(payload_codecs=qualification_integration_configuration_payload_codec_registry())
            if cls_name.startswith("InMemoryExisting")
            else cls()
        )
        assert instance.is_durable is False


def test_txp5r2p2_q08_state_x_current_head_delta_classification() -> None:
    assert_durable_state_discovery_fully_classified()
    assert_family_inventory_closed_world()


def test_txp5r2p2_q09_shared_persistence_has_no_sqlite_codec_import() -> None:
    source = _PERSISTENCE_MODULE.read_text(encoding="utf-8")
    assert "sqlite.configuration_payload_codec" not in source
    assert "relational_store.sqlite" not in source


def test_txp5r2p2_q10_wire_requires_explicit_payload_codecs() -> None:
    persistence = importlib.import_module(
        "intergrax.applications._shared.integrations.persistence",
    )
    signature = inspect.signature(persistence.wire_existing_capability_configuration_opportunity_store)
    assert signature.parameters["payload_codecs"].default is inspect.Parameter.empty
    assert not hasattr(persistence, "default_integration_configuration_payload_codec_registry")


def test_txp5r2p2_q11_codec_registry_is_mapping_proxy_immutable() -> None:
    codec_module = importlib.import_module(
        "intergrax.integrations.contracts.integration_configuration_payload_codec",
    )
    registry_cls = codec_module.IntegrationConfigurationPayloadCodecRegistry
    field = registry_cls.__dataclass_fields__["_codecs"]
    assert "Mapping" in str(field.type)


def test_txp5r2p2_q12_kv_crash_repair_regression_present() -> None:
    unit_tests = (
        _REPO_ROOT / "tests/unit/applications/integrations/test_trace_x_p5_r2_p2_persistence.py"
    ).read_text(encoding="utf-8")
    assert "test_kv_provenance_crash_after_index_before_record_fails_closed_then_repair" in unit_tests
    assert "test_kv_provenance_legacy_orphan_record_without_index_repaired_on_retry" in unit_tests


def test_txp5r2p2_q14_document_provenance_read_all_traverses_next_cursor() -> None:
    source = _PERSISTENCE_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    document_read_all = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == (
            "DocumentStoreExecutionIntegrationConfigurationPinningStore"
        ):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "read_all":
                    document_read_all = item
    assert document_read_all is not None
    body_source = ast.get_source_segment(source, document_read_all) or ""
    assert "next_cursor" in body_source
    assert "cursor=" in body_source.replace(" ", "")
    assert "while " in body_source


def test_txp5r2p2_q15_document_provenance_read_all_pagination_behavioral() -> None:
    from intergrax.contracts.execution_identity import validate_execution_id
    from intergrax.applications._shared.integrations.persistence import (
        DocumentStoreExecutionIntegrationConfigurationPinningStore,
    )
    from tests.unit.applications.integrations.test_trace_x_p5_r2_p2_persistence import (
        InMemoryConditionalDocumentStore,
        _pin_distinct_provenance_records,
    )

    execution_id = validate_execution_id("exec_01234567890123456789012345678901")
    backing = InMemoryConditionalDocumentStore(max_page_size=2)
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(backing)
    expected = _pin_distinct_provenance_records(store, 5)
    read = store.read_all(tenant_id="tenant-a", execution_id=execution_id)
    assert read == tuple(expected)
    assert backing.query_call_count >= 3


def test_txp5r2p2_q13_state_x_atomicity_does_not_claim_multi_record_transaction() -> None:
    from tests.qualification.state_x._state_x_explicit_mechanism_classifications import (
        EXPLICIT_MECHANISM_CLASSIFICATIONS,
    )

    entry = EXPLICIT_MECHANISM_CLASSIFICATIONS[
        "path:intergrax/applications/_shared/integrations/persistence.py"
    ]
    text = entry.atomicity_semantics.lower()
    assert "no distributed transaction" in text
    assert "index" in text and "record" in text

