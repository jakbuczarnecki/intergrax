# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.events.phase_coverage import (
    list_unmapped_event_types,
    list_unmapped_ops_filter_hints,
)
from intergrax.runtime.plugins.contract import RuntimePlugin
from intergrax.runtime.schema.registry import (
    RUNTIME_SCHEMA_REGISTRY,
    current_runtime_version,
    validate_schema_version,
)

pytestmark = pytest.mark.gate


def test_runtime_schema_registry_contains_core_contracts():
    assert RUNTIME_SCHEMA_REGISTRY["runtime_event"] == "runtime_event.v1"
    assert RUNTIME_SCHEMA_REGISTRY["agent_decision"] == "agent_decision.v1"


def test_validate_schema_version_accepts_known_versions():
    assert validate_schema_version("runtime_event", "runtime_event.v1") is True
    assert validate_schema_version("runtime_event", "runtime_event.v2") is False


def test_current_runtime_version_lists_registered_schemas():
    info = current_runtime_version()
    assert "runtime_event.v1" in info.supported_schemas
    assert info.contract_bundle == "uaep-1.0"


def test_runtime_plugin_contract_is_dataclass():
    plugin = RuntimePlugin(plugin_id="lab.metrics", version="1.0.0")
    assert plugin.plugin_id == "lab.metrics"
    assert plugin.compatible_runtime.contract_bundle == "uaep-1.0"


def test_all_runtime_event_types_have_execution_phase():
    assert list_unmapped_event_types() == []


def test_all_runtime_event_types_have_ops_filter_hint() -> None:
    assert list_unmapped_ops_filter_hints() == []
