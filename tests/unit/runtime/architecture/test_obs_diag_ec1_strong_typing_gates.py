# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-EC1 — strong typing gates for canonical OBS integration contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.integrations.contracts.observability_backend import TraceRecord

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OBS_BACKEND_CONTRACT = _REPO_ROOT / "intergrax" / "integrations" / "contracts" / "observability_backend.py"


def test_trace_record_metadata_is_typed_observability_attributes() -> None:
    from intergrax.contracts.application_observability_attributes import ObservabilityAttributeValue

    field = TraceRecord.model_fields["metadata"]
    assert field.annotation == dict[str, ObservabilityAttributeValue]


def test_observability_backend_contract_no_raw_any_metadata_annotation() -> None:
    tree = ast.parse(_OBS_BACKEND_CONTRACT.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "TraceRecord":
            continue
        for item in node.body:
            if (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id == "metadata"
                and item.annotation is not None
            ):
                src = ast.unparse(item.annotation)
                assert "Any" not in src
                return
    pytest.fail("TraceRecord.metadata annotation not found")


def test_interrupt_payload_metadata_is_typed_observability_attributes() -> None:
    from intergrax.contracts.application_observability_attributes import ObservabilityAttributeValue
    from intergrax.runtime.events.payloads.canonical import InterruptPayloadV1

    field = InterruptPayloadV1.model_fields["metadata"]
    assert field.annotation == dict[str, ObservabilityAttributeValue]


def test_runtime_event_payload_policy_covers_all_enum_members() -> None:
    from intergrax.contracts.runtime_event_type import RuntimeEventType
    from intergrax.runtime.events.runtime_event_payload_policy import iter_runtime_event_payload_policies

    covered = {event_type for event_type, _ in iter_runtime_event_payload_policies()}
    assert covered == set(RuntimeEventType)


def test_as_evidence_persistence_port_enforces_canonical_write_boundary() -> None:
    adapter_path = _REPO_ROOT / "intergrax" / "runtime" / "events" / "evidence_persistence_adapter.py"
    source = adapter_path.read_text(encoding="utf-8")
    assert "ValidatingEvidencePersistencePort" in source
    assert "CanonicalRuntimeEventWriteValidatedPort" in source


def test_event_bus_commits_accepted_canonical_representation() -> None:
    bus_path = _REPO_ROOT / "intergrax" / "runtime" / "events" / "event_bus.py"
    source = bus_path.read_text(encoding="utf-8")
    assert "committed = self._commit_durable_evidence" in source
    assert "positioned.event" in source
