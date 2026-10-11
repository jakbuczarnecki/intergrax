# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R2 adversarial persisted read-compat tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.agent_distribution.sqlite_payload_legacy import (
    decode_agent_artifact_metadata_payload,
)
from intergrax.agent_distribution.stores import SCHEMA_AGENT_ARTIFACT_METADATA_V1
from intergrax.applications._shared.registry_snapshot_persistence import (
    SCHEMA_REGISTRY_SNAPSHOT_V1,
    parse_registry_snapshot_persisted,
)
from intergrax.contracts.idempotency_store import (
    IdempotencyLedgerSchemaError,
    assert_tool_idempotency_ledger_schema_version,
)
from intergrax.contracts.runtime_event import parse_runtime_event_payload
from intergrax.memory.conversational_memory_persistence import (
    SCHEMA_CONVERSATIONAL_MEMORY_V1,
    parse_conversational_memory_persisted,
)
from intergrax.memory.user_profile_serialization import (
    SCHEMA_USER_PROFILE_V1,
    deserialize_user_profile_payload,
)
from intergrax.runtime.task_memory.sqlite_row_legacy import task_memory_record_from_sqlite_row

pytestmark = pytest.mark.unit

_HISTORICAL_AD_ARTIFACT = {
    "package_digest": "sha256:" + "a" * 64,
    "artifact_store_ref": "store",
    "distribution_package_id": "pkg",
    "agent_project_metadata_ref": "meta",
    "tombstoned": False,
}


def test_ad_legacy_artifact_accepted_and_versioned() -> None:
    restored = decode_agent_artifact_metadata_payload(json.dumps(_HISTORICAL_AD_ARTIFACT))
    assert restored.schema_version == SCHEMA_AGENT_ARTIFACT_METADATA_V1


def test_ad_legacy_artifact_extra_field_rejected() -> None:
    payload = {**_HISTORICAL_AD_ARTIFACT, "unexpected": True}
    with pytest.raises(ValueError):
        decode_agent_artifact_metadata_payload(json.dumps(payload))


def test_user_profile_legacy_historical_wire() -> None:
    payload = {
        "identity": {"user_id": "u1"},
        "preferences": {},
        "memory_entries": [],
        "version": 3,
    }
    profile = deserialize_user_profile_payload(payload)
    assert profile.version == 3
    assert profile.identity.user_id == "u1"


def test_user_profile_unknown_schema_version_fail_closed() -> None:
    with pytest.raises(ValueError, match="schema version"):
        deserialize_user_profile_payload(
            {"schema_version": "user_profile.v0", "identity": {"user_id": "u1"}}
        )


def test_conversational_memory_legacy_wire() -> None:
    raw = {
        "max_messages": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
    envelope = parse_conversational_memory_persisted(raw)
    assert envelope.schema_version == SCHEMA_CONVERSATIONAL_MEMORY_V1
    assert envelope.messages[0].content == "hi"


def test_registry_snapshot_legacy_wire() -> None:
    raw = {
        "tool_ids": ["t1"],
        "skill_ids": [],
        "prompt_ids": [],
        "agent_contract_ids": [],
        "evaluation_registry_ids": [],
    }
    envelope = parse_registry_snapshot_persisted(raw)
    assert envelope.schema_version == SCHEMA_REGISTRY_SNAPSHOT_V1


def test_idempotency_legacy_missing_version_fail_closed() -> None:
    with pytest.raises(IdempotencyLedgerSchemaError):
        assert_tool_idempotency_ledger_schema_version(None)


def test_runtime_event_missing_schema_version_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_runtime_event_payload(
            {
                "task_id": "task_" + "a" * 32,
                "run_id": "run_" + "b" * 32,
                "attempt_id": "attempt_" + "c" * 32,
                "execution_id": "exec_" + "d" * 32,
                "event_type": "tool.completed",
                "phase": "step_execution",
            }
        )


def test_runtime_event_v1_retired_fail_closed() -> None:
    with pytest.raises(ValueError, match="retired"):
        parse_runtime_event_payload({"schema_version": "runtime_event.v1"})


def test_task_memory_null_schema_version_migrates(tmp_path: Path) -> None:
    row = {
        "record_id": "r1",
        "tenant_id": "t",
        "task_id": "task",
        "namespace": "ns",
        "key": "k",
        "value_json": "{}",
        "created_at_utc": "2020-01-01T00:00:00+00:00",
        "updated_at_utc": "2020-01-01T00:00:00+00:00",
        "provenance_json": "{}",
        "schema_version": None,
    }

    class _Row(dict):
        def __getitem__(self, key: str) -> object:
            return super().__getitem__(key)

    record = task_memory_record_from_sqlite_row(_Row(row))
    assert record.schema_version == "task_memory.v1"
