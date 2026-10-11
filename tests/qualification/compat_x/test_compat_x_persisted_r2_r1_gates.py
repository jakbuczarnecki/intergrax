# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R1 durable schema identity & SQLite authority gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.agent_distribution.agent_contract_authority import (
    SCHEMA_PACKAGE_AGENT_CONTRACT_AUTHORITY_V1,
    PackageAgentContractAuthorityRecord,
)
from intergrax.agent_distribution.sqlite_stores import (
    _decode,
    build_sqlite_agent_distribution_store_bundle,
)
from intergrax.agent_distribution.stores import (
    SCHEMA_AGENT_ARTIFACT_METADATA_V1,
    AgentArtifactMetadata,
)
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.memory.user_profile_serialization import (
    SCHEMA_USER_PROFILE_V1,
    user_profile_from_dict,
    user_profile_to_dict,
)
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_r2_r1_gates import (
    persisted_r2_r1_inventory_gate_passes,
)
from tests.qualification.compat_x._compat_x_persisted_types import PersistedVersionDisposition

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_cx_r2_r1_inventory_gate() -> None:
    assert persisted_r2_r1_inventory_gate_passes()


def test_cx_r2_r1_no_missing_version_semantic_families() -> None:
    missing = [
        f
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.version_disposition == PersistedVersionDisposition.MISSING_VERSION
    ]
    assert missing == []


def test_cx_r2_r1_kv_opaque_carrier_classification() -> None:
    family = next(f for f in PERSISTED_SCHEMA_FAMILIES if f.family_id == "distributed.kv_entry")
    assert (
        family.version_disposition
        == PersistedVersionDisposition.OPAQUE_CARRIER_NOT_SEMANTIC_PAYLOAD
    )


def test_cx_r2_r1_ad_agent_artifact_metadata_decode_fail_closed() -> None:
    payload = {
        "package_digest": "sha256:" + "a" * 64,
        "artifact_store_ref": "store",
        "distribution_package_id": "pkg",
        "agent_project_metadata_ref": "meta",
        "tombstoned": False,
    }
    with pytest.raises(ValueError, match="schema version"):
        _decode(AgentArtifactMetadata, json.dumps(payload), SCHEMA_AGENT_ARTIFACT_METADATA_V1)


def test_cx_r2_r1_ad_agent_artifact_metadata_round_trip() -> None:
    model = AgentArtifactMetadata(
        package_digest="sha256:" + "b" * 64,
        artifact_store_ref="store",
        distribution_package_id="pkg",
        agent_project_metadata_ref="meta",
    )
    restored = _decode(
        AgentArtifactMetadata,
        model.model_dump_json(),
        SCHEMA_AGENT_ARTIFACT_METADATA_V1,
    )
    assert restored.schema_version == SCHEMA_AGENT_ARTIFACT_METADATA_V1


def test_cx_r2_r1_user_profile_schema_independent_of_business_version() -> None:
    profile = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
        version=99,
    )
    payload = user_profile_to_dict(profile)
    assert payload["schema_version"] == SCHEMA_USER_PROFILE_V1
    assert payload["version"] == 99
    payload.pop("schema_version")
    with pytest.raises(ValueError, match="schema version"):
        user_profile_from_dict(payload)


def _minimal_agent_contract() -> object:
    import sys

    sys.path.insert(0, "agents/echo")
    try:
        from contract import build_agent_contract

        return build_agent_contract()
    finally:
        sys.path.pop(0)


def test_cx_r2_r1_sqlite_contract_authority_survives_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "ad.db"
    bundle1 = build_sqlite_agent_distribution_store_bundle(db_path)
    digest = "sha256:" + "c" * 64
    record = PackageAgentContractAuthorityRecord.from_validated_contract(
        package_digest=digest,
        distribution_package_id="intergrax-echo-agent",
        artifact_store_ref="artifact://echo",
        agent_project_metadata_ref="meta://echo",
        contract=_minimal_agent_contract(),  # type: ignore[arg-type]
    )
    bundle1.artifact_metadata_store.persist_package_contract_authority(record)
    bundle2 = build_sqlite_agent_distribution_store_bundle(db_path)
    loaded = bundle2.artifact_metadata_store.get_package_contract_authority(
        record.package_digest,
        record.contract_id,
    )
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_PACKAGE_AGENT_CONTRACT_AUTHORITY_V1
    assert loaded.metadata_digest == record.metadata_digest


def test_cx_r2_r1_distributed_kv_opaque_bytes_round_trip(tmp_path: Path) -> None:
    store = SqliteDistributedKVStore(tmp_path / "kv.db")
    payload = b"\x00\x01opaque-bytes"
    store.set("tenant", "key", payload)
    assert store.get("tenant", "key") == payload
