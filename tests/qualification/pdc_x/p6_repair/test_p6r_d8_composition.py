# © Artur Czarnecki. All rights reserved.

"""B-P6-R items 8, 9 and 14 - composition-root wiring of D8 components, reference fragment, touchpoint registry."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
from intergrax.context.provider_lifecycle import canonicalize_fragment
from intergrax.contracts.artifact_ref import ArtifactClass
from intergrax.runtime.nexus.artifacts.context_reference import ArtifactReferenceToken, reference_fragment
from intergrax.runtime.nexus.artifacts.sqlite_artifact_store import SQLiteArtifactStore
from intergrax.runtime.nexus.artifacts.store_base import ArtifactStore
from intergrax.runtime.persistence.sqlite_composition import create_sqlite_runtime_persistence
from intergrax.runtime.persistence.sqlite_opens import open_artifact_store_at
from tests.qualification.ebh_5._sqlite_touchpoint_registry import (
    SqliteTouchpointClassification,
    sqlite_touchpoint_by_path,
)

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "a" * 64


def _token() -> ArtifactReferenceToken:
    return ArtifactReferenceToken(
        tenant_id="tenant-a",
        artifact_class=ArtifactClass.TOOL_OUTPUT,
        artifact_id="spill_1",
        version=1,
        content_digest=_DIGEST,
    )


# ----------------------------------------------------------------------- item 8: SQLite root


def test_open_artifact_store_at_creates_the_parent_and_a_durable_store(tmp_path: Path) -> None:
    store = open_artifact_store_at(tmp_path / "nested" / "artifacts.sqlite3")
    assert isinstance(store, SQLiteArtifactStore)
    assert (tmp_path / "nested" / "artifacts.sqlite3").exists()


def test_the_sqlite_persistence_bundle_composes_an_artifact_store(tmp_path: Path) -> None:
    bundle = create_sqlite_runtime_persistence(data_dir=tmp_path / "sqlite")
    stores = [f.name for f in fields(bundle) if isinstance(getattr(bundle, f.name), ArtifactStore)]
    assert stores == ["artifact_store"]
    assert (tmp_path / "sqlite" / "artifacts.sqlite3").exists()


# ----------------------------------------------------------------------- item 9: reference fragment


def test_the_reference_fragment_is_a_valid_provider_observation() -> None:
    from tests.qualification.pdc_x.xd.test_xd_09_artifacts_context import _ReferenceProvider as provider_type

    fragment = reference_fragment(_token(), fragment_id="ref-1")
    assert isinstance(fragment.metadata.get("tool_call_id"), str) and fragment.metadata["tool_call_id"].strip()
    assert isinstance(fragment.metadata.get("tool_name"), str) and fragment.metadata["tool_name"].strip()
    assert fragment.metadata["artifact_ref"]["artifact_id"] == "spill_1"
    assert fragment.mandatory is True

    provider = provider_type("p6r.refs", {fragment.source}, lambda: [fragment])
    canonical = canonicalize_fragment(fragment, descriptor=provider.descriptor)
    assert canonical.content == fragment.content


def test_the_originating_call_can_be_named_explicitly() -> None:
    fragment = reference_fragment(_token(), fragment_id="ref-1", tool_call_id="call-9", tool_name="sql.query")
    assert fragment.metadata["tool_call_id"] == "call-9"
    assert fragment.metadata["tool_name"] == "sql.query"


def test_the_default_call_link_is_stable_per_artifact() -> None:
    first = reference_fragment(_token(), fragment_id="a")
    second = reference_fragment(_token(), fragment_id="b")
    assert first.metadata["tool_call_id"] == second.metadata["tool_call_id"]


# ----------------------------------------------------------------------- item 14: touchpoint registry


def test_the_sqlite_artifact_store_is_registered_as_a_touchpoint() -> None:
    from tests.qualification.ebh_5._sqlite_touchpoint_registry import assert_closed_world_sqlite_accountability

    record = sqlite_touchpoint_by_path()["intergrax/runtime/nexus/artifacts/sqlite_artifact_store.py"]
    assert record.classification is SqliteTouchpointClassification.FIXED_SUBSTRATE
    assert_closed_world_sqlite_accountability()
