# © Artur Czarnecki. All rights reserved.

"""Typed durable registry projection descriptor contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.applications._shared.registry_projection_descriptor import (
    PROJECTION_DESCRIPTOR_CONTRACT_VERSION,
    BuildContextDescriptorSnapshot,
    EnvironmentIdentitySnapshot,
    InMemoryRuntimeRegistryProjectionDescriptorStore,
    RuntimeRegistryProjectionDescriptor,
    SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.agent_distribution.sqlite_stores import build_sqlite_agent_distribution_store_bundle

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app_descriptor_proof"
_ENV = "env-descriptor-proof"
_REVISION = "rev-descriptor-proof"
_DIGEST = "sha256:" + ("a" * 64)


def _manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=_APP,
        name="Descriptor Proof",
        agents=[],
        integration_profile=IntegrationProfile(),
    )


def _descriptor() -> RuntimeRegistryProjectionDescriptor:
    manifest = _manifest()
    skill_profile = SkillProfile(enabled=["skill.alpha"])
    tool_profile = ToolProfile(enabled=["tool.beta"])
    build_context = ApplicationBuildContext.for_manifest(
        manifest,
        skill_profile=skill_profile,
        tool_profile=tool_profile,
        strict_harness=True,
        environment=None,
    )
    snapshot = BuildContextDescriptorSnapshot.from_build_context(build_context)
    snapshot = snapshot.model_copy(
        update={
            "environment_identity": EnvironmentIdentitySnapshot(profile_id=_ENV),
        }
    )
    return RuntimeRegistryProjectionDescriptor(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=_REVISION,
        application_release_id="rel-1",
        effective_roster_revision_id="roster-1",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="sha256:" + ("b" * 64),
        materialization_artifact_locator="test:///tmp/artifact",
        materialization_artifact_digest=_DIGEST,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        manifest=manifest,
        build_context_snapshot=snapshot,
    )


def test_descriptor_typed_round_trip_in_memory() -> None:
    descriptor = _descriptor()
    store = InMemoryRuntimeRegistryProjectionDescriptorStore()
    store.put(descriptor)
    loaded = store.get_for_revision(_APP, _ENV, _REVISION)
    assert loaded == descriptor
    assert loaded is not None
    assert loaded.manifest == descriptor.manifest
    assert loaded.build_context_snapshot == descriptor.build_context_snapshot
    assert loaded.build_context_snapshot.skill_profile == SkillProfile(enabled=["skill.alpha"])
    assert loaded.build_context_snapshot.tool_profile == ToolProfile(enabled=["tool.beta"])


def test_descriptor_typed_round_trip_sqlite(tmp_path: Path) -> None:
    descriptor = _descriptor()
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "descriptor.db")
    bundle.projection_descriptor_store.put(descriptor)
    reopened = build_sqlite_agent_distribution_store_bundle(tmp_path / "descriptor.db")
    loaded = reopened.projection_descriptor_store.get_for_revision(_APP, _ENV, _REVISION)
    assert loaded == descriptor


def test_descriptor_schema_version_mismatch_rejected() -> None:
    payload = _descriptor().model_dump()
    payload["schema_version"] = "unsupported.v99"
    with pytest.raises(ValueError, match="unsupported schema version"):
        RuntimeRegistryProjectionDescriptor.model_validate(payload)


def test_descriptor_contract_version_mismatch_rejected() -> None:
    payload = _descriptor().model_dump()
    payload["descriptor_version"] = "projection_descriptor_contract.v99"
    with pytest.raises(ValueError, match="unsupported descriptor contract"):
        RuntimeRegistryProjectionDescriptor.model_validate(payload)


def test_descriptor_sqlite_schema_version_mismatch_rejected(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "schema.db")
    with bundle.database._connect() as conn:  # noqa: SLF001
        conn.execute(
            """
            INSERT INTO projection_descriptors(runtime_revision_id, scope_key, payload_json)
            VALUES (?, ?, ?)
            """,
            (
                _REVISION,
                f"{_APP}\0{_ENV}",
                json.dumps({"schema_version": "unsupported.v99"}),
            ),
        )
        conn.commit()
    with pytest.raises(ValueError, match="unsupported schema version"):
        bundle.projection_descriptor_store.get_for_revision(_APP, _ENV, _REVISION)


def test_descriptor_sqlite_contract_version_mismatch_rejected(tmp_path: Path) -> None:
    descriptor = _descriptor()
    payload = json.loads(descriptor.model_dump_json())
    payload["descriptor_version"] = "projection_descriptor_contract.v99"
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "contract.db")
    with bundle.database._connect() as conn:  # noqa: SLF001
        conn.execute(
            """
            INSERT INTO projection_descriptors(runtime_revision_id, scope_key, payload_json)
            VALUES (?, ?, ?)
            """,
            (
                _REVISION,
                f"{_APP}\0{_ENV}",
                json.dumps(payload),
            ),
        )
        conn.commit()
    with pytest.raises(ValueError, match="unsupported descriptor contract"):
        bundle.projection_descriptor_store.get_for_revision(_APP, _ENV, _REVISION)


def test_descriptor_snapshot_models_are_frozen() -> None:
    descriptor = _descriptor()
    with pytest.raises(Exception):
        descriptor.runtime_revision_id = "mutated"
    with pytest.raises(Exception):
        descriptor.build_context_snapshot.strict_harness = False
    identity = descriptor.build_context_snapshot.environment_identity
    assert identity is not None
    with pytest.raises(Exception):
        identity.profile_id = "mutated"


def test_environment_identity_only_profile_id_required_for_projection_rebuild() -> None:
    """Projection fingerprinting ignores rich environment config beyond profile_id."""
    manifest = _manifest()
    minimal_snapshot = BuildContextDescriptorSnapshot(
        environment_identity=EnvironmentIdentitySnapshot(profile_id=_ENV),
    )
    build_context = minimal_snapshot.to_build_context(manifest)
    assert build_context.environment is not None
    assert build_context.environment.profile_id == _ENV


def test_descriptor_module_has_no_dict_any_persistence_fields() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "intergrax"
        / "applications"
        / "_shared"
        / "registry_projection_descriptor.py"
    ).read_text(encoding="utf-8")
    assert "dict[str, Any]" not in source
    assert "manifest_json" not in source
    assert "skill_profile_json" not in source
    assert "tool_profile_json" not in source


def test_supported_schema_and_contract_versions_recorded() -> None:
    descriptor = _descriptor()
    assert descriptor.schema_version == SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1
    assert descriptor.descriptor_version == PROJECTION_DESCRIPTOR_CONTRACT_VERSION
