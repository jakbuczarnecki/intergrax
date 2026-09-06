# © Artur Czarnecki. All rights reserved.

"""Bounded SQLite agent distribution store unit tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.agent_distribution.binding import ApplicationAgentBinding
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    InstallationState,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.sqlite_stores import (
    SCHEMA_AGENT_DISTRIBUTION_SQLITE_V1,
    build_sqlite_agent_distribution_store_bundle,
)
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingRecord
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.registry_projection_descriptor import (
    BuildContextDescriptorSnapshot,
    RuntimeRegistryProjectionDescriptor,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.core.qualification import QualificationStatus

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app_sqlite_proof"
_ENV = "env-sqlite-proof"
_REVISION = "rev-sqlite-proof"
_DIGEST = "sha256:" + ("a" * 64)


def _bundle(tmp_path: Path):
    return build_sqlite_agent_distribution_store_bundle(tmp_path / "lifecycle.db")


def _trust_record() -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=_DIGEST,
        publisher_identity_ref="publisher:sqlite",
        source_provider_id="catalog:sqlite",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:sqlite",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def test_sqlite_installation_and_binding_persistence(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    installation = AgentInstallationRecord(
        installation_id="inst-1",
        environment_id=_ENV,
        installation_slot_id="slot-1",
        package_identity=AgentPackageIdentity(
            distribution_package_id="pkg-1",
            package_version="1.0.0",
            package_digest=_DIGEST,
        ),
        installation_state=InstallationState.INSTALLED_ACTIVE,
        active_for_slot=True,
        artifact_store_ref="store://artifacts/inst-1",
        trust_record=_trust_record(),
    )
    bundle.installation_store.persist_installation(installation)
    binding = ApplicationAgentBinding(
        application_binding_id="bind-1",
        application_id=_APP,
        application_environment_id=_ENV,
        logical_agent_id="agent-1",
        installation_slot_id="slot-1",
        factory_reference=None,
        enablement=True,
        binding_revision=0,
    )
    bundle.binding_store.persist_binding(binding)
    reopened = _bundle(tmp_path)
    assert reopened.installation_store.get_installation("inst-1") == installation
    assert reopened.binding_store.get_binding("bind-1") == binding


def test_sqlite_revision_serving_and_descriptor_reopen(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    activated_at = datetime.now(UTC)
    revision = RuntimeRevision(
        runtime_revision_id=_REVISION,
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-1",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="sha256:" + ("b" * 64),
        runtime_graph_digest="sha256:" + ("c" * 64),
        materialization_artifact_digest=_DIGEST,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=RuntimeRevisionState.ACTIVE,
        activated_at=activated_at,
    )
    bundle.revision_store.persist_candidate_revision(revision)
    bundle.serving_store.atomic_swap_serving_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=_REVISION,
        prior_revision_id=None,
        committed_at=activated_at,
    )
    manifest = ApplicationManifest.lab(app_id=_APP, name="SQLite Proof", agents=())
    descriptor = RuntimeRegistryProjectionDescriptor(
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
        manifest_json=manifest.model_dump(mode="json"),
        build_context_snapshot=BuildContextDescriptorSnapshot(),
    )
    bundle.projection_descriptor_store.put(descriptor)
    reopened = _bundle(tmp_path)
    loaded_revision = reopened.revision_store.get_revision(_REVISION)
    assert loaded_revision is not None
    assert loaded_revision.runtime_revision_id == _REVISION
    loaded_serving = reopened.serving_store.get_serving_record(_APP, _ENV)
    assert loaded_serving is not None
    assert loaded_serving.traffic_serving_revision_id == _REVISION
    loaded_descriptor = reopened.projection_descriptor_store.get_for_revision(
        _APP,
        _ENV,
        _REVISION,
    )
    assert loaded_descriptor == descriptor


def test_sqlite_descriptor_schema_version_rejection(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    db = bundle.database
    with db._connect() as conn:  # noqa: SLF001
        conn.execute(
            """
            INSERT INTO projection_descriptors(runtime_revision_id, scope_key, payload_json)
            VALUES (?, ?, ?)
            """,
            (
                _REVISION,
                f"{_APP}\0{_ENV}",
                '{"schema_version":"unsupported.v99"}',
            ),
        )
        conn.commit()
    with pytest.raises(ValueError, match="unsupported schema version"):
        bundle.projection_descriptor_store.get_for_revision(_APP, _ENV, _REVISION)


def test_sqlite_meta_schema_version_recorded(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with bundle.database._connect() as conn:  # noqa: SLF001
        row = conn.execute(
            "SELECT value FROM agent_distribution_meta WHERE key = 'schema_version'"
        ).fetchone()
    assert row is not None
    assert row["value"] == SCHEMA_AGENT_DISTRIBUTION_SQLITE_V1
