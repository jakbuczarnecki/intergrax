# © Artur Czarnecki. All rights reserved.

"""Cross-layer Agent Platform identity chain acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    AgentPlatformAdminBlockedError,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.errors import (
    MaterializationInputConflict,
    RuntimeRevisionLifecycleError,
)
from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.in_memory_stores import InMemoryRuntimeRevisionStore
from intergrax.applications._shared.registry_projection import (
    RegistryProjectionError,
    build_registry_projection,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _APP,
    _ARTIFACT,
    _ENV,
    _build_request,
    _install_bind,
    build_admin_stack,
)
from tests.unit.applications.test_registry_projection_ap10 import (
    _bundle_parts,
    _entry,
    _manifest,
    _revision,
    _roster,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_cross_layer_identity_chain_through_admin_serving_state() -> None:
    stack = build_admin_stack()
    _install_bind(stack)
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id="bind-search",
        request=SetAgentEnablementRequest(expected_revision=0),
    )

    roster_view = stack.service.inspect_effective_roster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id="rel-1",
    )
    built = stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-chain-1"),
    )
    revision = stack.service.inspect_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-chain-1",
    )

    assert revision.application_environment_id == _ENV
    assert revision.effective_roster_revision_id == built.effective_roster_revision_id
    assert revision.materialized_runtime_lock_id == built.materialized_runtime_lock_id
    assert revision.runtime_graph_digest == built.runtime_graph_digest
    assert revision.materialization_artifact_digest == built.materialization_artifact_digest
    assert revision.revision_state is RuntimeRevisionState.VALIDATED
    assert roster_view.effective_roster_revision_id == built.effective_roster_revision_id

    lock = stack.state.locks[built.materialized_runtime_lock_id or ""]
    assert lock.lock_digest == built.materialized_runtime_lock_digest

    activated = stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=ActivateRuntimeRevisionRequest(
            runtime_revision_id="rev-chain-1",
            artifact_locator=built.artifact_locator or "test://artifact",
            expected_artifact_digest=built.materialization_artifact_digest or _ARTIFACT,
            expected_serving_pointer_revision=0,
        ),
    )
    serving = stack.service.inspect_serving(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert activated.traffic_serving_revision_id == "rev-chain-1"
    assert serving.traffic_serving_revision_id == "rev-chain-1"
    assert serving.active_revision is not None
    assert serving.active_revision.runtime_revision_id == "rev-chain-1"
    assert serving.active_revision.effective_roster_revision_id == built.effective_roster_revision_id
    assert (
        serving.active_revision.materialized_runtime_lock_digest
        == built.materialized_runtime_lock_digest
    )


def test_cross_layer_build_blocked_without_explicit_dependency_resolver() -> None:
    stack = build_admin_stack()
    stack.service._lock_service = None  # type: ignore[attr-defined]
    with pytest.raises(AgentPlatformAdminBlockedError) as exc:
        stack.service.build_application_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            request=_build_request("rev-no-resolver"),
        )
    assert exc.value.blocker_code == "AP-11_BLOCKED_BY_MISSING_DEPENDENCY_RESOLVER"


def test_cross_layer_materialization_rejects_roster_application_mismatch(
    tmp_path: Path,
) -> None:
    from tests.unit.agent_distribution.test_agent_distribution_materialization import (
        _build_fixture,
    )

    materialization_input, _, _, _ = _build_fixture(tmp_path)
    foreign_roster = materialization_input.effective_roster.model_copy(
        update={"application_id": "foreign-app"},
    )
    tampered = materialization_input.model_copy(update={"effective_roster": foreign_roster})
    with pytest.raises(MaterializationInputConflict, match="application_id"):
        RuntimeMaterializationService._validate_input_consistency(tampered)


def test_cross_layer_registry_projection_rejects_app_b_revision_with_app_a_roster() -> None:
    roster_app_a = _roster((_entry("search"),))
    revision_app_b = _revision("rev-cross").model_copy(update={"application_id": "app_b"})
    bundle = _bundle_parts(revision_app_b, roster_app_a, _manifest(app_id="app_a"))
    with pytest.raises(RegistryProjectionError, match="application_id"):
        build_registry_projection(bundle)


def test_cross_layer_mark_validated_rejects_lock_digest_mutation() -> None:
    revision_service = RuntimeRevisionService(InMemoryRuntimeRevisionStore())
    candidate = RuntimeRevision(
        runtime_revision_id="rev-mut",
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-1",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="sha256:" + ("1" * 64),
        runtime_graph_digest="sha256:" + ("2" * 64),
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        revision_state=RuntimeRevisionState.CANDIDATE,
    )
    revision_service.persist_candidate_revision(candidate)
    mutated = candidate.model_copy(
        update={
            "revision_state": RuntimeRevisionState.VALIDATED,
            "materialization_artifact_digest": _ARTIFACT,
            "materialized_runtime_lock_digest": "sha256:" + ("9" * 64),
        }
    )
    with pytest.raises(RuntimeRevisionLifecycleError, match="materialized_runtime_lock_digest"):
        revision_service.mark_validated("rev-mut", validated_revision=mutated)
