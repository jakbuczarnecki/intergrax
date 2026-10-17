# © Artur Czarnecki. All rights reserved.

"""R7-Q §C dimension 9 — integrated strict production failure-before-readiness gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from unittest.mock import patch

import pytest

from intergrax.agent_distribution.deployment import (
    DeploymentInstanceState,
    ServingUnitAttestationError,
)
from intergrax.agent_distribution.errors import RuntimeReadinessError
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.production_process_lifecycle import (
    open_production_process_lifecycle,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    resolve_production_artifact_root,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionActivationIntent,
    StrictProductionServingLauncherError,
    open_strict_production_serving_session,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    assert_identity_continuity,
    load_d4_launcher_module,
    repo_root,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _activation_request,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "a4c18fa2e11de099f5025bc12f8aeca64fe98226"


def test_c_dim9_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


@dataclass(slots=True)
class IntegratedServingAuthoritySnapshot:
    traffic_revision_id: str | None
    serving_pointer_revision: int
    active_revision_ids: tuple[str, ...]
    physical_unit_count: int


def _physical_unit_count(adapter: VenvBundleColocatedRuntimeDeploymentAdapter) -> int:
    return len(adapter._units)


def _traffic_revision_id(stores, environment) -> str | None:
    record = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    if record is None:
        return None
    return record.traffic_serving_revision_id


def _authority_snapshot(
    stores,
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
    *,
    environment,
) -> IntegratedServingAuthoritySnapshot:
    serving = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    revisions = stores.revision_store.list_revisions_for_environment(
        environment.application_id,
        environment.application_environment_id,
    )
    active_ids = tuple(
        r.runtime_revision_id
        for r in revisions
        if r.revision_state is RuntimeRevisionState.ACTIVE
    )
    return IntegratedServingAuthoritySnapshot(
        traffic_revision_id=(
            serving.traffic_serving_revision_id if serving is not None else None
        ),
        serving_pointer_revision=(
            serving.serving_pointer_revision if serving is not None else 0
        ),
        active_revision_ids=active_ids,
        physical_unit_count=_physical_unit_count(adapter),
    )


def _corrupt_on_disk_bundle_without_changing_admission_digests(artifact_root: Path) -> None:
    marker = artifact_root / "site-packages" / "bundle_marker.py"
    marker.write_text('MARKER = "c-dim9-physical-corruption"\n', encoding="utf-8")


def _lifecycle_slot_released() -> bool:
    from intergrax.applications._shared import production_process_lifecycle as ppl

    active = ppl._active_production_process_lifecycle
    return active is None or active._process_closed


def _open_activation_bundle(
    d4,
    tmp_path: Path,
    *,
    environment,
    revision_id: str,
    marker: str,
):
    storage, adapter = d4._storage_and_adapter(tmp_path)
    governance = d4._governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    composition_builder = d4._composition_builder(tmp_path)
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
    ) as lifecycle:
        bundle, locator, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id=revision_id,
            marker=marker,
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        stores = lifecycle.composition.agent_platform_runtime.stores
    intent = StrictProductionActivationIntent(
        projection_input=bundle,
        activation_request=_activation_request(bundle, locator),
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    return storage, adapter, governance, composition_builder, stores, intent


def test_c_dim9_01_positive_strict_session_deploy_prepare_attest_handoff(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim9-positive"
    storage, adapter, governance, composition_builder, stores, intent = (
        _open_activation_bundle(
            d4,
            tmp_path,
            environment=environment,
            revision_id=revision_id,
            marker="dim9-positive",
        )
    )
    before = _authority_snapshot(stores, adapter, environment=environment)
    assert before.traffic_revision_id is None

    with patch.object(adapter, "prepare", wraps=adapter.prepare) as physical_prepare:
        session = open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=adapter,
            activation_intent=intent,
            composition_builder=composition_builder,
            missing_continuation_file_policy="empty",
        )
    try:
        assert physical_prepare.call_count == 1
        after = _authority_snapshot(stores, adapter, environment=environment)
        assert after.traffic_revision_id == revision_id
        assert after.serving_pointer_revision >= 1
        assert after.active_revision_ids == (revision_id,)
        assert session.readiness.ready_for_traffic is True
        assert_identity_continuity(session, revision_id=revision_id)
        handoff = session.serving_runtime_handoff
        assert handoff.colocated_execution_admission is not None
        instance = session.lifecycle.launcher.services.deployment_instance_store.get_instance(
            environment.application_id,
            environment.application_environment_id,
            revision_id,
        )
        assert instance is not None
        assert instance.instance_state in (
            DeploymentInstanceState.READY,
            DeploymentInstanceState.SERVING,
        )
        assert session.reconciliation.attestation is not None
        assert session.reconciliation.attestation.physical_unit_live is True
    finally:
        session.close()


def test_c_dim9_02_physical_prepare_failure_no_serving_promotion_or_handoff(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    storage, adapter, governance, composition_builder, stores, fail_intent = (
        _open_activation_bundle(
            d4,
            tmp_path,
            environment=environment,
            revision_id="rev-c-dim9-prepare-fail",
            marker="dim9-prepare-fail",
        )
    )
    fail_bundle = fail_intent.projection_input
    fail_revision_id = fail_bundle.runtime_revision.runtime_revision_id
    fail_locator = stores.materialization_store.get_by_revision(fail_revision_id)
    assert fail_locator is not None
    fail_root = resolve_production_artifact_root(fail_locator.artifact_locator)
    _corrupt_on_disk_bundle_without_changing_admission_digests(fail_root)

    before = _authority_snapshot(stores, adapter, environment=environment)
    prepare_revision_ids: list[str] = []
    real_prepare = adapter.prepare

    def _record_prepare(revision, **kwargs):
        prepare_revision_ids.append(revision.runtime_revision_id)
        return real_prepare(revision, **kwargs)

    failed_session = None
    with patch.object(adapter, "prepare", side_effect=_record_prepare):
        with pytest.raises(VenvBundleColocatedDeploymentError, match="digest"):
            failed_session = open_strict_production_serving_session(
                storage=storage,
                environment=environment,
                governance=governance,
                deployment_adapter=adapter,
                activation_intent=fail_intent,
                composition_builder=composition_builder,
                missing_continuation_file_policy="empty",
            )
    assert failed_session is None
    assert fail_revision_id in prepare_revision_ids
    after = _authority_snapshot(stores, adapter, environment=environment)
    assert after.traffic_revision_id == before.traffic_revision_id
    assert after.serving_pointer_revision == before.serving_pointer_revision
    assert after.active_revision_ids == before.active_revision_ids
    assert _lifecycle_slot_released()


def test_c_dim9_03_readiness_failure_after_physical_prepare_no_handoff(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim9-readiness-fail"
    storage, adapter, governance, composition_builder, stores, intent = (
        _open_activation_bundle(
            d4,
            tmp_path,
            environment=environment,
            revision_id=revision_id,
            marker="dim9-readiness-fail",
        )
    )
    before = _authority_snapshot(stores, adapter, environment=environment)
    units_before = _physical_unit_count(adapter)

    with patch.object(adapter, "prepare", wraps=adapter.prepare) as physical_prepare:
        with patch.object(
            adapter,
            "check_readiness",
            side_effect=RuntimeError("c-dim9-readiness-injected"),
        ):
            with pytest.raises(RuntimeReadinessError, match="candidate readiness validation failed"):
                open_strict_production_serving_session(
                    storage=storage,
                    environment=environment,
                    governance=governance,
                    deployment_adapter=adapter,
                    activation_intent=intent,
                    composition_builder=composition_builder,
                    missing_continuation_file_policy="empty",
                )
    assert physical_prepare.call_count >= 1
    assert _physical_unit_count(adapter) >= units_before
    after = _authority_snapshot(stores, adapter, environment=environment)
    assert after.traffic_revision_id == before.traffic_revision_id
    assert after.serving_pointer_revision == before.serving_pointer_revision
    assert after.active_revision_ids == before.active_revision_ids
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
    ) as lifecycle:
        instance = lifecycle.lifecycle_services.deployment_instance_store.get_instance(
            environment.application_id,
            environment.application_environment_id,
            revision_id,
        )
    assert instance is not None
    assert instance.instance_state is DeploymentInstanceState.FAILED
    assert instance.failure_evidence_ref is not None
    assert _lifecycle_slot_released()


def test_c_dim9_04_untrusted_physical_attestation_rejected_preserving_serving_state(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    storage, adapter, governance, composition_builder, stores, intent = (
        _open_activation_bundle(
            d4,
            tmp_path,
            environment=environment,
            revision_id="rev-c-dim9-attest-baseline",
            marker="dim9-attest-base",
        )
    )
    baseline_session = open_strict_production_serving_session(
        storage=storage,
        environment=environment,
        governance=governance,
        deployment_adapter=adapter,
        activation_intent=intent,
        composition_builder=composition_builder,
        missing_continuation_file_policy="empty",
    )
    baseline_revision = baseline_session.serving_runtime_handoff.identity.runtime_revision_id
    baseline_pointer = _authority_snapshot(stores, adapter, environment=environment)
    baseline_session.close()

    fresh_adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        allow_qualification_locators=True,
    )
    assert _physical_unit_count(fresh_adapter) == 0
    with pytest.raises(
        (StrictProductionServingLauncherError, ServingUnitAttestationError),
    ):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=fresh_adapter,
            composition_builder=composition_builder,
            missing_continuation_file_policy="empty",
        )
    after = _authority_snapshot(stores, fresh_adapter, environment=environment)
    assert after.traffic_revision_id == baseline_revision
    assert after.serving_pointer_revision == baseline_pointer.serving_pointer_revision
    assert after.active_revision_ids == (baseline_revision,)
    assert _lifecycle_slot_released()


def test_c_dim9_05_failed_strict_startup_cleans_process_slot_after_physical_unit(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim9-cleanup"
    storage, adapter, governance, composition_builder, stores, intent = (
        _open_activation_bundle(
            d4,
            tmp_path,
            environment=environment,
            revision_id=revision_id,
            marker="dim9-cleanup",
        )
    )
    units_before = _physical_unit_count(adapter)
    serving_unit_ref_holder: list[str] = []

    original_prepare = adapter.prepare

    def _prepare_and_capture(revision, **kwargs):
        prepared = original_prepare(revision, **kwargs)
        serving_unit_ref_holder.append(prepared.serving_unit_ref)
        return prepared

    with patch.object(adapter, "prepare", side_effect=_prepare_and_capture):
        with patch.object(
            adapter,
            "check_readiness",
            side_effect=RuntimeError("c-dim9-cleanup-readiness"),
        ):
            with pytest.raises(RuntimeReadinessError):
                open_strict_production_serving_session(
                    storage=storage,
                    environment=environment,
                    governance=governance,
                    deployment_adapter=adapter,
                    activation_intent=intent,
                    composition_builder=composition_builder,
                    missing_continuation_file_policy="empty",
                )
    assert len(serving_unit_ref_holder) == 1
    assert _physical_unit_count(adapter) == units_before + 1
    unit = adapter._units.get(serving_unit_ref_holder[0])
    assert unit is not None
    assert not unit.runtime_resource.is_live
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
    ) as lifecycle:
        failed_instance = lifecycle.lifecycle_services.deployment_instance_store.get_instance(
            environment.application_id,
            environment.application_environment_id,
            revision_id,
        )
    assert failed_instance is not None
    assert failed_instance.instance_state is DeploymentInstanceState.FAILED
    assert _traffic_revision_id(stores, environment) is None
    assert _lifecycle_slot_released()
