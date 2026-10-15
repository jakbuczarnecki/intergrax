# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P3-INTEGRATION-GATE — IG-E01 (strict activation intent, failed prepare).

Status: IMPLEMENTED — NOT EXECUTED — AWAITING OPERATOR AUTHORIZATION.
"""

from __future__ import annotations

import types
from typing import Final

import pytest

from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.production_process_lifecycle import (
    open_production_process_lifecycle,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionActivationIntent,
    open_strict_production_serving_session,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    load_d4_launcher_module,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _APP,
    _ENV,
    _activation_request,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_NODE_ID: Final = (
    "tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_integration_gate.py"
    "::test_ig_e01_strict_activation_intent_fail_prepare_no_handoff"
)


def _traffic_revision_id(stores) -> str | None:
    record = stores.serving_store.get_serving_record(_APP, _ENV)
    if record is None:
        return None
    return record.traffic_serving_revision_id


def _attach_prepare_call_spy(
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
) -> list[int]:
    """Non-invasive call counter on the real A1 ``prepare`` implementation."""
    calls: list[int] = [0]
    real_prepare = VenvBundleColocatedRuntimeDeploymentAdapter.prepare

    def _spy_prepare(
        self: VenvBundleColocatedRuntimeDeploymentAdapter,
        revision,
        *,
        artifact_locator: str,
        materialization_output=None,
    ):
        calls[0] += 1
        return real_prepare(
            self,
            revision,
            artifact_locator=artifact_locator,
            materialization_output=materialization_output,
        )

    adapter.prepare = types.MethodType(_spy_prepare, adapter)
    return calls


def _corrupt_on_disk_bundle_without_changing_admission_digests(artifact_root) -> None:
    """Leave projection / activation digests consistent; invalidate physical bytes."""
    marker = artifact_root / "site-packages" / "bundle_marker.py"
    marker.write_text('MARKER = "ig-e01-physical-corruption"\n', encoding="utf-8")


def test_ig_e01_strict_activation_intent_fail_prepare_no_handoff(tmp_path) -> None:
    """IG-E01: real A1 prepare failure on strict activation_intent path — no handoff/session."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"

    environment = d4._ac3_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    governance = d4._governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    composition_builder = d4._composition_builder(tmp_path)
    prepare_calls = _attach_prepare_call_spy(adapter)

    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        good_bundle, good_locator, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-ig-e01-good",
            marker="ig-e01-good",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        d4._activate_bundle(
            lifecycle.launcher,
            good_bundle,
            good_locator,
            principal=governance.principal,
        )
        stores = lifecycle.composition.agent_platform_runtime.stores

    traffic_before = _traffic_revision_id(stores)
    assert traffic_before == "rev-ig-e01-good"

    fail_bundle = None
    fail_locator = None
    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        fail_bundle, fail_locator, fail_artifact_root, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-ig-e01-fail",
            marker="ig-e01-fail",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        _corrupt_on_disk_bundle_without_changing_admission_digests(fail_artifact_root)

    assert fail_bundle is not None and fail_locator is not None
    fail_intent = StrictProductionActivationIntent(
        projection_input=fail_bundle,
        activation_request=_activation_request(fail_bundle, fail_locator),
        admission_mutation_id=reference_admission_mutation_id(
            fail_bundle.runtime_revision.runtime_revision_id,
        ),
    )

    failed_session = None
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
    assert prepare_calls[0] >= 1, "IG-E01 requires real A1 adapter.prepare invocation"

    traffic_after_fail = _traffic_revision_id(stores)
    assert traffic_after_fail == traffic_before

    healthy_session = open_strict_production_serving_session(
        storage=storage,
        environment=environment,
        governance=governance,
        deployment_adapter=adapter,
        composition_builder=composition_builder,
        missing_continuation_file_policy="empty",
    )
    try:
        assert healthy_session.readiness.ready_for_traffic is True
        assert healthy_session.serving_runtime_handoff is not None
        assert (
            healthy_session.composition.serving_runtime_handoff
            is healthy_session.serving_runtime_handoff
        )
        handoff = healthy_session.serving_runtime_handoff
        assert handoff.colocated_execution_admission is not None
        assert handoff.identity.runtime_revision_id == traffic_before
    finally:
        healthy_session.close()

    assert _NODE_ID.endswith("test_ig_e01_strict_activation_intent_fail_prepare_no_handoff")
