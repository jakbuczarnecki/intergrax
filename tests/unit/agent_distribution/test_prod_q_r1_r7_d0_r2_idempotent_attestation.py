# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D0-R2 — idempotent commit/rollback attestation closure."""

from __future__ import annotations

import threading

import pytest

from intergrax.agent_distribution.activation import ActivationService
from intergrax.agent_distribution.deployment import (
    FakeInMemoryRuntimeDeploymentAdapter,
    ServingUnitAttestation,
    ServingUnitAttestationError,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.errors import RuntimeActivationError, RuntimeRollbackError
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from tests.unit.agent_distribution.test_agent_distribution_activation import (
    ActivationHarness,
)
from tests.unit.agent_distribution.test_agent_distribution_rollback import (
    _bootstrap_active_pair,
)

_APP = "app-a"
_ENV = "env-prod"
_ARTIFACT = "sha256:" + ("e" * 64)
_LOCATOR = f"artifact://{_ARTIFACT}"
_OTHER_LOCATOR = f"artifact://sha256:{'f' * 64}"


def _snapshot_serving(harness: ActivationHarness) -> dict[str, object] | None:
    rec = harness.serving_store.get_serving_record(_APP, _ENV)
    return rec.model_dump() if rec is not None else None


def test_r2_01_idempotent_commit_live_unit_no_pointer_mutation() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    before = _snapshot_serving(harness)
    second = harness.commit("rev-1", pointer=0)
    after = _snapshot_serving(harness)
    assert before == after
    assert second.value.serving_record.traffic_serving_revision_id == "rev-1"


def test_r2_02_idempotent_commit_stale_adapter_after_restart_fails() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.deployment_adapter = FakeInMemoryRuntimeDeploymentAdapter()
    harness.activation = ActivationService(
        revision_store=harness.revision_store,
        deployment_instance_store=harness.deployment_store,
        serving_store=harness.serving_store,
        activation_store=harness.activation_store,
        deployment_adapter=harness.deployment_adapter,
        projection_coordinator=harness.projection,
        materialization_store=harness.materialization_store,
    )
    with pytest.raises(RuntimeActivationError, match="attestation failed"):
        harness.commit("rev-1", pointer=0)


def test_r2_03_idempotent_rollback_live_restored_unit() -> None:
    activation, _, _ = _bootstrap_active_pair()
    first = activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    second = activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    assert first.value.serving_record.traffic_serving_revision_id == second.value.serving_record.traffic_serving_revision_id


def test_r2_04_idempotent_rollback_after_unit_loss_fails() -> None:
    activation, _, adapter = _bootstrap_active_pair()
    activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    adapter._units.clear()
    with pytest.raises(RuntimeActivationError, match="attestation failed"):
        activation.rollback(
            application_id=_APP,
            application_environment_id=_ENV,
            expected_current_traffic_revision_id="rev-n1",
            expected_serving_pointer_revision=2,
        )


def test_r2_05_idempotent_commit_missing_materialization_fails_closed() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.state.materializations.clear()
    with pytest.raises(RuntimeActivationError, match="canonical materialization authority missing"):
        harness.commit("rev-1", pointer=0)


def test_r2_06_idempotent_rollback_missing_materialization_fails_closed() -> None:
    activation, _, _ = _bootstrap_active_pair()
    activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    activation._materialization_store._state.materializations.clear()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeRollbackError, match="canonical materialization authority missing"):
        activation.rollback(
            application_id=_APP,
            application_environment_id=_ENV,
            expected_current_traffic_revision_id="rev-n1",
            expected_serving_pointer_revision=2,
        )


def test_r2_07_idempotent_rollback_wrong_locator_for_restored_revision_fails() -> None:
    activation, _, _ = _bootstrap_active_pair()
    activation.rollback(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id="rev-n1",
        expected_serving_pointer_revision=2,
    )
    state = activation._materialization_store._state  # type: ignore[attr-defined]
    record = state.materializations["rev-n"]
    state.materializations["rev-n"] = record.model_copy(update={"artifact_locator": _OTHER_LOCATOR})
    with pytest.raises(RuntimeActivationError, match="attestation failed"):
        activation.rollback(
            application_id=_APP,
            application_environment_id=_ENV,
            expected_current_traffic_revision_id="rev-n1",
            expected_serving_pointer_revision=2,
        )


def test_r2_08_idempotent_commit_wrong_environment_fails_closed() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    with pytest.raises(RuntimeActivationError, match="environment mismatch"):
        harness.activation.commit_activation(
            application_id=_APP,
            application_environment_id="env-other",
            runtime_revision_id="rev-1",
            expected_prior_traffic_revision_id=None,
            expected_serving_pointer_revision=0,
            expected_artifact_digest=_ARTIFACT,
        )


def test_r2_09_idempotent_commit_mismatched_attestation_identity_rejected() -> None:
    class _LieAdapter(FakeInMemoryRuntimeDeploymentAdapter):
        def attest_serving_unit(
            self,
            revision: RuntimeRevision,
            *,
            request: ServingUnitAttestationRequest,
        ) -> ServingUnitAttestation:
            base = super().attest_serving_unit(revision, request=request)
            return base.model_copy(update={"application_id": "wrong-app"})

    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    donor = harness.deployment_adapter
    lie = _LieAdapter()
    lie._units = dict(donor._units)
    harness.deployment_adapter = lie
    harness.activation = ActivationService(
        revision_store=harness.revision_store,
        deployment_instance_store=harness.deployment_store,
        serving_store=harness.serving_store,
        activation_store=harness.activation_store,
        deployment_adapter=lie,
        projection_coordinator=harness.projection,
        materialization_store=harness.materialization_store,
    )
    with pytest.raises(RuntimeActivationError, match="application_id mismatch"):
        harness.commit("rev-1", pointer=0)


def test_r2_10_idempotent_commit_adapter_error_preserves_cause() -> None:
    class _RejectAdapter(FakeInMemoryRuntimeDeploymentAdapter):
        def attest_serving_unit(
            self,
            revision: RuntimeRevision,
            *,
            request: ServingUnitAttestationRequest,
        ) -> ServingUnitAttestation:
            raise ServingUnitAttestationError("injected", reason_code="injected")

    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    harness.deployment_adapter = _RejectAdapter()
    harness.activation = ActivationService(
        revision_store=harness.revision_store,
        deployment_instance_store=harness.deployment_store,
        serving_store=harness.serving_store,
        activation_store=harness.activation_store,
        deployment_adapter=harness.deployment_adapter,
        projection_coordinator=harness.projection,
        materialization_store=harness.materialization_store,
    )
    with pytest.raises(RuntimeActivationError, match="attestation failed") as exc_info:
        harness.commit("rev-1", pointer=0)
    assert isinstance(exc_info.value.__cause__, ServingUnitAttestationError)


def test_r2_11_failed_idempotent_commit_attest_no_serving_mutation() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    before = _snapshot_serving(harness)
    harness.deployment_adapter = FakeInMemoryRuntimeDeploymentAdapter()
    harness.activation = ActivationService(
        revision_store=harness.revision_store,
        deployment_instance_store=harness.deployment_store,
        serving_store=harness.serving_store,
        activation_store=harness.activation_store,
        deployment_adapter=harness.deployment_adapter,
        projection_coordinator=harness.projection,
        materialization_store=harness.materialization_store,
    )
    with pytest.raises(RuntimeActivationError):
        harness.commit("rev-1", pointer=0)
    assert _snapshot_serving(harness) == before


def test_r2_12_failed_idempotent_commit_no_prepare_resume_retry() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    prepare_before = harness.deployment_adapter.prepare_count.get("rev-1", 0)
    harness.deployment_adapter = FakeInMemoryRuntimeDeploymentAdapter()
    harness.activation = ActivationService(
        revision_store=harness.revision_store,
        deployment_instance_store=harness.deployment_store,
        serving_store=harness.serving_store,
        activation_store=harness.activation_store,
        deployment_adapter=harness.deployment_adapter,
        projection_coordinator=harness.projection,
        materialization_store=harness.materialization_store,
    )
    with pytest.raises(RuntimeActivationError):
        harness.commit("rev-1", pointer=0)
    assert harness.deployment_adapter.prepare_count.get("rev-1", 0) == 0
    assert prepare_before == 1


def test_r2_13_healthy_idempotent_commit_and_rollback_unchanged() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    first = harness.commit("rev-1")
    second = harness.commit("rev-1", pointer=0)
    assert first.value.serving_record.serving_pointer_revision == second.value.serving_record.serving_pointer_revision
    test_r2_03_idempotent_rollback_live_restored_unit()


def test_r2_14_concurrent_idempotent_commit_cannot_bypass_authoritative_checks() -> None:
    harness = ActivationHarness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.commit("rev-1")
    errors: list[BaseException] = []
    successes: list[str] = []

    def stale_digest_retry() -> None:
        try:
            harness.activation.commit_activation(
                application_id=_APP,
                application_environment_id=_ENV,
                runtime_revision_id="rev-1",
                expected_prior_traffic_revision_id=None,
                expected_serving_pointer_revision=0,
                expected_artifact_digest="sha256:" + ("0" * 64),
            )
        except BaseException as exc:
            errors.append(exc)

    def idempotent_retry() -> None:
        try:
            harness.commit("rev-1", pointer=0)
            successes.append("ok")
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=stale_digest_retry)
    t2 = threading.Thread(target=idempotent_retry)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert any(isinstance(e, RuntimeActivationError) for e in errors)
    assert successes == ["ok"]
    serving = harness.serving_store.get_serving_record(_APP, _ENV)
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-1"


def test_r2_15_d0_r1_locator_regression_still_green() -> None:
    from tests.unit.agent_distribution.test_prod_q_r1_r7_d0_serving_unit_attestation import (
        test_r1_01_prepare_file_locator_commit_without_locator_mismatch,
        test_r1_07_missing_materialization_authority_fails_closed,
    )

    test_r1_01_prepare_file_locator_commit_without_locator_mismatch()
    test_r1_07_missing_materialization_authority_fails_closed()
