# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D0 — serving-unit attestation and READY reuse safety."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.agent_distribution.activation import (
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceRecord,
    DeploymentInstanceState,
    DrainPolicy,
    FakeInMemoryRuntimeDeploymentAdapter,
    PreparedDeployment,
    RuntimeDeploymentAdapter,
    ServingUnitAttestation,
    ServingUnitAttestationError,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.errors import RuntimeActivationError
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.materialization import MaterializationOutput
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService

_APP = "app-a"
_ENV = "env-prod"
_ARTIFACT = "sha256:" + ("e" * 64)
_LOCATOR = f"artifact://{_ARTIFACT}"
_OTHER_ARTIFACT = "sha256:" + ("f" * 64)


def _revision(
    revision_id: str,
    *,
    application_id: str = _APP,
    environment: str = _ENV,
    artifact_digest: str = _ARTIFACT,
    state: RuntimeRevisionState = RuntimeRevisionState.VALIDATED,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=application_id,
        application_environment_id=environment,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-hash",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="lock-digest",
        runtime_graph_digest="graph-digest",
        materialization_artifact_digest=artifact_digest,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
    )


def _persist_validated(
    revision_service: RuntimeRevisionService,
    revision_id: str,
    *,
    application_id: str = _APP,
    environment: str = _ENV,
    artifact_digest: str = _ARTIFACT,
) -> RuntimeRevision:
    revision_service.persist_candidate_revision(
        _revision(
            revision_id,
            application_id=application_id,
            environment=environment,
            artifact_digest=artifact_digest,
            state=RuntimeRevisionState.CANDIDATE,
        )
    )
    validated = _revision(
        revision_id,
        application_id=application_id,
        environment=environment,
        artifact_digest=artifact_digest,
    )
    revision_service.mark_validated(revision_id, validated_revision=validated)
    return validated


class _D0Harness:
    def __init__(self, deployment_adapter: FakeInMemoryRuntimeDeploymentAdapter | None = None) -> None:
        self.state = AgentDistributionStoreState()
        self.revision_store = InMemoryRuntimeRevisionStore(self.state)
        self.deployment_store = InMemoryDeploymentInstanceStore(self.state)
        self.serving_store = InMemoryApplicationEnvironmentServingStore(self.state)
        self.activation_store = InMemoryApplicationEnvironmentActivationStore(self.state)
        self.revision_service = RuntimeRevisionService(self.revision_store)
        self.deployment_adapter = deployment_adapter or FakeInMemoryRuntimeDeploymentAdapter()
        self.activation = ActivationService(
            revision_store=self.revision_store,
            deployment_instance_store=self.deployment_store,
            serving_store=self.serving_store,
            activation_store=self.activation_store,
            deployment_adapter=self.deployment_adapter,
            projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
        )

    def seed_validated(self, revision_id: str, **kwargs: str) -> None:
        _persist_validated(self.revision_service, revision_id, **kwargs)

    def prepare(self, revision_id: str, *, locator: str = _LOCATOR) -> DeploymentInstanceRecord:
        result = self.activation.prepare_candidate(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision_id,
            artifact_locator=locator,
        )
        return result.value

    def replace_adapter(self, adapter: FakeInMemoryRuntimeDeploymentAdapter) -> None:
        self.deployment_adapter = adapter
        self.activation = ActivationService(
            revision_store=self.revision_store,
            deployment_instance_store=self.deployment_store,
            serving_store=self.serving_store,
            activation_store=self.activation_store,
            deployment_adapter=adapter,
            projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
        )


def test_d0_01_valid_ready_reuse_after_attestation() -> None:
    harness = _D0Harness()
    harness.seed_validated("rev-1")
    first = harness.prepare("rev-1")
    second = harness.prepare("rev-1")
    assert second.instance_state is DeploymentInstanceState.READY
    assert second.serving_unit_ref == first.serving_unit_ref
    assert second.record_revision == first.record_revision
    assert harness.deployment_adapter.prepare_count.get("rev-1") == 1


def test_d0_02_persisted_ready_empty_adapter_after_restart_fails() -> None:
    harness = _D0Harness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.replace_adapter(FakeInMemoryRuntimeDeploymentAdapter())
    with pytest.raises(RuntimeActivationError, match="attestation failed"):
        harness.prepare("rev-1")


def test_d0_03_unknown_serving_unit_reference_fails_closed() -> None:
    harness = _D0Harness()
    harness.seed_validated("rev-1")
    ready = harness.prepare("rev-1")
    ghost = ready.model_copy(update={"serving_unit_ref": "ghost-unit"})
    harness.deployment_store.persist_instance(ghost)
    with pytest.raises(RuntimeActivationError, match="attestation failed"):
        harness.prepare("rev-1")


def test_d0_04_wrong_artifact_locator_fails_closed() -> None:
    harness = _D0Harness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    with pytest.raises(RuntimeActivationError, match="attestation failed"):
        harness.prepare("rev-1", locator=f"artifact://{_OTHER_ARTIFACT}")


def test_d0_05_wrong_runtime_revision_in_store_fails_closed() -> None:
    harness = _D0Harness()
    harness.seed_validated("rev-1")
    harness.seed_validated("rev-2")
    ready = harness.prepare("rev-1")
    stale_cross_revision = ready.model_copy(update={"runtime_revision_id": "rev-2"})
    harness.deployment_store.persist_instance(stale_cross_revision)
    with pytest.raises(RuntimeActivationError, match="attestation failed"):
        harness.prepare("rev-2")


def test_d0_06_wrong_application_environment_identity_fails_closed() -> None:
    harness = _D0Harness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    with pytest.raises(RuntimeActivationError, match="environment mismatch"):
        harness.activation.prepare_candidate(
            application_id=_APP,
            application_environment_id="env-other",
            runtime_revision_id="rev-1",
            artifact_locator=_LOCATOR,
        )


def test_d0_07_mismatched_attestation_response_rejected() -> None:
    class _LieAdapter(FakeInMemoryRuntimeDeploymentAdapter):
        def attest_serving_unit(
            self,
            revision: RuntimeRevision,
            *,
            request: ServingUnitAttestationRequest,
        ) -> ServingUnitAttestation:
            base = super().attest_serving_unit(revision, request=request)
            return base.model_copy(update={"runtime_revision_id": "rev-lie"})

    harness = _D0Harness(_LieAdapter())
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    with pytest.raises(RuntimeActivationError, match="runtime_revision_id mismatch"):
        harness.prepare("rev-1")


def test_d0_08_adapter_attestation_error_preserves_cause() -> None:
    class _RejectAdapter(FakeInMemoryRuntimeDeploymentAdapter):
        def attest_serving_unit(
            self,
            revision: RuntimeRevision,
            *,
            request: ServingUnitAttestationRequest,
        ) -> ServingUnitAttestation:
            raise ServingUnitAttestationError("injected", reason_code="injected")

    harness = _D0Harness(_RejectAdapter())
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    with pytest.raises(RuntimeActivationError, match="attestation failed") as exc_info:
        harness.prepare("rev-1")
    assert isinstance(exc_info.value.__cause__, ServingUnitAttestationError)


def test_d0_09_no_serving_pointer_mutation_on_failed_attestation() -> None:
    harness = _D0Harness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.activation.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-1",
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=_ARTIFACT,
    )
    before = harness.serving_store.get_serving_record(_APP, _ENV)
    harness.replace_adapter(FakeInMemoryRuntimeDeploymentAdapter())
    with pytest.raises(RuntimeActivationError):
        harness.prepare("rev-1")
    after = harness.serving_store.get_serving_record(_APP, _ENV)
    assert after is not None
    assert before is not None
    assert after.traffic_serving_revision_id == before.traffic_serving_revision_id
    assert after.serving_pointer_revision == before.serving_pointer_revision


def test_d0_10_no_automatic_prepare_after_failed_attestation() -> None:
    harness = _D0Harness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    harness.replace_adapter(FakeInMemoryRuntimeDeploymentAdapter())
    with pytest.raises(RuntimeActivationError):
        harness.prepare("rev-1")
    assert harness.deployment_adapter.prepare_count.get("rev-1", 0) == 0


class _MinimalAttestingAdapter:
    """Alternate RuntimeDeploymentAdapter without subclassing the fake."""

    def __init__(self) -> None:
        self._live: dict[str, ServingUnitAttestationRequest] = {}

    def prepare(
        self,
        revision: RuntimeRevision,
        *,
        artifact_locator: str,
        materialization_output: MaterializationOutput | None = None,
    ) -> PreparedDeployment:
        digest = revision.materialization_artifact_digest
        if digest is None:
            raise RuntimeError("missing digest")
        ref = f"alt:{revision.runtime_revision_id}"
        self._live[ref] = ServingUnitAttestationRequest(
            application_id=revision.application_id,
            application_environment_id=revision.application_environment_id,
            runtime_revision_id=revision.runtime_revision_id,
            serving_unit_ref=ref,
            materialization_artifact_digest=digest,
            artifact_locator=artifact_locator,
        )
        return PreparedDeployment(serving_unit_ref=ref, materialization_artifact_digest=digest)

    def check_readiness(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> str:
        return f"ready:{serving_unit_ref}"

    def begin_drain(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> None:
        return None

    def check_drain(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
        policy: DrainPolicy,
    ):
        from intergrax.agent_distribution.deployment import DrainStatus

        return DrainStatus(completed=True)

    def stop(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> None:
        self._live.pop(serving_unit_ref, None)

    def resume_serving(
        self,
        revision: RuntimeRevision,
        *,
        serving_unit_ref: str,
    ) -> str:
        return f"resume:{serving_unit_ref}"

    def attest_serving_unit(
        self,
        revision: RuntimeRevision,
        *,
        request: ServingUnitAttestationRequest,
    ) -> ServingUnitAttestation:
        bound = self._live.get(request.serving_unit_ref)
        if bound is None:
            raise ServingUnitAttestationError("unknown")
        return ServingUnitAttestation(
            application_id=bound.application_id,
            application_environment_id=bound.application_environment_id,
            serving_unit_ref=bound.serving_unit_ref,
            runtime_revision_id=bound.runtime_revision_id,
            materialization_artifact_digest=bound.materialization_artifact_digest,
            physical_unit_live=True,
            evidence_ref="alt-evidence",
            attested_at=datetime.now(UTC),
        )


def test_d0_11_alternate_adapter_satisfies_protocol() -> None:
    adapter: RuntimeDeploymentAdapter = _MinimalAttestingAdapter()
    harness = _D0Harness()
    harness.deployment_adapter = adapter  # type: ignore[assignment]
    harness.activation = ActivationService(
        revision_store=harness.revision_store,
        deployment_instance_store=harness.deployment_store,
        serving_store=harness.serving_store,
        activation_store=harness.activation_store,
        deployment_adapter=adapter,
        projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
    )
    harness.seed_validated("rev-alt")
    ready = harness.prepare("rev-alt")
    assert ready.serving_unit_ref.startswith("alt:")
    reused = harness.prepare("rev-alt")
    assert reused.serving_unit_ref == ready.serving_unit_ref


def test_d0_13_deployment_module_has_no_applications_imports() -> None:
    module_path = Path("intergrax/agent_distribution/deployment.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "applications" not in alias.name
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "applications" not in module


def test_d0_14_attestation_does_not_grant_traffic_authority() -> None:
    harness = _D0Harness()
    harness.seed_validated("rev-1")
    harness.prepare("rev-1")
    assert harness.serving_store.get_serving_record(_APP, _ENV) is None
    harness.replace_adapter(FakeInMemoryRuntimeDeploymentAdapter())
    with pytest.raises(RuntimeActivationError, match="attestation failed"):
        harness.prepare("rev-1")
    assert harness.serving_store.get_serving_record(_APP, _ENV) is None
