# © Artur Czarnecki. All rights reserved.

"""R7-Q §C dimension 7 — integrated strict-path adversarial gate (DENY, stale, CAS, idempotency)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from unittest.mock import patch

import pytest

from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceState,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.errors import RuntimeActivationConflict
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.production_process_lifecycle import (
    open_production_process_lifecycle,
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
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    assert_identity_continuity,
    load_d4_launcher_module,
    repo_root,
)
from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
    _DenyEvaluator,
    _governance as d2_governance,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _activation_request,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "4a4b42ce54556360a95ab56cec67f03150f1bfef"


def test_c_dim7_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


@dataclass(slots=True)
class ServingAuthoritySnapshot:
    traffic_revision_id: str | None
    serving_pointer_revision: int
    active_revision_count: int


def _serving_snapshot(session, *, environment) -> ServingAuthoritySnapshot:
    stores = session.composition.agent_platform_runtime.stores
    serving = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    revisions = stores.revision_store.list_revisions_for_environment(
        environment.application_id,
        environment.application_environment_id,
    )
    active_count = sum(
        1 for r in revisions if r.revision_state is RuntimeRevisionState.ACTIVE
    )
    return ServingAuthoritySnapshot(
        traffic_revision_id=(
            serving.traffic_serving_revision_id if serving is not None else None
        ),
        serving_pointer_revision=(
            serving.serving_pointer_revision if serving is not None else 0
        ),
        active_revision_count=active_count,
    )


@dataclass(slots=True)
class GovernanceDenyEvidence:
    evaluator: _DenyEvaluator = field(default_factory=_DenyEvaluator)

    def build(self, environment) -> ProductionControlPlaneGovernance:
        env = ApplicationEnvironmentProfile(profile_id=environment.application_environment_id)
        base = d2_governance(env)
        return ProductionControlPlaneGovernance(
            principal=base.principal,
            mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
                evaluator=self.evaluator,
            ),
            environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
                tenant_id=env.profile_id,
            ),
        )


def test_c_dim7_01_positive_strict_activation_prepare_attest_and_handoff(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    revision_id = "rev-c-dim7-positive"
    session = d4._open_with_activation(
        tmp_path,
        environment=environment,
        revision_id=revision_id,
    )
    try:
        snap = _serving_snapshot(session, environment=environment)
        assert snap.traffic_revision_id == revision_id
        assert snap.serving_pointer_revision >= 1
        assert snap.active_revision_count == 1
        assert session.readiness.ready_for_traffic is True
        assert_identity_continuity(session, revision_id=revision_id)
        handoff = session.serving_runtime_handoff
        assert handoff is not None
        identity = handoff.identity
        stores = session.composition.agent_platform_runtime.stores
        record = stores.materialization_store.get_by_revision(revision_id)
        revision = stores.revision_store.get_revision(revision_id)
        assert record is not None and revision is not None
        assert revision.revision_state is RuntimeRevisionState.ACTIVE
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
        assert instance.serving_unit_ref == identity.serving_unit_ref
        gate_adapter = handoff.colocated_execution_admission.physical_gate.adapter
        attestation = gate_adapter.attest_serving_unit(
            revision,
            request=ServingUnitAttestationRequest(
                application_id=identity.application_id,
                application_environment_id=identity.application_environment_id,
                runtime_revision_id=identity.runtime_revision_id,
                serving_unit_ref=identity.serving_unit_ref,
                materialization_artifact_digest=identity.materialization_artifact_digest,
                artifact_locator=record.artifact_locator,
            ),
        )
        assert attestation.physical_unit_live is True
    finally:
        session.close()


def test_c_dim7_02_governance_deny_before_physical_prepare(tmp_path: Path) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    deny_evidence = GovernanceDenyEvidence()
    governance = deny_evidence.build(environment)
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
            revision_id="rev-c-dim7-deny",
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
    with patch.object(adapter, "prepare", wraps=adapter.prepare) as physical_prepare:
        with pytest.raises(StrictProductionServingLauncherError):
            open_strict_production_serving_session(
                storage=storage,
                environment=environment,
                governance=governance,
                deployment_adapter=adapter,
                activation_intent=intent,
                composition_builder=composition_builder,
                missing_continuation_file_policy="empty",
            )
    assert physical_prepare.call_count == 0
    assert len(deny_evidence.evaluator.calls) >= 1
    serving = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    assert serving is None


def test_c_dim7_03_stale_prior_traffic_rejected_without_serving_promotion(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
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
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        stores = lifecycle.composition.agent_platform_runtime.stores
        bundle_a, locator_a, *_ = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-dim7-stale-a",
            marker="dim7-a",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=stores,
        )
        d4._activate_bundle(
            lifecycle.launcher,
            bundle_a,
            locator_a,
            principal=governance.principal,
        )
        before = stores.serving_store.get_serving_record(
            environment.application_id,
            environment.application_environment_id,
        )
        assert before is not None
        bundle_b, locator_b, *_ = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-dim7-stale-b",
            marker="dim7-b",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=stores,
        )
        stale_request = ActivateRuntimeRevisionRequest(
            mutation_id="activate:rev-c-dim7-stale-b",
            runtime_revision_id=bundle_b.runtime_revision.runtime_revision_id,
            artifact_locator=locator_b,
            expected_artifact_digest=bundle_b.materialization_artifact_digest,
            expected_serving_pointer_revision=before.serving_pointer_revision,
            expected_prior_traffic_revision_id="rev-nonexistent-prior",
        )
        intent = StrictProductionActivationIntent(
            projection_input=bundle_b,
            activation_request=stale_request,
            admission_mutation_id=reference_admission_mutation_id(
                bundle_b.runtime_revision.runtime_revision_id,
            ),
        )
    with patch.object(adapter, "prepare", wraps=adapter.prepare) as physical_prepare:
        with pytest.raises(
            (RuntimeActivationConflict, StrictProductionServingLauncherError),
        ):
            open_strict_production_serving_session(
                storage=storage,
                environment=environment,
                governance=governance,
                deployment_adapter=adapter,
                activation_intent=intent,
                composition_builder=composition_builder,
                missing_continuation_file_policy="empty",
            )
    prepare_calls_for_b = [
        c
        for c in physical_prepare.call_args_list
        if c.args and c.args[0].runtime_revision_id == "rev-c-dim7-stale-b"
    ]
    after = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    assert after is not None
    assert after.traffic_serving_revision_id == before.traffic_serving_revision_id
    assert after.serving_pointer_revision == before.serving_pointer_revision
    active_b = stores.revision_store.get_revision(
        "rev-c-dim7-stale-b",
    )
    assert active_b is not None
    assert active_b.revision_state is not RuntimeRevisionState.ACTIVE
    if prepare_calls_for_b:
        assert active_b.revision_state is not RuntimeRevisionState.ACTIVE


def test_c_dim7_04_cas_stale_pointer_rejected_single_active_authority(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
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
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        stores = lifecycle.composition.agent_platform_runtime.stores
        bundle_a, locator_a, *_ = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-dim7-cas-a",
            marker="dim7-cas-a",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=stores,
        )
        d4._activate_bundle(
            lifecycle.launcher,
            bundle_a,
            locator_a,
            principal=governance.principal,
        )
        before = stores.serving_store.get_serving_record(
            environment.application_id,
            environment.application_environment_id,
        )
        assert before is not None
        assert before.serving_pointer_revision >= 1
        bundle_b, locator_b, *_ = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-dim7-cas-b",
            marker="dim7-cas-b",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=stores,
        )
        cas_stale_request = ActivateRuntimeRevisionRequest(
            mutation_id="activate:rev-c-dim7-cas-b",
            runtime_revision_id=bundle_b.runtime_revision.runtime_revision_id,
            artifact_locator=locator_b,
            expected_artifact_digest=bundle_b.materialization_artifact_digest,
            expected_serving_pointer_revision=0,
            expected_prior_traffic_revision_id=before.traffic_serving_revision_id,
        )
        intent = StrictProductionActivationIntent(
            projection_input=bundle_b,
            activation_request=cas_stale_request,
            admission_mutation_id=reference_admission_mutation_id(
                bundle_b.runtime_revision.runtime_revision_id,
            ),
        )
    with pytest.raises(
        (RuntimeActivationConflict, StrictProductionServingLauncherError),
        match=(
            "stale serving pointer|activation atomic commit|"
            "serving pointer revision does not match expected value"
        ),
    ):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=adapter,
            activation_intent=intent,
            composition_builder=composition_builder,
            missing_continuation_file_policy="empty",
        )
    after = stores.serving_store.get_serving_record(
        environment.application_id,
        environment.application_environment_id,
    )
    assert after is not None
    assert after.traffic_serving_revision_id == before.traffic_serving_revision_id
    assert after.serving_pointer_revision == before.serving_pointer_revision
    active_revisions = [
        r
        for r in stores.revision_store.list_revisions_for_environment(
            environment.application_id,
            environment.application_environment_id,
        )
        if r.revision_state is RuntimeRevisionState.ACTIVE
    ]
    assert len(active_revisions) == 1
    assert active_revisions[0].runtime_revision_id == before.traffic_serving_revision_id


def test_c_dim7_05_idempotent_replay_no_duplicate_physical_prepare(tmp_path: Path) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
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
        bundle, locator, *_ = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-dim7-idempotent",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        stores = lifecycle.composition.agent_platform_runtime.stores
        activation = lifecycle.launcher.services.activation_service
        revision_id = bundle.runtime_revision.runtime_revision_id
        with patch.object(adapter, "prepare", wraps=adapter.prepare) as physical_prepare:
            activation.prepare_candidate(
                application_id=environment.application_id,
                application_environment_id=environment.application_environment_id,
                runtime_revision_id=revision_id,
                artifact_locator=locator,
            )
            first_prepare_count = physical_prepare.call_count
            assert first_prepare_count >= 1
            activation.prepare_candidate(
                application_id=environment.application_id,
                application_environment_id=environment.application_environment_id,
                runtime_revision_id=revision_id,
                artifact_locator=locator,
            )
        assert physical_prepare.call_count == first_prepare_count
        serving = stores.serving_store.get_serving_record(
            environment.application_id,
            environment.application_environment_id,
        )
        assert serving is None
