# © Artur Czarnecki. All rights reserved.

"""R7-Q §C — dimensions 2 + 3 integrated strict-path qualification slice.

Dimension 2: immutable artifact identity + digest validation on strict deploy/activate.
Dimension 3: serving-unit liveness attestation on the traffic-serving unit after full session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from intergrax.agent_distribution.deployment import (
    ServingUnitAttestationError,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    ProductionRegistryProjectionInputError,
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

_EXPECTED_QUALIFICATION_BASELINE: Final = (
    "775605893f4bb58c6a52a17eb68b2cac789100b1"
)


def test_c_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_QUALIFICATION_BASELINE, "HEAD"],
        cwd=repo_root(),
    )


def _corrupt_bundle_bytes(artifact_root: Path) -> None:
    marker = artifact_root / "site-packages" / "bundle_marker.py"
    marker.write_text('MARKER = "dim2-digest-corruption"\n', encoding="utf-8")


def test_c_dim2_01_positive_strict_session_digest_and_locator_resolvable(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    revision_id = "rev-c-dim2-positive"
    session = d4._open_with_activation(
        tmp_path,
        environment=d4._ac3_environment(),
        revision_id=revision_id,
    )
    try:
        handoff = session.serving_runtime_handoff
        assert handoff is not None
        identity = handoff.identity
        assert identity.materialization_artifact_digest.startswith("sha256:")
        assert_identity_continuity(session, revision_id=revision_id)
        stores = session.composition.agent_platform_runtime.stores
        record = stores.materialization_store.get_by_revision(identity.runtime_revision_id)
        assert record is not None
        resolved = resolve_production_artifact_root(record.artifact_locator)
        assert resolved.is_dir()
        assert record.materialization_artifact_digest == identity.materialization_artifact_digest
    finally:
        session.close()


def test_c_dim2_02_negative_missing_artifact_fail_closed(tmp_path: Path) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    governance = d4._governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    gca_kwargs = d4._lifecycle_gca_kwargs(storage)
    composition_builder = gca_kwargs["composition_builder"]
    from intergrax.applications._shared.production_process_lifecycle import (
        open_production_process_lifecycle,
    )

    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
        owned_governed_external_work_host_store_bundle=gca_kwargs[
            "owned_governed_external_work_host_store_bundle"
        ],
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        bundle, locator, artifact_root, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-dim2-missing",
            marker="dim2-missing",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
    import shutil

    shutil.rmtree(artifact_root)
    intent = StrictProductionActivationIntent(
        projection_input=bundle,
        activation_request=_activation_request(bundle, locator),
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    with pytest.raises(VenvBundleColocatedDeploymentError):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=adapter,
            activation_intent=intent,
            **gca_kwargs,
            missing_continuation_file_policy="empty",
        )


def test_c_dim2_03_negative_wrong_digest_fail_closed_on_strict_activation(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    governance = d4._governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    gca_kwargs = d4._lifecycle_gca_kwargs(storage)
    composition_builder = gca_kwargs["composition_builder"]
    from intergrax.applications._shared.production_process_lifecycle import (
        open_production_process_lifecycle,
    )

    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
        owned_governed_external_work_host_store_bundle=gca_kwargs[
            "owned_governed_external_work_host_store_bundle"
        ],
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        bundle, locator, artifact_root, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-dim2-wrong-digest",
            marker="dim2-wrong-digest",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        _corrupt_bundle_bytes(artifact_root)
    intent = StrictProductionActivationIntent(
        projection_input=bundle,
        activation_request=_activation_request(bundle, locator),
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="digest"):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=adapter,
            activation_intent=intent,
            **gca_kwargs,
            missing_continuation_file_policy="empty",
        )


def test_c_dim2_04_negative_reference_locator_forbidden() -> None:
    with pytest.raises(ProductionRegistryProjectionInputError, match="reference://"):
        resolve_production_artifact_root("reference://artifacts/forbidden")


def test_c_dim2_05_negative_untrusted_artifact_root_fail_closed(tmp_path: Path) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    trusted_parent = tmp_path / "trusted"
    trusted_parent.mkdir()
    from intergrax.applications._shared.strict_production_serving_launcher import (
        StrictProductionProcessStorage,
    )

    roots = (trusted_parent,)
    storage = StrictProductionProcessStorage(
        agent_distribution_db=tmp_path / "ad-untrusted.db",
        continuation_state_file=tmp_path / "continuation-untrusted.json",
        provider_invocation_db=tmp_path / "invocation-untrusted.db",
        governed_external_work_host_db=tmp_path / "gca_host-untrusted.db",
        trusted_artifact_parent_roots=roots,
    )
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=roots,
        allow_qualification_locators=True,
    )
    governance = d4._governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    gca_kwargs = d4._lifecycle_gca_kwargs(storage)
    composition_builder = gca_kwargs["composition_builder"]
    from intergrax.applications._shared.production_process_lifecycle import (
        open_production_process_lifecycle,
    )

    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
        owned_governed_external_work_host_store_bundle=gca_kwargs[
            "owned_governed_external_work_host_store_bundle"
        ],
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        bundle, locator, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-dim2-untrusted",
            marker="dim2-untrusted",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
    intent = StrictProductionActivationIntent(
        projection_input=bundle,
        activation_request=_activation_request(bundle, locator),
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="trusted"):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=adapter,
            activation_intent=intent,
            **gca_kwargs,
            missing_continuation_file_policy="empty",
        )


def _adapter_from_session(session):
    handoff = session.serving_runtime_handoff
    assert handoff is not None
    admission = handoff.colocated_execution_admission
    assert admission is not None
    return handoff, admission.physical_gate.adapter


def test_c_dim3_01_positive_attest_traffic_serving_unit_after_full_session(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    revision_id = "rev-c-dim3-attest-live"
    session = d4._open_with_activation(
        tmp_path,
        environment=d4._ac3_environment(),
        revision_id=revision_id,
    )
    try:
        handoff, adapter = _adapter_from_session(session)
        identity = handoff.identity
        stores = session.composition.agent_platform_runtime.stores
        revision = stores.revision_store.get_revision(identity.runtime_revision_id)
        assert revision is not None
        record = stores.materialization_store.get_by_revision(identity.runtime_revision_id)
        assert record is not None
        attestation = adapter.attest_serving_unit(
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
        assert attestation.serving_unit_ref == identity.serving_unit_ref
    finally:
        session.close()


def test_c_dim3_02_negative_stale_serving_unit_unknown_on_fresh_adapter(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    session = d4._open_with_activation(
        tmp_path,
        environment=environment,
        revision_id="rev-c-dim3-stale",
    )
    handoff, _live_adapter = _adapter_from_session(session)
    identity = handoff.identity
    stores = session.composition.agent_platform_runtime.stores
    revision = stores.revision_store.get_revision(identity.runtime_revision_id)
    record = stores.materialization_store.get_by_revision(identity.runtime_revision_id)
    session.close()
    fresh_adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        allow_qualification_locators=True,
    )
    assert revision is not None and record is not None
    with pytest.raises(ServingUnitAttestationError, match="unknown"):
        fresh_adapter.attest_serving_unit(
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


def test_c_dim3_03_negative_stale_ready_physical_unit_denied_on_reopen(
    tmp_path: Path,
) -> None:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    governance = d4._governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )
    gca_kwargs = d4._lifecycle_gca_kwargs(storage)
    composition_builder = gca_kwargs["composition_builder"]
    from intergrax.applications._shared.production_process_lifecycle import (
        open_production_process_lifecycle,
    )

    with open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=governance,
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        missing_file_policy="empty",
        provider_invocation_db=storage.provider_invocation_db,
        composition_builder=composition_builder,
        owned_governed_external_work_host_store_bundle=gca_kwargs[
            "owned_governed_external_work_host_store_bundle"
        ],
        serving_reconciliation_targets=(
            (environment.application_id, environment.application_environment_id),
        ),
    ) as lifecycle:
        bundle, locator, *_rest = d4._qualified_canonical_bundle(
            tmp_path,
            revision_id="rev-c-dim3-stale-ready",
            application_id=environment.application_id,
            application_environment_id=environment.application_environment_id,
            stores=lifecycle.composition.agent_platform_runtime.stores,
        )
        d4._activate_bundle(
            lifecycle.launcher,
            bundle,
            locator,
            principal=governance.principal,
        )
    fresh_adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        allow_qualification_locators=True,
    )
    with pytest.raises(StrictProductionServingLauncherError):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=fresh_adapter,
            **gca_kwargs,
            missing_continuation_file_policy="empty",
        )
