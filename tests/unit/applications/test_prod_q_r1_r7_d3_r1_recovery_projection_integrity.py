# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D3-R1 — mandatory recovery coverage and projection cache integrity."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryApplicationEnvironmentServingStore,
)
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingRecord
from intergrax.applications._shared.production_lifecycle import (
    build_descriptor_backed_projection_input_store,
)
from intergrax.applications._shared.production_process_lifecycle import (
    ProductionServingReconciliationError,
    open_production_process_lifecycle,
)
from intergrax.applications._shared.production_process_serving_reconciliation import (
    ProductionServingReconciliationEvidence,
    ProductionServingReconciliationResult,
    RECONCILIATION_OUTCOME_SKIPPED,
    resolve_mandatory_production_serving_reconciliation_targets,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionError
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
    _governance,
    _venv_adapter,
)
from tests.unit.applications.test_prod_q_r1_r7_d3_production_restart_reconciliation import (
    _durable_launcher,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _APP,
    _ENV,
    _activate_bundle,
    _canonical_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ENV_B = "env-prod-b"


def test_d3_r1_resolve_mandatory_targets_includes_all_durable_traffic() -> None:
    state = AgentDistributionStoreState()
    serving = InMemoryApplicationEnvironmentServingStore(state)
    state.serving_records[(_APP, _ENV)] = ApplicationEnvironmentServingRecord(
        application_id=_APP,
        application_environment_id=_ENV,
        traffic_serving_revision_id="rev-a",
        serving_pointer_revision=1,
    )
    state.serving_records[(_APP, _ENV_B)] = ApplicationEnvironmentServingRecord(
        application_id=_APP,
        application_environment_id=_ENV_B,
        traffic_serving_revision_id="rev-b",
        serving_pointer_revision=1,
    )
    resolved = resolve_mandatory_production_serving_reconciliation_targets(
        serving,
        (),
    )
    assert resolved == ((_APP, _ENV), (_APP, _ENV_B))


def test_d3_r1_caller_cannot_omit_active_environment() -> None:
    state = AgentDistributionStoreState()
    serving = InMemoryApplicationEnvironmentServingStore(state)
    state.serving_records[(_APP, _ENV)] = ApplicationEnvironmentServingRecord(
        application_id=_APP,
        application_environment_id=_ENV,
        traffic_serving_revision_id="rev-a",
        serving_pointer_revision=1,
    )
    resolved = resolve_mandatory_production_serving_reconciliation_targets(
        serving,
        (("other-app", "other-env"),),
    )
    assert (_APP, _ENV) in resolved


def _skip_reconcile(**kwargs: object) -> ProductionServingReconciliationResult:
    return ProductionServingReconciliationResult(
        evidence=ProductionServingReconciliationEvidence(
            application_id=str(kwargs["application_id"]),
            application_environment_id=str(kwargs["application_environment_id"]),
            runtime_revision_id=None,
            serving_pointer_revision=None,
            outcome=RECONCILIATION_OUTCOME_SKIPPED,
        ),
        attestation=None,
        rehydration=None,
    )


def _activate_serving_in_durable_db(
    tmp_path: Path,
    db_path: Path,
    *,
    revision_id: str,
) -> None:
    launcher, _durable, governance = _durable_launcher(tmp_path, db_path)
    bundle, locator, *_rest = _canonical_bundle(
        tmp_path,
        revision_id=revision_id,
        stores=launcher.process_composition.agent_platform_runtime.stores,
    )
    _activate_bundle(launcher, bundle, locator, principal=governance.principal)


def test_d3_r1_open_without_targets_reconciles_active_serving(tmp_path: Path) -> None:
    db_path = tmp_path / "d3-r1-mandatory.db"
    _activate_serving_in_durable_db(
        tmp_path,
        db_path,
        revision_id="rev-mandatory-open",
    )
    env = ApplicationEnvironmentProfile(profile_id=_ENV)
    roots = (tmp_path / "artifacts",)
    roots[0].mkdir(parents=True, exist_ok=True)
    reconcile_calls: list[tuple[str, str]] = []

    def _track(**kwargs: object) -> ProductionServingReconciliationResult:
        reconcile_calls.append(
            (str(kwargs["application_id"]), str(kwargs["application_environment_id"])),
        )
        return _skip_reconcile(**kwargs)

    with patch(
        "intergrax.applications._shared.production_process_lifecycle."
        "reconcile_production_serving_environment_on_process_open",
        side_effect=_track,
    ):
        with open_production_process_lifecycle(
            agent_distribution_db=db_path,
            continuation_state_file=tmp_path / "continuation-reopen.json",
            deployment_adapter=_venv_adapter(tmp_path),
            governance=_governance(env),
            trusted_artifact_parent_roots=roots,
            missing_file_policy="empty",
        ) as lifecycle:
            assert lifecycle.composition.agent_platform_runtime is not None
    assert reconcile_calls == [(_APP, _ENV)]


def test_d3_r1_open_clean_when_no_traffic_serving(tmp_path: Path) -> None:
    env = ApplicationEnvironmentProfile(profile_id=_ENV)
    roots = (tmp_path / "artifacts",)
    roots[0].mkdir(parents=True, exist_ok=True)
    with open_production_process_lifecycle(
        agent_distribution_db=tmp_path / "empty-ad.db",
        continuation_state_file=tmp_path / "continuation.json",
        deployment_adapter=_venv_adapter(tmp_path),
        governance=_governance(env),
        trusted_artifact_parent_roots=roots,
        missing_file_policy="empty",
    ):
        pass


def test_d3_r1_open_fails_stale_physical_without_caller_targets(tmp_path: Path) -> None:
    db_path = tmp_path / "d3-r1-stale.db"
    _activate_serving_in_durable_db(tmp_path, db_path, revision_id="rev-stale-r1")
    env = ApplicationEnvironmentProfile(profile_id=_ENV)
    roots = (tmp_path,)
    with pytest.raises(ProductionServingReconciliationError, match="attestation"):
        open_production_process_lifecycle(
            agent_distribution_db=db_path,
            continuation_state_file=tmp_path / "continuation.json",
            deployment_adapter=_venv_adapter(tmp_path),
            governance=_governance(env),
            trusted_artifact_parent_roots=roots,
            missing_file_policy="empty",
        )


def test_d3_r1_reconciliation_failure_releases_opening_slot(tmp_path: Path) -> None:
    from intergrax.applications._shared import production_process_lifecycle as ppl

    db_path = tmp_path / "d3-r1-slot.db"
    _activate_serving_in_durable_db(tmp_path, db_path, revision_id="rev-slot-release")
    env = ApplicationEnvironmentProfile(profile_id=_ENV)
    roots = (tmp_path / "artifacts",)
    roots[0].mkdir(parents=True, exist_ok=True)
    with pytest.raises(ProductionServingReconciliationError):
        open_production_process_lifecycle(
            agent_distribution_db=db_path,
            continuation_state_file=tmp_path / "continuation.json",
            deployment_adapter=_venv_adapter(tmp_path),
            governance=_governance(env),
            trusted_artifact_parent_roots=roots,
            missing_file_policy="empty",
        )
    assert ppl._production_slot_phase is ppl._ProductionSlotPhase.AVAILABLE
    with patch(
        "intergrax.applications._shared.production_process_lifecycle."
        "reconcile_production_serving_environment_on_process_open",
        side_effect=_skip_reconcile,
    ):
        with open_production_process_lifecycle(
            agent_distribution_db=db_path,
            continuation_state_file=tmp_path / "continuation-retry.json",
            deployment_adapter=_venv_adapter(tmp_path),
            governance=_governance(env),
            trusted_artifact_parent_roots=roots,
            missing_file_policy="empty",
        ) as lifecycle:
            assert lifecycle is not None


def test_d3_r1_multiple_serving_environments_all_reconciled(tmp_path: Path) -> None:
    db_path = tmp_path / "d3-r1-multi.db"
    _activate_serving_in_durable_db(tmp_path, db_path, revision_id="rev-multi")
    seen: list[tuple[str, str]] = []

    def _track(**kwargs: object) -> ProductionServingReconciliationResult:
        seen.append(
            (str(kwargs["application_id"]), str(kwargs["application_environment_id"])),
        )
        return _skip_reconcile(**kwargs)

    env = ApplicationEnvironmentProfile(profile_id=_ENV)
    roots = (tmp_path / "artifacts",)
    roots[0].mkdir(parents=True, exist_ok=True)
    mandatory = ((_APP, _ENV), (_APP, _ENV_B))
    with (
        patch(
            "intergrax.applications._shared.production_process_lifecycle."
            "resolve_mandatory_production_serving_reconciliation_targets",
            return_value=mandatory,
        ),
        patch(
            "intergrax.applications._shared.production_process_lifecycle."
            "reconcile_production_serving_environment_on_process_open",
            side_effect=_track,
        ),
    ):
        with open_production_process_lifecycle(
            agent_distribution_db=db_path,
            continuation_state_file=tmp_path / "continuation.json",
            deployment_adapter=_venv_adapter(tmp_path),
            governance=_governance(env),
            trusted_artifact_parent_roots=roots,
            missing_file_policy="empty",
        ) as lifecycle:
            assert lifecycle is not None
    assert sorted(seen) == list(mandatory)


def test_d3_r1_projection_register_stages_without_descriptor(tmp_path: Path) -> None:
    from intergrax.applications._shared.durable_agent_platform_runtime import (
        build_durable_production_agent_platform_runtime,
    )

    db_path = tmp_path / "d3-r1-stage.db"
    durable = build_durable_production_agent_platform_runtime(db_path)
    bundle, locator, *_rest = _canonical_bundle(
        tmp_path,
        revision_id="rev-stage",
        stores=durable.agent_platform_runtime.stores,
    )
    store = build_descriptor_backed_projection_input_store(
        durable.distribution_store_bundle,
    )
    store.register(bundle)
    store.register(bundle)
    revision_id = bundle.runtime_revision.runtime_revision_id
    assert store.get(revision_id) is not None


def test_d3_r1_projection_register_rejects_durable_mismatch(tmp_path: Path) -> None:
    from intergrax.applications._shared.durable_agent_platform_runtime import (
        build_durable_production_agent_platform_runtime,
    )

    revision_id = "rev-mismatch"
    db_path = tmp_path / "d3-r1-mismatch.db"
    _activate_serving_in_durable_db(tmp_path, db_path, revision_id=revision_id)
    durable = build_durable_production_agent_platform_runtime(db_path)
    store = build_descriptor_backed_projection_input_store(
        durable.distribution_store_bundle,
    )
    durable_bundle = store.get(revision_id)
    assert durable_bundle is not None
    conflicting = replace(
        durable_bundle,
        materialization_artifact_digest="sha256:" + ("f" * 64),
    )
    with pytest.raises(RegistryProjectionError, match="mismatch"):
        store.register(conflicting)
    assert store.get(revision_id) is not None


def test_d3_r1_failed_register_does_not_publish_rejected_bundle(tmp_path: Path) -> None:
    from intergrax.applications._shared.durable_agent_platform_runtime import (
        build_durable_production_agent_platform_runtime,
    )

    revision_id = "rev-failed-reg"
    db_path = tmp_path / "d3-r1-failed-reg.db"
    _activate_serving_in_durable_db(tmp_path, db_path, revision_id=revision_id)
    durable = build_durable_production_agent_platform_runtime(db_path)
    store = build_descriptor_backed_projection_input_store(
        durable.distribution_store_bundle,
    )
    durable_bundle = store.get(revision_id)
    assert durable_bundle is not None
    bad = replace(
        durable_bundle,
        materialization_artifact_digest="sha256:" + ("e" * 64),
    )
    with pytest.raises(RegistryProjectionError):
        store.register(bad)
    reconstructed = store.get(revision_id)
    assert reconstructed is not None
    assert (
        reconstructed.materialization_artifact_digest
        == durable_bundle.materialization_artifact_digest
    )


def test_d3_r1_concurrent_register_same_revision(tmp_path: Path) -> None:
    from intergrax.applications._shared.durable_agent_platform_runtime import (
        build_durable_production_agent_platform_runtime,
    )

    db_path = tmp_path / "d3-r1-concurrent.db"
    durable = build_durable_production_agent_platform_runtime(db_path)
    bundle, locator, *_rest = _canonical_bundle(
        tmp_path,
        revision_id="rev-concurrent",
        stores=durable.agent_platform_runtime.stores,
    )
    store = build_descriptor_backed_projection_input_store(
        durable.distribution_store_bundle,
    )
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def _register() -> None:
        try:
            barrier.wait(timeout=5)
            store.register(bundle)
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_register)
    t2 = threading.Thread(target=_register)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not errors
    assert store.get(bundle.runtime_revision.runtime_revision_id) is not None
