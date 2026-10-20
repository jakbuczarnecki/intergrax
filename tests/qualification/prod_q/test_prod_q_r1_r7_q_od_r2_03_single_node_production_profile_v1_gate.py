# © Artur Czarnecki. All rights reserved.

"""PROD-Q OD-R2-03 — single-node Linux VM production profile v1 (CONFIGURED / REPO-TESTED gates)."""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

from governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle import (
    open_sqlite_governed_external_work_host_store_bundle_at,
)
from intergrax.applications._shared.colocated_serving_runtime_composition import (
    ColocatedServingAdmissionTopologyError,
)
from intergrax.applications._shared.production_process_lifecycle import (
    ProductionProcessLifecycle,
    ProductionProcessLifecycleError,
    open_production_process_lifecycle,
)
from intergrax.applications._shared.settings_loader import EnvReader
from intergrax.applications._shared.single_node_production_profile_v1 import (
    SINGLE_NODE_PRODUCTION_PROFILE_V1_ID,
    SingleNodeProductionProfileV1Configuration,
    SingleNodeProductionProfileV1ConfigurationError,
    SqliteEffectiveDurabilityPragmas,
    acquire_cross_process_writer_admission_for_profile_v1,
    assert_owned_connection_sqlite_wal_full_pragmas,
    load_single_node_production_profile_v1_from_env,
    load_validate_and_admit_single_node_production_profile_v1_from_env,
    probe_linux_ext4_mount_for_path,
    validate_configured_single_node_production_profile_v1,
    writer_lock_path_for_profile_v1,
)
from intergrax.applications._shared.production_process_serving_reconciliation import (
    ProductionServingReconciliationEvidence,
    ProductionServingReconciliationResult,
    RECONCILIATION_OUTCOME_READY,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingEnvironment,
    StrictProductionServingLauncherError,
    StrictProductionServingReadiness,
    StrictProductionServingSession,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.hosting.instance._native_lock import NativeFileLock, NativeFileLockError
from governed_contractor_application.host.single_node_profile_v1_strict_startup import (
    open_gca_strict_production_serving_session_from_single_node_production_profile_v1,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionProcessStorage,
)
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    open_provider_invocation_store_at,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import repo_root
from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
    _governance,
)
from tests.unit.applications.test_prod_q_r1_r7_d2_production_process_lifecycle import (
    _env as d2_env,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_c_dim10_asgi_runtime_identity_continuity_gate import (
    _dim10_http_fixture,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "19a809af2bd7d6df425c718dea29a35aa28116a2"
_SERVICE_FILE: Final = (
    repo_root()
    / "applications"
    / "governed_contractor_application"
    / "deploy"
    / "linux"
    / "single-node-production-profile-v1"
    / "intergrax-gca-strict-production.service"
)
_ENTRYPOINT_FILE: Final = (
    repo_root()
    / "applications"
    / "governed_contractor_application"
    / "deploy"
    / "linux"
    / "single-node-production-profile-v1"
    / "intergrax-gca-strict-production-entrypoint.sh"
)
_PROBE_FILE: Final = (
    repo_root()
    / "applications"
    / "governed_contractor_application"
    / "deploy"
    / "linux"
    / "single-node-production-profile-v1"
    / "target-storage-probe.sh"
)


def _lab_durable_dir() -> Path:
    base = (
        repo_root()
        / ".tmp"
        / "session"
        / "od-r2-03-single-node-profile-v1"
        / uuid.uuid4().hex
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _config_for_data_dir(data: Path) -> SingleNodeProductionProfileV1Configuration:
    roots = (data / "trusted_artifacts",)
    roots[0].mkdir(parents=True, exist_ok=True)
    return SingleNodeProductionProfileV1Configuration(
        profile_id=SINGLE_NODE_PRODUCTION_PROFILE_V1_ID,
        durable_data_directory=data,
        service_runtime_user="intergrax-gca",
        trusted_artifact_parent_roots=roots,
        gca_host_sqlite_path=data / "gca_host.sqlite",
        provider_invocation_sqlite_path=data / "provider_invocation.sqlite",
        agent_distribution_sqlite_path=data / "agent_distribution.sqlite",
        continuation_state_file=data / "execution_continuation.seal.json",
    )


def test_od_r2_03_profile_g00_baseline_ancestor() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def test_od_r2_03_profile_01_configured_identity_from_env() -> None:
    data = _lab_durable_dir()
    reader = EnvReader("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR", str(data))
        mp.setenv("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_SERVICE_RUNTIME_USER", "intergrax-gca")
        config = load_single_node_production_profile_v1_from_env(env=reader)
    assert config.profile_id == SINGLE_NODE_PRODUCTION_PROFILE_V1_ID
    validate_configured_single_node_production_profile_v1(config)
    storage = config.resolve_strict_production_process_storage()
    assert isinstance(storage, StrictProductionProcessStorage)


def test_od_r2_03_profile_missing_data_dir_fail_closed() -> None:
    reader = EnvReader("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_")
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR", raising=False)
        with pytest.raises(SingleNodeProductionProfileV1ConfigurationError, match="DATA_DIR"):
            load_single_node_production_profile_v1_from_env(env=reader)


def test_od_r2_03_profile_02_rejects_ephemeral_data_dir() -> None:
    ephemeral = Path(tempfile.gettempdir()) / "intergrax-od-r2-03-ephemeral"
    ephemeral.mkdir(parents=True, exist_ok=True)
    config = _config_for_data_dir(ephemeral)
    with pytest.raises(SingleNodeProductionProfileV1ConfigurationError, match="ephemeral"):
        validate_configured_single_node_production_profile_v1(config)


def test_od_r2_03_profile_02_ext4_probe_not_target_verified_off_linux() -> None:
    data = _lab_durable_dir()
    probe = probe_linux_ext4_mount_for_path(data)
    assert probe.applicable is False


def test_pv1_h05_owned_connection_sqlite_wal_full_after_sanctioned_backends() -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    validate_configured_single_node_production_profile_v1(config)
    gca = open_sqlite_governed_external_work_host_store_bundle_at(
        config.gca_host_sqlite_path,
    )
    pis = open_provider_invocation_store_at(config.provider_invocation_sqlite_path)
    try:
        gca_journal, gca_sync = gca.read_owned_connection_durability_pragmas()
        pis_journal, pis_sync = pis.read_owned_connection_durability_pragmas()
        assert_owned_connection_sqlite_wal_full_pragmas(
            SqliteEffectiveDurabilityPragmas(journal_mode=gca_journal, synchronous=gca_sync),
            db_path=config.gca_host_sqlite_path,
        )
        assert_owned_connection_sqlite_wal_full_pragmas(
            SqliteEffectiveDurabilityPragmas(journal_mode=pis_journal, synchronous=pis_sync),
            db_path=config.provider_invocation_sqlite_path,
        )
    finally:
        gca.close()
        pis.close()


def test_pv1_h01_profile_reaches_sanctioned_strict_startup() -> None:
    data = _lab_durable_dir()
    reader = EnvReader("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR", str(data))
        expected_storage = load_validate_and_admit_single_node_production_profile_v1_from_env(
            env=reader,
            acquire_cross_process_writer=False,
        )[1]
        environment = StrictProductionServingEnvironment(
            application_id="gca",
            application_environment_id="production",
        )
        with patch(
            "governed_contractor_application.host.single_node_profile_v1_strict_startup."
            "open_gca_strict_production_serving_session",
        ) as open_gca:
            open_gca.side_effect = RuntimeError("sanctioned-boundary-reached")
            with pytest.raises(RuntimeError, match="sanctioned-boundary-reached"):
                open_gca_strict_production_serving_session_from_single_node_production_profile_v1(
                    environment=environment,
                    governance=_governance(d2_env()),
                    deployment_adapter=object(),  # type: ignore[arg-type]
                    env=reader,
                )
            assert open_gca.call_args.kwargs["storage"] == expected_storage


def test_pv1_h02_missing_configuration_fails_closed() -> None:
    test_od_r2_03_profile_missing_data_dir_fail_closed()


def test_pv1_h06_ephemeral_storage_rejected() -> None:
    test_od_r2_03_profile_02_rejects_ephemeral_data_dir()


def test_pv1_h04_cross_process_writer_lock_excludes_second_holder() -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    validate_configured_single_node_production_profile_v1(config)
    first = acquire_cross_process_writer_admission_for_profile_v1(config)
    lock_path = writer_lock_path_for_profile_v1(config)
    try:
        with pytest.raises(SingleNodeProductionProfileV1ConfigurationError, match="writer lock"):
            acquire_cross_process_writer_admission_for_profile_v1(config)
        with pytest.raises(NativeFileLockError):
            NativeFileLock(lock_path)
    finally:
        first.release()


def test_pv1_h03_no_alternate_startup_composition_authority() -> None:
    startup = (
        repo_root()
        / "applications"
        / "governed_contractor_application"
        / "host"
        / "single_node_profile_v1_strict_startup.py"
    ).read_text(encoding="utf-8")
    assert startup.count("open_gca_strict_production_serving_session(") == 1
    assert "open_strict_production_serving_session(" not in startup
    main_py = (
        repo_root() / "applications" / "governed_contractor_application" / "host" / "main.py"
    ).read_text(encoding="utf-8")
    assert "Do not use main.run()" in main_py


def test_pv1_h07_systemd_and_entrypoint_agree() -> None:
    unit = _SERVICE_FILE.read_text(encoding="utf-8")
    entry = _ENTRYPOINT_FILE.read_text(encoding="utf-8")
    assert "EnvironmentFile=/etc/intergrax/intergrax-gca-strict-production.env" in unit
    assert "ConditionPathExists=/etc/intergrax/intergrax-gca-strict-production.env" in unit
    assert "single_node_profile_v1_admission_gate" in entry
    assert "open_gca_strict_production_serving_session_from_single_node_production_profile_v1" in entry
    assert "INTERGRAX_GCA_STRICT_SERVE_EXEC" in entry


def test_pv1_h08_probe_informational_admission_gate_fail_closed() -> None:
    from governed_contractor_application.host.single_node_profile_v1_admission_gate import main

    probe = _PROBE_FILE.read_text(encoding="utf-8")
    assert "INFORMATIONAL ONLY" in probe
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR", raising=False)
        assert main() == 2


def test_pv1_h09_failed_startup_does_not_expand_provider_surface() -> None:
    from governed_contractor_application.host.strict_gca_production_process_composition import (
        open_gca_strict_production_serving_session,
    )

    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    storage = config.resolve_strict_production_process_storage()
    environment = StrictProductionServingEnvironment(application_id="a", application_environment_id="e")
    with patch(
        "governed_contractor_application.host.strict_gca_production_process_composition."
        "open_strict_production_serving_session",
        side_effect=RuntimeError("startup failed"),
    ):
        with pytest.raises(RuntimeError, match="startup failed"):
            open_gca_strict_production_serving_session(
                storage=storage,
                environment=environment,
                governance=_governance(d2_env()),
                deployment_adapter=object(),  # type: ignore[arg-type]
            )
    reopened = open_sqlite_governed_external_work_host_store_bundle_at(storage.governed_external_work_host_db)
    reopened.close()


def test_pv1_h10_target_production_qualification_explicitly_unproven() -> None:
    dossier = (
        repo_root()
        / "docs"
        / "project"
        / "maintainers"
        / "qualification"
        / "PROD_Q_R1_R7_Q_OD_R2_03_SINGLE_NODE_PRODUCTION_PROFILE_V1.md"
    ).read_text(encoding="utf-8")
    assert "TARGET-VERIFIED" in dossier
    assert "OPERATOR-ACCEPTED" in dossier
    assert "B6" in dossier


def test_pv1_h04_in_process_single_lifecycle_slot() -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    storage = config.resolve_strict_production_process_storage()
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        allow_qualification_locators=True,
    )
    env = d2_env()
    first = open_production_process_lifecycle(
        agent_distribution_db=storage.agent_distribution_db,
        continuation_state_file=storage.continuation_state_file,
        deployment_adapter=adapter,
        governance=_governance(env),
        trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
        provider_invocation_db=storage.provider_invocation_db,
        missing_file_policy="empty",
    )
    with pytest.raises(ProductionProcessLifecycleError, match="duplicate"):
        open_production_process_lifecycle(
            agent_distribution_db=storage.agent_distribution_db,
            continuation_state_file=storage.continuation_state_file,
            deployment_adapter=adapter,
            governance=_governance(env),
            trusted_artifact_parent_roots=storage.trusted_artifact_parent_roots,
            provider_invocation_db=storage.provider_invocation_db,
            missing_file_policy="empty",
        )
    first.close_process_resources()


def test_od_r2_03_profile_05_systemd_unit_static_contract() -> None:
    text = _SERVICE_FILE.read_text(encoding="utf-8")
    assert "User=intergrax-gca" in text
    assert "Restart=on-failure" in text
    assert "TimeoutStopSec=120" in text
    assert "PrivateTmp=true" in text
    assert "intergrax-gca-strict-production-entrypoint" in text
    assert "ExecStartPre=" in text


def test_od_r2_03_profile_rejects_missing_data_directory() -> None:
    parent = _lab_durable_dir()
    missing = parent / "not-created-yet"
    config = SingleNodeProductionProfileV1Configuration(
        profile_id=SINGLE_NODE_PRODUCTION_PROFILE_V1_ID,
        durable_data_directory=missing,
        service_runtime_user="intergrax-gca",
        trusted_artifact_parent_roots=(missing / "trusted_artifacts",),
        gca_host_sqlite_path=missing / "gca_host.sqlite",
        provider_invocation_sqlite_path=missing / "provider_invocation.sqlite",
        agent_distribution_sqlite_path=missing / "agent_distribution.sqlite",
        continuation_state_file=missing / "execution_continuation.seal.json",
    )
    with pytest.raises(SingleNodeProductionProfileV1ConfigurationError, match="directory"):
        validate_configured_single_node_production_profile_v1(config)


def _minimal_strict_session_for_writer_hold_lifecycle() -> StrictProductionServingSession:
    lifecycle = MagicMock()
    lifecycle._process_closed = False

    def _close_process_resources() -> None:
        lifecycle._process_closed = True

    lifecycle.close_process_resources = _close_process_resources
    lifecycle.composition.serving_runtime_handoff = None
    reconciliation = ProductionServingReconciliationResult(
        evidence=ProductionServingReconciliationEvidence(
            application_id="gca",
            application_environment_id="production",
            runtime_revision_id=None,
            serving_pointer_revision=None,
            outcome=RECONCILIATION_OUTCOME_READY,
        ),
        attestation=None,
        rehydration=None,
    )
    readiness = StrictProductionServingReadiness(
        process_lifecycle_open=True,
        production_dependencies_validated=True,
        serving_state_reconciled=True,
        serving_handoff_bound=False,
        execution_admission_eligible=False,
        ready_for_traffic=False,
    )
    return StrictProductionServingSession(
        lifecycle=lifecycle,
        environment=StrictProductionServingEnvironment(
            application_id="gca",
            application_environment_id="production",
        ),
        reconciliation=reconciliation,
        readiness=readiness,
    )


def test_life_01_startup_failure_after_writer_lock_releases_hold() -> None:
    data = _lab_durable_dir()
    reader = EnvReader("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_")
    config = _config_for_data_dir(data)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR", str(data))
        environment = StrictProductionServingEnvironment(
            application_id="gca",
            application_environment_id="production",
        )
        with patch(
            "governed_contractor_application.host.single_node_profile_v1_strict_startup."
            "open_gca_strict_production_serving_session",
            side_effect=RuntimeError("gca-open-failed"),
        ):
            with pytest.raises(RuntimeError, match="gca-open-failed"):
                open_gca_strict_production_serving_session_from_single_node_production_profile_v1(
                    environment=environment,
                    governance=_governance(d2_env()),
                    deployment_adapter=object(),  # type: ignore[arg-type]
                    env=reader,
                )
    released = acquire_cross_process_writer_admission_for_profile_v1(config)
    released.release()


def test_life_02_cross_process_writer_lock_excludes_second_holder() -> None:
    test_pv1_h04_cross_process_writer_lock_excludes_second_holder()


def test_life_03_writer_lock_held_for_active_strict_session_lifecycle() -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    hold = acquire_cross_process_writer_admission_for_profile_v1(config)
    session = _minimal_strict_session_for_writer_hold_lifecycle()
    session.bind_process_scoped_release_on_close(hold.release)
    lock_path = writer_lock_path_for_profile_v1(config)
    try:
        with pytest.raises(SingleNodeProductionProfileV1ConfigurationError, match="writer lock"):
            acquire_cross_process_writer_admission_for_profile_v1(config)
        with pytest.raises(NativeFileLockError):
            NativeFileLock(lock_path)
    finally:
        session.close()


def test_life_04_session_close_releases_writer_lock_for_subsequent_startup() -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    hold = acquire_cross_process_writer_admission_for_profile_v1(config)
    session = _minimal_strict_session_for_writer_hold_lifecycle()
    session.bind_process_scoped_release_on_close(hold.release)
    session.close()
    second = acquire_cross_process_writer_admission_for_profile_v1(config)
    second.release()


def test_life_05_failed_startup_does_not_return_serving_session() -> None:
    data = _lab_durable_dir()
    reader = EnvReader("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR", str(data))
        environment = StrictProductionServingEnvironment(
            application_id="gca",
            application_environment_id="production",
        )
        with patch(
            "governed_contractor_application.host.single_node_profile_v1_strict_startup."
            "open_gca_strict_production_serving_session",
            side_effect=RuntimeError("no-session"),
        ):
            with pytest.raises(RuntimeError, match="no-session"):
                open_gca_strict_production_serving_session_from_single_node_production_profile_v1(
                    environment=environment,
                    governance=_governance(d2_env()),
                    deployment_adapter=object(),  # type: ignore[arg-type]
                    env=reader,
                )
    test_pv1_h09_failed_startup_does_not_expand_provider_surface()


def test_life_06_sanctioned_d4_gca_asgi_factory_preserved() -> None:
    serving = (
        repo_root()
        / "applications"
        / "governed_contractor_application"
        / "host"
        / "strict_production_serving.py"
    ).read_text(encoding="utf-8")
    assert "create_governed_contractor_strict_production_asgi_app" in serving
    assert "build_governed_contractor_strict_production_uvicorn_factory" in serving
    assert "attach_strict_production_serving_lifespan" in serving
    test_pv1_h03_no_alternate_startup_composition_authority()


def test_life_07_missing_operator_serve_configuration_fails_closed() -> None:
    entry = _ENTRYPOINT_FILE.read_text(encoding="utf-8")
    assert "INTERGRAX_GCA_STRICT_SERVE_EXEC" in entry
    assert 'serve_exec="${INTERGRAX_GCA_STRICT_SERVE_EXEC:-}"' in entry
    assert 'if [[ -z "${serve_exec}" ]]' in entry
    assert "exit 2" in entry
    test_pv1_h07_systemd_and_entrypoint_agree()


def test_life_08_target_verified_and_operator_acceptance_remain_open() -> None:
    test_pv1_h10_target_production_qualification_explicitly_unproven()


def _minimal_strict_session_with_colocated_handoff() -> StrictProductionServingSession:
    session = _minimal_strict_session_for_writer_hold_lifecycle()
    handoff = MagicMock()
    stores = MagicMock()
    session.lifecycle.composition.serving_runtime_handoff = handoff
    session.lifecycle.composition.agent_platform_runtime.stores = stores
    session.lifecycle.deployment_adapter = MagicMock()
    return session


def _assert_writer_lock_held(config: SingleNodeProductionProfileV1Configuration) -> None:
    with pytest.raises(SingleNodeProductionProfileV1ConfigurationError, match="writer lock"):
        acquire_cross_process_writer_admission_for_profile_v1(config)


def _assert_writer_lock_releasable(
    config: SingleNodeProductionProfileV1Configuration,
) -> None:
    second = acquire_cross_process_writer_admission_for_profile_v1(config)
    second.release()


def test_sfc_01_normal_close_releases_writer_lock() -> None:
    test_life_04_session_close_releases_writer_lock_for_subsequent_startup()


def test_sfc_02_admission_close_failure_retains_writer_lock_and_error() -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    hold = acquire_cross_process_writer_admission_for_profile_v1(config)
    session = _minimal_strict_session_with_colocated_handoff()
    session.bind_process_scoped_release_on_close(hold.release)
    close_fail = VenvBundleColocatedDeploymentError("sfc-02-admission-close-fail")
    close_calls = 0
    original_close = ProductionProcessLifecycle.close_process_resources

    def _tracked_close(self: ProductionProcessLifecycle) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(self)

    with patch(
        (
            "intergrax.applications._shared.strict_production_serving_launcher."
            "close_colocated_serving_execution_admission_on_strict_session_close"
        ),
        side_effect=close_fail,
    ), patch.object(
        ProductionProcessLifecycle,
        "close_process_resources",
        _tracked_close,
    ):
        with pytest.raises(StrictProductionServingLauncherError) as exc_info:
            session.close()
    assert "sfc-02-admission-close-fail" in str(exc_info.value)
    assert close_calls == 0
    assert session._closed is False
    assert session.lifecycle._process_closed is False
    _assert_writer_lock_held(config)
    hold.release()


def test_sfc_03_drain_failure_releases_writer_lock_when_lifecycle_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    hold = acquire_cross_process_writer_admission_for_profile_v1(config)
    drain_fail = ColocatedServingAdmissionTopologyError("sfc-03-drain-fail")
    with _dim10_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-od-r2-03-sfc-03-drain-fail",
    ) as fixture:
        fixture.session.bind_process_scoped_release_on_close(hold.release)
        with patch(
            (
                "intergrax.applications._shared.strict_production_serving_launcher."
                "begin_colocated_serving_execution_drain_on_strict_session_close"
            ),
            side_effect=drain_fail,
        ):
            with pytest.raises(StrictProductionServingLauncherError) as exc_info:
                fixture.session.close()
        assert "sfc-03-drain-fail" in str(exc_info.value)
        assert fixture.session.lifecycle._process_closed is True
        _assert_writer_lock_releasable(config)


def test_sfc_04_lifecycle_cleanup_failure_retains_writer_lock() -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    hold = acquire_cross_process_writer_admission_for_profile_v1(config)
    session = _minimal_strict_session_for_writer_hold_lifecycle()
    session.bind_process_scoped_release_on_close(hold.release)
    cleanup_fail = RuntimeError("sfc-04-lifecycle-cleanup-fail")

    def _close_raises() -> None:
        raise cleanup_fail

    session.lifecycle.close_process_resources = _close_raises
    with pytest.raises(RuntimeError, match="sfc-04-lifecycle-cleanup-fail"):
        session.close()
    assert session.lifecycle._process_closed is False
    _assert_writer_lock_held(config)
    hold.release()


def test_sfc_05_second_close_after_failed_shutdown_does_not_release_writer_lock() -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    hold = acquire_cross_process_writer_admission_for_profile_v1(config)
    session = _minimal_strict_session_with_colocated_handoff()
    session.bind_process_scoped_release_on_close(hold.release)
    close_fail = VenvBundleColocatedDeploymentError("sfc-05-admission-close-fail")
    with patch(
        (
            "intergrax.applications._shared.strict_production_serving_launcher."
            "close_colocated_serving_execution_admission_on_strict_session_close"
        ),
        side_effect=close_fail,
    ):
        with pytest.raises(StrictProductionServingLauncherError):
            session.close()
        with pytest.raises(StrictProductionServingLauncherError):
            session.close()
        _assert_writer_lock_held(config)
    hold.release()


def test_sfc_06_cross_process_denied_after_unsafe_shutdown_failure() -> None:
    test_sfc_02_admission_close_failure_retains_writer_lock_and_error()


def test_sfc_07_startup_failure_releases_writer_lock() -> None:
    test_life_01_startup_failure_after_writer_lock_releases_hold()


def test_sfc_08_writer_release_bound_to_lifecycle_owner_only() -> None:
    launcher = (
        repo_root()
        / "intergrax"
        / "applications"
        / "_shared"
        / "strict_production_serving_launcher.py"
    ).read_text(encoding="utf-8")
    assert "if self.lifecycle._process_closed:" in launcher
    assert "_invoke_process_scoped_close_releases()" in launcher
    startup = (
        repo_root()
        / "applications"
        / "governed_contractor_application"
        / "host"
        / "single_node_profile_v1_strict_startup.py"
    ).read_text(encoding="utf-8")
    assert "bind_process_scoped_release_on_close(writer_hold.release)" in startup
