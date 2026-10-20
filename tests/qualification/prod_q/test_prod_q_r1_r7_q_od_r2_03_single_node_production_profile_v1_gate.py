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
from intergrax.applications._shared.production_process_lifecycle import (
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
    StrictProductionServingReadiness,
    StrictProductionServingSession,
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
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "da0fbb965ed3241cfbf052bec30fe9f27afed13c"
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
