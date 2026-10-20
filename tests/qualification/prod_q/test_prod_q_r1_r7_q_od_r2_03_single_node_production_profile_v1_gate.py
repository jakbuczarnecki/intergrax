# © Artur Czarnecki. All rights reserved.

"""PROD-Q OD-R2-03 — single-node Linux VM production profile v1 (CONFIGURED / REPO-TESTED gates)."""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Final

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
    assert_repo_tested_sqlite_wal_full_pragmas,
    load_single_node_production_profile_v1_from_env,
    probe_linux_ext4_mount_for_path,
    validate_configured_single_node_production_profile_v1,
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

_EXPECTED_START_HEAD: Final = "a7122b6e019db9123a3dccad2816389f735fa855"
_SERVICE_FILE: Final = (
    repo_root()
    / "applications"
    / "governed_contractor_application"
    / "deploy"
    / "linux"
    / "single-node-production-profile-v1"
    / "intergrax-gca-strict-production.service"
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


def test_od_r2_03_profile_04_sqlite_wal_full_after_sanctioned_backends() -> None:
    data = _lab_durable_dir()
    config = _config_for_data_dir(data)
    validate_configured_single_node_production_profile_v1(config)
    gca = open_sqlite_governed_external_work_host_store_bundle_at(
        config.gca_host_sqlite_path,
    )
    pis = open_provider_invocation_store_at(config.provider_invocation_sqlite_path)
    try:
        gca.close()
        pis.close()
        assert_repo_tested_sqlite_wal_full_pragmas(config.gca_host_sqlite_path)
        assert_repo_tested_sqlite_wal_full_pragmas(config.provider_invocation_sqlite_path)
    finally:
        gca.close()
        pis.close()


def test_od_r2_03_profile_03_single_writer_slot_in_process() -> None:
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
