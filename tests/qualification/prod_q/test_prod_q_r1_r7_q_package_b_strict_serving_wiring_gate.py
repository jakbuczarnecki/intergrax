# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q Package B — strict serving wiring (PB-S01…PB-S06 static + fail-closed)."""

from __future__ import annotations

import subprocess
import sys
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications._shared.package_b_operator_production_governance_configuration import (
    PackageBOperatorProductionGovernanceConfigurationError,
    load_package_b_operator_production_control_plane_governance_from_env,
)
from intergrax.applications._shared.settings_loader import EnvReader
from governed_contractor_application.host import package_b_strict_production_serving_command as pb_serve_cmd
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingLauncherError,
    StrictProductionServingReadiness,
    StrictProductionServingSession,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_od_r2_03_single_node_production_profile_v1_gate import (
    _minimal_strict_session_for_writer_hold_lifecycle,
    repo_root,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "c95c322f28e17e01744e5c28e7c52c7e84c4f3f4"
_GOVERNANCE_MODULE: Final = (
    repo_root()
    / "intergrax"
    / "applications"
    / "_shared"
    / "package_b_operator_production_governance_configuration.py"
)
_SERVE_COMMAND: Final = (
    repo_root()
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "package_b_strict_production_serving_command.py"
)
_COMPOSE: Final = (
    repo_root()
    / "applications"
    / "governed_contractor_application"
    / "deploy"
    / "docker-compose-local-qualification"
    / "docker-compose.yml"
)


def test_pb_s_g00_baseline_ancestor() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def test_pb_s01_single_governance_configuration_owner() -> None:
    text = _GOVERNANCE_MODULE.read_text(encoding="utf-8")
    assert "load_package_b_operator_production_control_plane_governance_from_env" in text
    assert "build_reference_production_control_plane_mutation_boundary" not in text
    assert "_FORBIDDEN_IN_REPO_BUNDLE_IDS" in text


def _package_b_command_ready_session() -> StrictProductionServingSession:
    session = _minimal_strict_session_for_writer_hold_lifecycle()
    session.readiness = StrictProductionServingReadiness(
        process_lifecycle_open=True,
        production_dependencies_validated=True,
        serving_state_reconciled=True,
        serving_handoff_bound=True,
        execution_admission_eligible=True,
        ready_for_traffic=True,
    )
    return session


def _patch_package_b_serve_preamble() -> tuple[MagicMock, ...]:
    settings = MagicMock()
    settings.backend_port = 18080
    settings.api_keys_map = {"operator": "secret"}
    settings.strict_external_work_provider_id = "provider-pin"
    settings.strict_external_work_provider_descriptor_digest = "digest-pin"
    return (
        patch.object(
            pb_serve_cmd.GovernedContractorBackendSettings,
            "from_env",
            return_value=settings,
        ),
        patch.object(
            pb_serve_cmd,
            "load_package_b_operator_production_control_plane_governance_from_env",
        ),
        patch.object(pb_serve_cmd, "load_strict_production_serving_environment_from_env"),
        patch.object(pb_serve_cmd, "load_single_node_production_profile_v1_from_env"),
        patch.object(pb_serve_cmd, "validate_configured_single_node_production_profile_v1"),
        patch.object(pb_serve_cmd, "VenvBundleColocatedRuntimeDeploymentAdapter"),
        settings,
    )


def test_pb_s03_s04_sanctioned_gca_and_d4_factory_in_serve_command() -> None:
    text = _SERVE_COMMAND.read_text(encoding="utf-8")
    assert (
        "open_gca_strict_production_serving_session_from_single_node_production_profile_v1"
        in text
    )
    assert "build_governed_contractor_strict_production_uvicorn_factory" in text
    assert "_LEGACY_STRICT_SERVE_MARKERS" in text
    assert "host.main:app" in text
    assert "open_strict_production_serving_session(" not in text

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sys,
            "argv",
            ["package_b_strict_production_serving_command", "host.main:app"],
        )
        with pytest.raises(
            pb_serve_cmd.PackageBStrictProductionServingCommandError,
            match="legacy strict production entry 'host.main:app'",
        ):
            pb_serve_cmd._reject_legacy_strict_serve_invocation()


def test_pb_s07_post_session_open_factory_failure_closes_session_and_releases_writer() -> (
    None
):
    session = _package_b_command_ready_session()
    release_calls = 0

    def _release_writer() -> None:
        nonlocal release_calls
        release_calls += 1

    session.bind_process_scoped_release_on_close(_release_writer)
    patches = _patch_package_b_serve_preamble()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(
            pb_serve_cmd,
            "open_gca_strict_production_serving_session_from_single_node_production_profile_v1",
            return_value=(session, MagicMock(), MagicMock(), MagicMock()),
        ):
            with patch.object(
                pb_serve_cmd,
                "build_governed_contractor_strict_production_uvicorn_factory",
                side_effect=RuntimeError("factory-fail"),
            ):
                with pytest.raises(
                    pb_serve_cmd.PackageBStrictProductionServingCommandError,
                    match="factory-fail",
                ):
                    pb_serve_cmd.run_package_b_strict_production_serving_command()
    assert release_calls == 1


def test_pb_s08_uvicorn_failure_with_failed_close_retains_writer_lock() -> None:
    session = _package_b_command_ready_session()
    release_calls = 0

    def _release_writer() -> None:
        nonlocal release_calls
        release_calls += 1

    session.bind_process_scoped_release_on_close(_release_writer)

    def _close_raises(self: object) -> None:
        raise StrictProductionServingLauncherError("pb_close_drain_fail")

    mock_uvicorn = MagicMock()
    mock_uvicorn.run.side_effect = OSError("uvicorn-bind-fail")
    patches = _patch_package_b_serve_preamble()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch.object(
            pb_serve_cmd,
            "open_gca_strict_production_serving_session_from_single_node_production_profile_v1",
            return_value=(session, MagicMock(), MagicMock(), MagicMock()),
        ):
            with patch.object(
                pb_serve_cmd,
                "build_governed_contractor_strict_production_uvicorn_factory",
                return_value=lambda: MagicMock(),
            ):
                with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
                    with patch.object(
                        StrictProductionServingSession,
                        "close",
                        _close_raises,
                    ):
                        with pytest.raises(
                            StrictProductionServingLauncherError,
                            match="pb_close_drain_fail",
                        ):
                            pb_serve_cmd.run_package_b_strict_production_serving_command()
    assert release_calls == 0


def test_pb_s05_missing_operator_governance_configuration_fails_closed() -> None:
    reader = EnvReader("INTERGRAX_OPERATOR_PRODUCTION_")
    with pytest.raises(PackageBOperatorProductionGovernanceConfigurationError):
        load_package_b_operator_production_control_plane_governance_from_env(env=reader)


def test_pb_s06_docker_qualification_defaults_to_sanctioned_serve_module() -> None:
    text = _COMPOSE.read_text(encoding="utf-8")
    assert "package_b_strict_production_serving_command" in text
    assert "host.main:app" not in text
