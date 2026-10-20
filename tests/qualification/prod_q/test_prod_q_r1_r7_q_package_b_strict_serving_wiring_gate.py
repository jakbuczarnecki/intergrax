# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q Package B — strict serving wiring (PB-S01…PB-S06 static + fail-closed)."""

from __future__ import annotations

import subprocess
from typing import Final

import pytest

from intergrax.applications._shared.package_b_operator_production_governance_configuration import (
    PackageBOperatorProductionGovernanceConfigurationError,
    load_package_b_operator_production_control_plane_governance_from_env,
)
from intergrax.applications._shared.settings_loader import EnvReader
from tests.qualification.prod_q.test_prod_q_r1_r7_q_od_r2_03_single_node_production_profile_v1_gate import (
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


def test_pb_s03_s04_sanctioned_gca_and_d4_factory_in_serve_command() -> None:
    text = _SERVE_COMMAND.read_text(encoding="utf-8")
    assert (
        "open_gca_strict_production_serving_session_from_single_node_production_profile_v1"
        in text
    )
    assert "build_governed_contractor_strict_production_uvicorn_factory" in text
    assert "host.main:app" not in text
    assert "open_strict_production_serving_session(" not in text


def test_pb_s05_missing_operator_governance_configuration_fails_closed() -> None:
    reader = EnvReader("INTERGRAX_OPERATOR_PRODUCTION_")
    with pytest.raises(PackageBOperatorProductionGovernanceConfigurationError):
        load_package_b_operator_production_control_plane_governance_from_env(env=reader)


def test_pb_s06_docker_qualification_defaults_to_sanctioned_serve_module() -> None:
    text = _COMPOSE.read_text(encoding="utf-8")
    assert "package_b_strict_production_serving_command" in text
    assert "host.main:app" not in text
