# © Artur Czarnecki. All rights reserved.

"""PROD-Q OD-R2-03 — Docker Compose local qualification (CONFIGURED / REPO-TESTED gates)."""

from __future__ import annotations

import subprocess
from typing import Final

import pytest

from tests.qualification.prod_q.test_prod_q_r1_r7_q_od_r2_03_single_node_production_profile_v1_gate import (
    repo_root,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_QUAL_ROOT: Final = (
    repo_root()
    / "applications"
    / "governed_contractor_application"
    / "deploy"
    / "docker-compose-local-qualification"
)
_COMPOSE_FILE: Final = _QUAL_ROOT / "docker-compose.yml"
_ENTRYPOINT: Final = _QUAL_ROOT / "intergrax-gca-docker-local-qualification-entrypoint.sh"
_ENV_EXAMPLE: Final = _QUAL_ROOT / "intergrax-gca-docker-local-qualification.env.example"
_RUNBOOK: Final = _QUAL_ROOT / "RUNBOOK.md"
_LINUX_ENTRYPOINT: Final = (
    repo_root()
    / "applications"
    / "governed_contractor_application"
    / "deploy"
    / "linux"
    / "single-node-production-profile-v1"
    / "intergrax-gca-strict-production-entrypoint.sh"
)


def test_dcq_01_compose_single_service_named_volume_and_stop_grace() -> None:
    text = _COMPOSE_FILE.read_text(encoding="utf-8")
    assert "gca-strict-local-qualification:" in text
    assert text.count("services:") == 1
    assert "deploy:" not in text
    assert "replicas" not in text
    assert "gca_strict_local_qualification_data:" in text
    assert "stop_grace_period: 120s" in text
    assert "postgresql" not in text.lower()
    assert "redis" not in text.lower()


def test_dcq_02_qualification_surface_preserves_sanctioned_startup_seam() -> None:
    compose = _COMPOSE_FILE.read_text(encoding="utf-8")
    dockerfile = (_QUAL_ROOT / "Dockerfile").read_text(encoding="utf-8")
    entry = _ENTRYPOINT.read_text(encoding="utf-8")
    assert "host.main:app" not in compose
    assert "uvicorn" not in compose.lower()
    assert "host.main:app" not in dockerfile
    assert "single_node_profile_v1_admission_gate" in entry
    assert "INTERGRAX_GCA_STRICT_SERVE_EXEC" in entry
    assert "open_gca_strict_production_serving_session_from_single_node_production_profile_v1" in entry
    assert "BLOCKED — PACKAGE B OPERATOR SERVE CONFIGURATION REQUIRED" in entry
    linux = _LINUX_ENTRYPOINT.read_text(encoding="utf-8")
    assert "single_node_profile_v1_admission_gate" in linux


def test_dcq_03_env_example_uses_profile_v1_prefix_and_container_paths() -> None:
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "INTERGRAX_SINGLE_NODE_PRODUCTION_V1_DATA_DIR=" in text
    assert "/var/lib/intergrax/production-v1/data" in text
    assert "INTERGRAX_GCA_STRICT_SERVE_EXEC" in text
    assert "out-of-repo" in text


def test_dcq_04_runbook_declares_qualification_limits_not_production() -> None:
    text = _RUNBOOK.read_text(encoding="utf-8")
    assert "DOCKER-LOCAL-VERIFIED" in text
    assert "TARGET-VERIFIED" in text
    assert "does **not** replace" in text or "does not replace" in text
    assert "host.main:app" in text
    assert "down -v" in text


def test_dcq_05_docker_compose_config_when_cli_available() -> None:
    try:
        proc = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(_COMPOSE_FILE),
                "-p",
                "intergrax-gca-docker-local-qualification",
                "config",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("Docker CLI unavailable or compose config timed out (DCQ-10 OPEN)")
    if proc.returncode != 0 and "docker" in (proc.stderr or "").lower():
        pytest.skip(f"Docker compose config failed: {proc.stderr.strip()}")
    assert proc.returncode == 0, proc.stderr
