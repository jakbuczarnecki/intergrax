# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LKW_ROOT = _PROJECT_ROOT / "applications" / "local_workspace_application"
_DOCKER_DIR = _LKW_ROOT / "docker"
_SCRIPTS_DIR = _LKW_ROOT / "scripts"
_DOCS_DIR = _LKW_ROOT / "docs"

_SENTRY_OVERLAY = _DOCKER_DIR / "docker-compose.sentry.yml"
_SENTRY_SERVICES_FRAGMENT = _DOCKER_DIR / "sentry.services.yml"
_SENTRY_CONF = _DOCKER_DIR / "sentry" / "sentry.conf.py"
_SENTRY_PROOF_DIR = _DOCKER_DIR / "sentry-proof"
_SENTRY_PROOF_PLACEHOLDER = _SENTRY_PROOF_DIR / "generated.env.placeholder"
_SENTRY_PROOF_START_SCRIPT = _DOCKER_DIR / "start-local-workspace-sentry-proof.sh"
_ROOT_GITIGNORE = _PROJECT_ROOT / ".gitignore"
_OLD_SENTRY_SERVICES_FRAGMENT = _DOCKER_DIR / "docker-compose.sentry.services.yml"
_RUN_LOCAL_DOCKER_ALL_BAT = _SCRIPTS_DIR / "run-local-docker-all.bat"
_RUN_LOCAL_DOCKER_ALL_SH = _SCRIPTS_DIR / "run-local-docker-all.sh"
_HARD_RESET_LOCAL_DOCKER_ALL_BAT = _SCRIPTS_DIR / "hard-reset-local-docker-all.bat"
_BOOTSTRAP_SH = _DOCKER_DIR / "sentry" / "bootstrap" / "bootstrap.sh"

_TOP_LEVEL_OVERLAY_PATTERN = re.compile(r"^docker-compose\..+\.yml$")
_LOCAL_PROOF_SECRET = "intergrax-local-sentry-proof-secret-key-not-for-production"


def _discovered_top_level_overlays() -> list[Path]:
    """Mirror run-local-docker-all.* discovery: docker-compose.*.yml only."""
    return sorted(_DOCKER_DIR.glob("docker-compose.*.yml"))


def test_internal_sentry_fragment_does_not_match_overlay_pattern() -> None:
    assert _SENTRY_SERVICES_FRAGMENT.exists()
    assert not _TOP_LEVEL_OVERLAY_PATTERN.match(_SENTRY_SERVICES_FRAGMENT.name)


def test_old_sentry_services_fragment_removed() -> None:
    assert not _OLD_SENTRY_SERVICES_FRAGMENT.exists()


def test_sentry_overlay_includes_internal_fragment() -> None:
    overlay = _SENTRY_OVERLAY.read_text(encoding="utf-8")
    assert _SENTRY_OVERLAY.exists()
    assert "sentry.services.yml" in overlay
    assert "docker-compose.sentry.services.yml" not in overlay


def test_auto_discovery_does_not_double_load_sentry_fragment() -> None:
    overlay_names = [path.name for path in _discovered_top_level_overlays()]
    assert "docker-compose.sentry.yml" in overlay_names
    assert "sentry.services.yml" not in overlay_names
    assert "docker-compose.sentry.services.yml" not in overlay_names


def test_run_local_docker_all_sh_exists_and_discovers_overlays() -> None:
    assert _RUN_LOCAL_DOCKER_ALL_SH.exists()
    script = _RUN_LOCAL_DOCKER_ALL_SH.read_text(encoding="utf-8")
    assert "docker-compose.*.yml" in script
    assert "docker-compose.yml" in script


def test_run_local_docker_all_bat_exists() -> None:
    assert _RUN_LOCAL_DOCKER_ALL_BAT.exists()
    script = _RUN_LOCAL_DOCKER_ALL_BAT.read_text(encoding="utf-8")
    assert "docker-compose.*.yml" in script


def test_hard_reset_local_docker_all_bat_resets_runtime_state_and_starts_stack() -> None:
    assert _HARD_RESET_LOCAL_DOCKER_ALL_BAT.exists()
    script = _HARD_RESET_LOCAL_DOCKER_ALL_BAT.read_text(encoding="utf-8")
    assert "run-local-docker-all.bat" in script
    assert "down -v --remove-orphans" in script
    assert "generated.env" in script
    assert "generated.env.tmp" in script
    assert ".bootstrapped" in script
    assert "up --build" in script
    assert "del /f /q" in script
    assert "APP_DIR%\\.env" not in script
    assert "credentials.json" not in script


def test_sentry_snuba_bootstrap_before_api_and_upgrade() -> None:
    services = _SENTRY_SERVICES_FRAGMENT.read_text(encoding="utf-8")
    assert "sentry-snuba-bootstrap:" in services

    bootstrap_block = services.split("sentry-snuba-bootstrap:", 1)[1].split(
        "\n  sentry-snuba-api:", 1
    )[0]
    assert "command: bootstrap --force" in bootstrap_block
    assert "sentry-clickhouse:" in bootstrap_block
    assert "sentry-kafka:" in bootstrap_block
    assert "sentry-redis:" in bootstrap_block
    assert 'restart: "no"' in bootstrap_block

    api_block = services.split("sentry-snuba-api:", 1)[1].split(
        "\n  sentry-snuba-errors-consumer:", 1
    )[0]
    assert "sentry-snuba-bootstrap:" in api_block
    assert "condition: service_completed_successfully" in api_block

    upgrade_block = services.split("sentry-upgrade:", 1)[1].split("\n  sentry-web:", 1)[0]
    assert "sentry-snuba-api:" in upgrade_block or "sentry-snuba-bootstrap:" in upgrade_block

    worker_block = services.split("sentry-worker:", 1)[1].split("\n  sentry-cron:", 1)[0]
    assert "sentry-web:" in worker_block
    assert "sentry-snuba-bootstrap:" not in worker_block
    assert "sentry-snuba-api:" not in worker_block


def test_sentry_upgrade_runs_before_web() -> None:
    services = _SENTRY_SERVICES_FRAGMENT.read_text(encoding="utf-8")
    assert "sentry-upgrade:" in services
    assert "command: upgrade --noinput" in services

    web_block = services.split("sentry-web:", 1)[1].split("\n  sentry-worker:", 1)[0]
    assert "sentry-upgrade:" in web_block
    assert "condition: service_completed_successfully" in web_block


def test_sentry_bootstrap_does_not_run_upgrade() -> None:
    bootstrap = _BOOTSTRAP_SH.read_text(encoding="utf-8")
    assert "sentry upgrade" not in bootstrap


def test_sentry_bootstrap_team_creation_compatible_with_sentry_24() -> None:
    bootstrap = _BOOTSTRAP_SH.read_text(encoding="utf-8")
    assert "create_default_team" not in bootstrap
    assert "Team.objects.get_or_create" in bootstrap
    assert "Project.objects.get_or_create" in bootstrap
    assert "ProjectKey.objects" in bootstrap
    assert "os.replace" in bootstrap


def test_sentry_bootstrap_membership_compatible_with_sentry_24() -> None:
    bootstrap = _BOOTSTRAP_SH.read_text(encoding="utf-8")
    assert "user=user" not in bootstrap
    assert "user_id=user.id" in bootstrap
    assert "user_email" in bootstrap
    assert "user_is_active" in bootstrap
    assert "role" in bootstrap
    assert "Team.objects.get_or_create" in bootstrap
    assert "Project.objects.get_or_create" in bootstrap
    assert "os.replace" in bootstrap


def test_sentry_bootstrap_sh_has_lf_line_endings_only() -> None:
    content = _BOOTSTRAP_SH.read_bytes()
    assert b"\r\n" not in content
    assert b"\r" not in content
    assert content.split(b"\n", 1)[0] == b"#!/usr/bin/env bash"


def test_sentry_relay_uses_config_directory_not_file() -> None:
    services = _SENTRY_SERVICES_FRAGMENT.read_text(encoding="utf-8")
    assert "sentry-relay:" in services
    assert "command: run -c /work/.relay" in services
    assert "command: run -c /work/.relay/config.yml" not in services


def test_sentry_relay_has_local_proof_credentials_json() -> None:
    credentials_path = _DOCKER_DIR / "sentry" / "relay" / "credentials.json"
    assert credentials_path.is_file()
    payload = json.loads(credentials_path.read_text(encoding="utf-8"))
    for key in ("secret_key", "public_key", "id"):
        assert key in payload
        assert isinstance(payload[key], str) and payload[key]


def test_sentry_proof_runtime_state_gitignored() -> None:
    gitignore = _ROOT_GITIGNORE.read_text(encoding="utf-8")
    assert "applications/local_workspace_application/docker/sentry-proof/generated.env" in gitignore
    assert "applications/local_workspace_application/docker/sentry-proof/.bootstrapped" in gitignore


def test_sentry_overlay_loads_generated_env_at_container_start() -> None:
    overlay = _SENTRY_OVERLAY.read_text(encoding="utf-8")
    assert "./sentry-proof:/proof:ro" in overlay
    assert "start-local-workspace-sentry-proof.sh" in overlay
    assert "path: ./sentry-proof/generated.env" not in overlay


def test_sentry_proof_start_script_sources_generated_env_before_uvicorn() -> None:
    assert _SENTRY_PROOF_START_SCRIPT.is_file()
    script = _SENTRY_PROOF_START_SCRIPT.read_text(encoding="utf-8")
    assert "/proof/generated.env" in script
    assert "sentry-bootstrap must complete before local_workspace starts" in script
    assert "exec uvicorn local_workspace_application.host.main:app" in script


def test_sentry_proof_start_script_has_lf_line_endings_only() -> None:
    content = _SENTRY_PROOF_START_SCRIPT.read_bytes()
    assert b"\r\n" not in content
    assert b"\r" not in content
    assert content.split(b"\n", 1)[0] == b"#!/bin/sh"


def test_sentry_proof_generated_env_placeholder_committed() -> None:
    assert _SENTRY_PROOF_PLACEHOLDER.is_file()
    content = _SENTRY_PROOF_PLACEHOLDER.read_text(encoding="utf-8")
    assert "bootstrap-pending@sentry-relay:3000/1" in content
    assert "LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN=" in content


def test_sentry_secret_key_in_shared_env() -> None:
    services = _SENTRY_SERVICES_FRAGMENT.read_text(encoding="utf-8")
    assert "SENTRY_SECRET_KEY:" in services
    assert _LOCAL_PROOF_SECRET in services
    assert "INTERGRAX_LOCAL_SENTRY_SECRET_KEY" in services


def test_sentry_conf_wires_secret_key_from_env() -> None:
    conf = _SENTRY_CONF.read_text(encoding="utf-8")
    assert "SECRET_KEY" in conf
    assert "SENTRY_SECRET_KEY" in conf
    assert 'SENTRY_OPTIONS["system.secret-key"]' in conf
    assert _LOCAL_PROOF_SECRET in conf or _LOCAL_PROOF_SECRET in _SENTRY_SERVICES_FRAGMENT.read_text(
        encoding="utf-8"
    )


def test_sentry_conf_secret_key_not_empty_literal() -> None:
    conf = _SENTRY_CONF.read_text(encoding="utf-8")
    assert 'SECRET_KEY = ""' not in conf
    assert "SECRET_KEY = ''" not in conf
    assert re.search(r"SECRET_KEY\s*=\s*LOCAL_PROOF_SECRET_KEY", conf)


def test_docs_mention_canonical_all_in_one_startup() -> None:
    sentry_doc = (_DOCS_DIR / "SENTRY_OBSERVABILITY.md").read_text(encoding="utf-8")
    platform_proof = (
        _PROJECT_ROOT / "docs" / "public-adoption" / "LKW_PLATFORM_PROOF.md"
    ).read_text(encoding="utf-8")

    for doc in (sentry_doc, platform_proof):
        assert "run-local-docker-all.bat" in doc
        assert "run-local-docker-all.sh" in doc
