# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

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
_OLD_SENTRY_SERVICES_FRAGMENT = _DOCKER_DIR / "docker-compose.sentry.services.yml"
_RUN_LOCAL_DOCKER_ALL_BAT = _SCRIPTS_DIR / "run-local-docker-all.bat"
_RUN_LOCAL_DOCKER_ALL_SH = _SCRIPTS_DIR / "run-local-docker-all.sh"
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


def test_sentry_secret_key_in_shared_env() -> None:
    services = _SENTRY_SERVICES_FRAGMENT.read_text(encoding="utf-8")
    assert "SENTRY_SECRET_KEY:" in services
    assert _LOCAL_PROOF_SECRET in services
    assert "INTERGRAX_LOCAL_SENTRY_SECRET_KEY" in services


def test_docs_mention_canonical_all_in_one_startup() -> None:
    sentry_doc = (_DOCS_DIR / "SENTRY_OBSERVABILITY.md").read_text(encoding="utf-8")
    platform_proof = (
        _PROJECT_ROOT / "docs" / "public-adoption" / "LKW_PLATFORM_PROOF.md"
    ).read_text(encoding="utf-8")

    for doc in (sentry_doc, platform_proof):
        assert "run-local-docker-all.bat" in doc
        assert "run-local-docker-all.sh" in doc
