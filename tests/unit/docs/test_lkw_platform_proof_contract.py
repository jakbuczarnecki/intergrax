# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LKW_PLATFORM_PROOF = _REPO_ROOT / "docs/public-adoption/LKW_PLATFORM_PROOF.md"
_SCRIPTS = _REPO_ROOT / "applications/local_workspace_application/scripts"
_DOCKER = _REPO_ROOT / "applications/local_workspace_application/docker"


def test_lkw_platform_proof_step_4_is_self_contained() -> None:
    text = _LKW_PLATFORM_PROOF.read_text(encoding="utf-8")
    assert "## Step 4 — Open the generated Sentry issue" in text
    assert "Use this local proof login:" in text
    assert "admin@intergrax.local" in text
    assert "proof-local-only" in text
    assert (
        "http://127.0.0.1:9000/organizations/intergrax-local/issues/?project=2"
        in text
    )
    assert (
        "Use the local proof credentials from "
        "`applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md`"
        not in text
    )


def test_lkw_platform_proof_uses_canonical_reviewer_commands() -> None:
    text = _LKW_PLATFORM_PROOF.read_text(encoding="utf-8")
    assert (
        r"applications\local_workspace_application\scripts\hard-reset-local-docker-all.bat"
        in text
    )
    assert (
        r"applications\local_workspace_application\scripts\check-lkw-platform-proof-status.bat"
        in text
    )
    assert (
        r"applications\local_workspace_application\scripts\run-sentry-observability-proof.bat --run-id lkw-sentry-live-001 --correlation-id lkw-sentry-live-001"
        in text
    )
    assert (
        r"applications\local_workspace_application\scripts\run-lkw-elasticsearch-proof.bat"
        in text
    )
    assert (
        r"applications\local_workspace_application\scripts\run-lkw-persistence-proof.bat"
        in text
    )
    assert (
        r"applications\local_workspace_application\scripts\run-lkw-background-task-proof.bat"
        in text
    )


def test_lkw_platform_proof_documents_persistent_storage_step() -> None:
    text = _LKW_PLATFORM_PROOF.read_text(encoding="utf-8")
    assert "persistent local knowledge" in text
    assert "non-destructive restart" in text
    assert "volumes_removed=false" in text
    assert "reindexed_after_restart=false" in text
    assert "Do not use hard-reset-local-docker-all" in text
    assert "LKW persists indexed local knowledge across a non-destructive restart." in text
    assert "LKW_5_PERSISTENCE_VERIFICATION.md" in text


def test_persistence_proof_bat_delegates_to_powershell() -> None:
    text = (_SCRIPTS / "run-lkw-persistence-proof.bat").read_text(encoding="utf-8")
    assert "run-lkw-persistence-proof.ps1" in text
    assert "-NoProfile" in text
    assert "-ExecutionPolicy Bypass" in text
    assert '-File "%PROOF%"' in text
    assert "exit /b %ERRORLEVEL%" in text


def test_persistence_proof_helper_avoids_destructive_restart_commands() -> None:
    text = (_SCRIPTS / "run-lkw-persistence-proof.ps1").read_text(encoding="utf-8")
    assert "down -v" not in text
    assert "--volumes" not in text
    assert "Remove-Volume" not in text
    assert "hard-reset-local-docker-all" not in text


def test_persistence_proof_helper_implements_reviewer_contract() -> None:
    text = (_SCRIPTS / "run-lkw-persistence-proof.ps1").read_text(encoding="utf-8")
    for needle in (
        "/v1/local_workspace/run",
        "local.workspace.index",
        "local.workspace.search",
        "docker compose",
        "restart",
        "proof_result=PASS",
        "volumes_removed=false",
        "reindexed_after_restart=false",
    ):
        assert needle in text


def test_windows_hard_reset_launcher_delegates_to_powershell() -> None:
    text = (_SCRIPTS / "hard-reset-local-docker-all.bat").read_text(encoding="utf-8")
    assert "hard-reset-local-docker-all.ps1" in text
    assert "-NoProfile" in text
    assert "-ExecutionPolicy Bypass" in text
    assert '-File "%LAUNCHER%"' in text
    assert "exit /b %ERRORLEVEL%" in text


def test_windows_status_checker_delegates_to_powershell() -> None:
    text = (
        _SCRIPTS / "check-lkw-platform-proof-status.bat"
    ).read_text(encoding="utf-8")
    assert "check-lkw-platform-proof-status.ps1" in text
    assert "-NoProfile" in text
    assert "-ExecutionPolicy Bypass" in text
    assert '-File "%CHECKER%"' in text
    assert "exit /b %ERRORLEVEL%" in text
    assert "proof_status=FAIL" in text


def test_elasticsearch_proof_helper_switches_backend_before_run() -> None:
    text = (_SCRIPTS / "run-lkw-elasticsearch-proof.bat").read_text(encoding="utf-8")
    assert "switching local_workspace to Elasticsearch observability backend" in text
    assert '"%BASE_COMPOSE%"' in text
    assert '"%ES_COMPOSE%"' in text
    assert "up -d --build local_workspace" in text
    assert "LOCAL_WORKSPACE_OBSERVABILITY_PROOF_KIBANA_URL" in text
    assert "http://127.0.0.1:5601" in text


def test_elasticsearch_proof_helper_waits_for_lkw_health_before_post() -> None:
    text = (_SCRIPTS / "run-lkw-elasticsearch-proof.bat").read_text(encoding="utf-8")
    assert "Waiting for LKW health after backend switch" in text
    assert "%LKW_BASE_URL%/health" in text
    assert "$response.status -eq 'ok'" in text
    health_wait_index = text.index("Waiting for LKW health after backend switch")
    post_index = text.index("/v1/local_workspace/run")
    assert health_wait_index < post_index


def test_elasticsearch_overlay_configures_policy_safe_backend() -> None:
    text = (
        _DOCKER / "docker-compose.elasticsearch.yml"
    ).read_text(encoding="utf-8")
    assert 'LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED: "true"' in text
    assert "LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND: elasticsearch" in text
    assert (
        "LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX: intergrax-lkw-observability"
        in text
    )
    assert 'LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_CONTENT: "false"' in text


def test_sentry_events_consumer_waits_for_kafka_topics() -> None:
    text = (_DOCKER / "sentry.services.yml").read_text(encoding="utf-8")
    assert "sentry-kafka-topics:" in text
    assert "--topic ingest-events" in text
    assert "sentry-events-consumer:" in text
    assert "run consumer ingest-events" in text
    kafka_topics_index = text.index("sentry-kafka-topics:")
    events_consumer_index = text.index("sentry-events-consumer:")
    assert kafka_topics_index < events_consumer_index
    consumer_block = text[events_consumer_index:]
    assert "condition: service_completed_successfully" in consumer_block
