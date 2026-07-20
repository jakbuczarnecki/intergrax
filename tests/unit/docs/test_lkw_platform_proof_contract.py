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
        "http://127.0.0.1:9000/organizations/intergrax-local/issues/?project=2" in text
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
    assert (
        "LKW persists indexed local knowledge across a non-destructive restart." in text
    )
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
    text = (_SCRIPTS / "check-lkw-platform-proof-status.bat").read_text(
        encoding="utf-8"
    )
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
    text = (_DOCKER / "docker-compose.elasticsearch.yml").read_text(encoding="utf-8")
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


_IMPL_PLAN = (
    _REPO_ROOT / "applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md"
)


def _proof_text() -> str:
    return _LKW_PLATFORM_PROOF.read_text(encoding="utf-8")


def test_lkw_platform_proof_core_and_optional_headings() -> None:
    text = _proof_text()
    for heading in (
        "## Core platform claims",
        "## Optional operating-system interaction claims",
        "## Choose your operating system",
        "## Operating-system proof status",
        "## Core Platform Proof completion",
        "# Optional operating-system interaction proofs",
    ):
        assert heading in text


def test_lkw_platform_proof_names_windows_linux_macos() -> None:
    text = _proof_text()
    assert "### Windows" in text
    assert "### Linux" in text
    assert "### macOS" in text
    assert "| Windows" in text
    assert "| Linux" in text
    assert "| macOS" in text


def test_lkw_platform_proof_core_prerequisites_exclude_powershell_on_windows() -> None:
    text = _proof_text()
    core_start = text.index("## Core prerequisites")
    commands_start = text.index("## Current reviewer-command requirements")
    core_section = text[core_start:commands_start]
    assert "PowerShell on Windows" not in core_section


def test_lkw_platform_proof_windows_optional_contract() -> None:
    text = _proof_text()
    assert "Windows users — Optional W1" in text
    assert "Windows users — Optional W2" in text
    assert "This section is optional" in text
    assert "Its omission does not invalidate the Core Platform Proof" in text
    assert "run-lkw-windows-interaction-proof.bat" in text


def test_lkw_platform_proof_linux_honesty() -> None:
    text = _proof_text()
    start = text.index("## Linux users — Optional interaction proof")
    end = text.index("## macOS users — Optional interaction proof")
    linux_section = text[start:end]
    assert "Status: planned" in linux_section
    assert "not implemented" in linux_section
    assert "not certified" in linux_section
    for forbidden in (
        "proof_result=PASS",
        "proof_kind=platform_linux_interaction",
        "run-lkw-linux-interaction-proof",
        "lkw.linux_shell",
    ):
        assert forbidden not in linux_section


def test_lkw_platform_proof_macos_honesty() -> None:
    text = _proof_text()
    start = text.index("## macOS users — Optional interaction proof")
    end = text.index("## Core reviewer shortcuts")
    macos_section = text[start:end]
    assert "Status: planned" in macos_section
    assert "not implemented" in macos_section
    assert "not certified" in macos_section
    for forbidden in (
        "proof_result=PASS",
        "proof_kind=platform_macos_interaction",
        "run-lkw-macos-interaction-proof",
        "lkw.macos_shell",
    ):
        assert forbidden not in macos_section


def test_lkw_platform_proof_core_numbering() -> None:
    text = _proof_text()
    assert "Step 12 — Run the File Watcher E2E proof" in text
    assert "Step 13 — Inspect the File Watcher ProofReceipt in Mongo Express" in text
    assert "Step 12 — Run the Windows PowerShell interaction proof" not in text
    assert "Step 13 — Inspect the Windows Interaction ProofReceipt" not in text


def test_lkw_platform_proof_document_ordering() -> None:
    text = _proof_text()
    markers = [
        "## Core platform claims",
        "## Core prerequisites",
        "## Step 1 — Start a clean local proof stack",
        "## Step 13 — Inspect the File Watcher ProofReceipt in Mongo Express",
        "## Core Platform Proof completion",
        "# Optional operating-system interaction proofs",
        "## Windows users — Optional W1",
        "## Linux users — Optional interaction proof",
        "## macOS users — Optional interaction proof",
    ]
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)


def test_lkw_platform_proof_core_independence() -> None:
    text = _proof_text()
    assert (
        "Skipping an OS-specific interaction proof does not invalidate\n"
        "the Core Platform Proof."
    ) in text
    assert (
        "No operating-system interaction proof is required for core\ncompletion."
    ) in text


def test_lkw_platform_proof_certification_matrix() -> None:
    text = _proof_text()
    assert "Live-certified" in text
    assert "Not certified" in text
    windows_row_idx = text.index("| Windows")
    linux_row_idx = text.index("| Linux")
    macos_row_idx = text.index("| macOS")
    assert "Live-certified" in text[windows_row_idx : windows_row_idx + 200]
    assert "Not certified" in text[linux_row_idx : linux_row_idx + 200]
    assert "Not certified" in text[macos_row_idx : macos_row_idx + 200]
    assert "PROOF-PORTABILITY-1B" in text
    assert "PROOF-PORTABILITY-1C" in text


def test_lkw_platform_proof_plan_portability_contract() -> None:
    text = _IMPL_PLAN.read_text(encoding="utf-8")
    assert "PROOF-PORTABILITY-1A" in text
    assert "PROOF-PORTABILITY-1B" in text
    assert "PROOF-PORTABILITY-1C" in text
    assert "PROOF-PORTABILITY-1D" in text
    a_idx = text.index("PROOF-PORTABILITY-1A")
    b_idx = text.index("PROOF-PORTABILITY-1B")
    c_idx = text.index("PROOF-PORTABILITY-1C")
    d_idx = text.index("PROOF-PORTABILITY-1D")
    assert "**Done**" in text[a_idx : a_idx + 160]
    assert "**Planned**" in text[b_idx : b_idx + 160]
    assert "**Planned**" in text[c_idx : c_idx + 160]
    assert "**Planned**" in text[d_idx : d_idx + 160]
