# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LKW_PLATFORM_PROOF = _REPO_ROOT / "docs/project/proofs/LKW_PLATFORM_PROOF.md"
_SCRIPTS = _REPO_ROOT / "applications/local_workspace_application/scripts"
_DOCKER = _REPO_ROOT / "applications/local_workspace_application/docker"
_PROOFS = _REPO_ROOT / "docs/project/proofs/PROOFS.md"
_MD_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _github_heading_slug(title: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    )
    normalized = re.sub(r"[^\w\s-]", "", normalized.lower().strip())
    normalized = re.sub(r"[\s_-]+", "-", normalized)
    return re.sub(r"^-+|-+$", "", normalized)


def _lkw_heading_slugs(text: str) -> set[str]:
    return {
        _github_heading_slug(match.group(1))
        for match in re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    }


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
_IMPL_PLAN_HISTORICAL = (
    _REPO_ROOT
    / "applications/local_workspace_application/docs/archive/IMPLEMENTATION_PLAN_2026-07-22.md"
)
_LKW_ARCHITECTURE = (
    _REPO_ROOT / "applications/local_workspace_application/docs/ARCHITECTURE.md"
)
_PROOF_RECEIPTS = _REPO_ROOT / "docs/project/architecture/PROOF_RECEIPTS.md"
_RUNTIME_ARCHITECTURE = _REPO_ROOT / "docs/project/architecture/intergrax_runtime_architecture.md"
_FILE_WATCHER_SYNC_DOCS = (
    _IMPL_PLAN,
    _LKW_ARCHITECTURE,
    _PROOF_RECEIPTS,
    _RUNTIME_ARCHITECTURE,
)
_STALE_FILE_WATCHER_STEP_REFS = (
    "Public reviewer Steps 14–15",
    "File Watcher Steps 14–15",
    "Step 14 — Run the File Watcher E2E proof",
    "Step 15 — Inspect the File Watcher ProofReceipt",
    "Steps 14–15",
    "Public Steps:** 14–15",
    " / 14–15",
)
_CURRENT_FILE_WATCHER_STEP_HEADINGS = (
    "Step 12 — Run the File Watcher E2E proof",
    "Step 13 — Inspect the File Watcher ProofReceipt in Mongo Express",
)


def _proof_text() -> str:
    return _LKW_PLATFORM_PROOF.read_text(encoding="utf-8")


def _lkw7c2_reviewer_path_snippet(text: str) -> str | None:
    marker = "LKW.7C2 records the live workload"
    if marker not in text:
        return None
    idx = text.index(marker)
    return text[idx : idx + 500]


def _has_current_file_watcher_compact_or_headings(text: str) -> bool:
    if all(heading in text for heading in _CURRENT_FILE_WATCHER_STEP_HEADINGS):
        return True
    return (
        "Public reviewer Steps 12–13" in text
        or "File Watcher Steps 12–13" in text
        or "Public Steps:** Steps 12–13" in text
        or "/ Steps 12–13" in text
    )


def _has_numbered_file_watcher_public_ref(text: str) -> bool:
    if any(stale in text for stale in _STALE_FILE_WATCHER_STEP_REFS):
        return True
    if _has_current_file_watcher_compact_or_headings(text):
        return True
    snippet = _lkw7c2_reviewer_path_snippet(text)
    if snippet is None:
        return False
    return "Steps 12–13" in snippet or "Steps 14–15" in snippet


def _uses_current_file_watcher_public_numbering(text: str) -> bool:
    if _has_current_file_watcher_compact_or_headings(text):
        return True
    snippet = _lkw7c2_reviewer_path_snippet(text)
    if snippet is None:
        return False
    return "Steps 12–13" in snippet and "Steps 14–15" not in snippet


def test_proofs_doc_core_platform_proof_deep_link_resolves() -> None:
    proofs_text = _PROOFS.read_text(encoding="utf-8")
    lkw_text = _proof_text()
    heading_slugs = _lkw_heading_slugs(lkw_text)

    fragments = [
        target.split("#", 1)[1]
        for target in _MD_LINK.findall(proofs_text)
        if target.startswith("LKW_PLATFORM_PROOF.md#")
    ]
    assert fragments, "PROOFS.md must link to anchored LKW Platform Proof sections"
    assert all(fragment in heading_slugs for fragment in fragments), fragments

    core_platform_heading = lkw_text.split("## Core Platform Proof", 1)[1].split("\n", 1)[0]
    assert core_platform_heading == ""
    assert "core-platform-proof" in fragments


def test_lkw_platform_proof_core_and_optional_headings() -> None:
    text = _proof_text()
    for heading in (
        "## Core Platform Proof",
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
    assert "live-certified in Linux Docker runtime" in linux_section
    assert "not separately certified" in linux_section
    assert "run-lkw-linux-interaction-proof.sh" in linux_section
    assert "run-lkw-linux-container-certification.bat" in linux_section
    assert "LKW_LINUX_DOCKER_CERTIFICATION.json" in linux_section
    assert "proof_kind=platform_linux_interaction" in linux_section
    assert "lkw.linux_shell" in linux_section
    assert "native Linux host" in linux_section.lower() or (
        "Native Linux host" in linux_section
    )
    assert "every Linux distribution" in linux_section


def test_lkw_platform_proof_macos_honesty() -> None:
    text = _proof_text()
    start = text.index("## macOS users — Optional interaction proof")
    end = text.index("## Core reviewer shortcuts")
    macos_section = text[start:end]
    assert "Status: implemented, not live-certified" in macos_section
    assert "not live-certified" in macos_section
    assert "run-lkw-macos-interaction-proof.sh" in macos_section
    assert "proof_kind=platform_macos_interaction" in macos_section
    assert "lkw.macos_shell" in macos_section
    assert "proof_result=PASS" not in macos_section
    assert "live-certified" not in macos_section.replace("not live-certified", "")


def test_lkw_platform_proof_certification_matrix() -> None:
    text = _proof_text()
    assert "live-certified on native Windows" in text
    assert "live-certified in Linux Docker runtime" in text
    assert "Implemented, not certified" in text
    assert "LKW_PLATFORM_CERTIFICATION_MATRIX.md" in text
    assert "LKW_PLATFORM_CERTIFICATION_MATRIX.json" in text
    assert "PROOF-PORTABILITY-1D-MATRIX" in text
    assert "remain not live-certified" in text
    windows_row_idx = text.index("| Windows")
    linux_row_idx = text.index("| Linux")
    macos_row_idx = text.index("| macOS")
    windows_row = text[windows_row_idx : windows_row_idx + 320]
    assert "Application hosting + interaction live-certified on native Windows" in (
        windows_row
    )
    assert "windows_native_runtime" in windows_row
    linux_row = text[linux_row_idx : linux_row_idx + 420]
    assert "Application hosting + interaction live-certified in Linux Docker runtime" in (
        linux_row
    )
    assert "full multi-phase Core Platform Proof not separately certified" in linux_row
    assert "Implemented, not certified" in text[macos_row_idx : macos_row_idx + 220]
    assert "Shared Python runner through Windows BAT" in text
    assert "Shared Python runner through Linux SH" in text
    assert "Shared Python runner through macOS SH" in text
    assert "PROOF-PORTABILITY-1B" in text
    assert "PROOF-PORTABILITY-1C" in text
    assert "native Linux host" in text.lower() or "Native Linux host" in text
    assert "LKW_WINDOWS_NATIVE_CERTIFICATION.json" in text
    assert (
        _SCRIPTS / "run-lkw-windows-native-certification.py"
    ).is_file()
    assert (
        _SCRIPTS / "run-lkw-windows-native-certification.bat"
    ).is_file()
    assert (
        _SCRIPTS / "generate-lkw-platform-certification-matrix.py"
    ).is_file()
    matrix_md = (
        _REPO_ROOT / "docs/project/maintainers/public-adoption/LKW_PLATFORM_CERTIFICATION_MATRIX.md"
    )
    assert matrix_md.is_file()
    matrix_text = matrix_md.read_text(encoding="utf-8")
    assert "live-certified" in matrix_text
    assert "not live-certified" in matrix_text
    assert "Application Hosting certification is not the same as complete" in (
        matrix_text
    )


def _proof_portability_section(text: str, task_id: str) -> str:
    heading = f"### {task_id}"
    assert heading in text, f"missing authoritative heading {heading}"
    start = text.index(heading)
    next_heading = text.find("\n### ", start + len(heading))
    end = next_heading if next_heading != -1 else start + 1200
    return text[start:end]


def test_lkw_platform_proof_plan_portability_contract() -> None:
    # Current product-first plan points at the historical full plan for the
    # structured PROOF-PORTABILITY status records; do not bind to coincidental
    # first-string matches in the active roadmap.
    current = _IMPL_PLAN.read_text(encoding="utf-8")
    assert "archive/IMPLEMENTATION_PLAN_2026-07-22.md" in current
    assert "LKW_PLATFORM_PROOF.md" in current

    text = _IMPL_PLAN_HISTORICAL.read_text(encoding="utf-8")
    section_1a = _proof_portability_section(text, "PROOF-PORTABILITY-1A")
    section_1b = _proof_portability_section(text, "PROOF-PORTABILITY-1B")
    section_1c = _proof_portability_section(text, "PROOF-PORTABILITY-1C")
    section_1d = _proof_portability_section(text, "PROOF-PORTABILITY-1D")
    section_matrix = _proof_portability_section(text, "PROOF-PORTABILITY-1D-MATRIX")

    assert "**Status:** **Done**" in section_1a
    assert "**Status:** **Done**" in section_1b
    assert "**Status:** **Done**" in section_1c
    assert "**Status:** **Partial**" in section_1d
    assert "LKW_PLATFORM_CERTIFICATION_MATRIX.md" in section_matrix
    assert "live-certified on native Windows through current shared runner" in text
    assert "live-certified in Linux Docker runtime" in text
    assert "implemented, not live-certified" in text or (
        "implemented, not live-certified" in text.lower()
    )
    assert "not separately certified" in text
    assert "Linux Application Hosting Proof" in text
    assert "Windows Application Hosting Proof" in text
    assert "full multi-phase Core Platform Proof" in text
    assert "linux_docker_runtime" in text
    assert "windows_native_runtime" in text
    assert "LKW_WINDOWS_NATIVE_CERTIFICATION.json" in text
    assert "macos_native_runtime" in text or "macOS" in text
    assert "not live-certified" in text


def test_lkw_platform_proof_shared_os_interaction_architecture() -> None:
    text = _proof_text()
    assert "invoke-lkw-interaction.py" in text
    assert "run-lkw-os-interaction-proof.py" in text
    assert "Windows PowerShell wrapper" in text
    assert "Linux shell wrapper" in text
    assert "macOS shell wrapper" in text
    assert "lkw.windows_powershell" in text
    assert "lkw.linux_shell" in text
    assert "lkw.macos_shell" in text
    assert "platform_windows_interaction" in text
    assert "platform_linux_interaction" in text
    assert "platform_macos_interaction" in text
    assert (
        "Windows Application Hosting Proof:\n"
        "  live-certified on native Windows through current shared runner"
        in text
    )
    assert (
        "Windows Optional OS Interaction Proof:\n"
        "  live-certified on native Windows through shared Python client/proof runner"
        in text
    )
    assert (
        "Linux Application Hosting Proof:\n  live-certified in Linux Docker runtime"
        in text
    )
    assert (
        "Linux Optional OS Interaction Proof:\n"
        "  live-certified in Linux Docker runtime"
        in text
    )
    assert (
        "Linux full multi-phase Core Platform Proof:\n"
        "  not separately certified by Linux Docker profile"
        in text
    )
    assert "macOS:\n  implemented, not live-certified" in text
    assert "Linux native-host deployment:\n  not separately certified" in text
    assert "planned, not implemented, not certified" not in text
    assert (_SCRIPTS / "invoke-lkw-interaction.py").is_file()
    assert (_SCRIPTS / "run-lkw-os-interaction-proof.py").is_file()
    assert (_SCRIPTS / "run-lkw-linux-container-certification.py").is_file()
    assert (_SCRIPTS / "run-lkw-linux-container-certification.bat").is_file()
    assert (_SCRIPTS / "run-lkw-windows-native-certification.py").is_file()
    assert (_SCRIPTS / "run-lkw-windows-native-certification.bat").is_file()
    assert (
        _DOCKER / "Dockerfile.linux-certification"
    ).is_file()
    assert (_DOCKER / "linux-certification.compose.yml").is_file()
    compose = (_DOCKER / "linux-certification.compose.yml").read_text(encoding="utf-8")
    assert "docker.sock" not in compose
    assert "lkw-linux-certification-mongodb" in compose
    assert "lkw-linux-certification:" in compose
    assert "privileged: true" not in compose


def test_lkw_linux_docker_certification_docs_do_not_overclaim() -> None:
    text = _proof_text()
    forbidden = (
        "Linux fully certified on native host",
        "Linux native deployment certified",
        "all Linux distributions certified",
        "Linux desktop certified",
        "Linux Core Platform Proof certified",
        "Linux core proof certified",
        "full Linux platform proof certified",
        "Linux fully certified",
        "Linux native certified",
        "all Linux environments certified",
    )
    for phrase in forbidden:
        assert phrase not in text
    assert "live-certified in Linux Docker runtime" in text
    assert "not separately certified" in text
    assert "Linux Application Hosting Proof" in text
    assert "full multi-phase Core Platform Proof" in text
    assert "linux_docker_runtime" in text
    # Forbidden heading for application-hosting-only blocks.
    assert "\nCore proof\n" not in text
    assert "Linux core proof:" not in text


def test_lkw_platform_proof_core_numbering() -> None:
    text = _proof_text()
    assert "Step 12 — Run the File Watcher E2E proof" in text
    assert "Step 13 — Inspect the File Watcher ProofReceipt in Mongo Express" in text
    assert "Step 12 — Run the Windows PowerShell interaction proof" not in text
    assert "Step 13 — Inspect the Windows Interaction ProofReceipt" not in text


def test_lkw_platform_proof_document_ordering() -> None:
    text = _proof_text()
    markers = [
        "## Core Platform Proof",
        "## Core prerequisites",
        "## Recommended one-command Core Platform Proof",
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


def test_lkw_platform_proof_shared_core_entrypoint_commands() -> None:
    text = _proof_text()
    assert "## Recommended one-command Core Platform Proof" in text
    assert (
        r"applications\local_workspace_application\scripts\run-lkw-core-platform-proof-windows.bat"
        in text
    )
    assert (
        "./applications/local_workspace_application/scripts/"
        "run-lkw-core-platform-proof-linux.sh"
    ) in text
    assert (
        "./applications/local_workspace_application/scripts/"
        "run-lkw-core-platform-proof-macos.sh"
    ) in text
    assert "All three launchers invoke the same Python implementation." in text
    assert "The launchers contain no proof workload or acceptance logic." in text
    assert (_SCRIPTS / "run-lkw-core-platform-proof.py").is_file()
    assert (_SCRIPTS / "run-lkw-core-platform-proof-windows.bat").is_file()
    assert (_SCRIPTS / "run-lkw-core-platform-proof-linux.sh").is_file()
    assert (_SCRIPTS / "run-lkw-core-platform-proof-macos.sh").is_file()


def test_lkw_platform_proof_powershell_optional_only() -> None:
    text = _proof_text()
    commands_start = text.index("## Current reviewer-command requirements")
    commands_end = text.index("## Recommended one-command Core Platform Proof")
    commands_section = text[commands_start:commands_end]
    assert (
        "Windows PowerShell required only for the optional Windows" in commands_section
    )
    assert "PowerShell required only for the optional" in commands_section
    core_start = text.index("## Core prerequisites")
    core_section = text[core_start:commands_start]
    assert "PowerShell" not in core_section


def test_file_watcher_public_reviewer_step_references_are_synchronized() -> None:
    for path in _FILE_WATCHER_SYNC_DOCS:
        text = path.read_text(encoding="utf-8")
        for stale in _STALE_FILE_WATCHER_STEP_REFS:
            assert stale not in text, f"{path.name} still contains {stale!r}"
        if _has_numbered_file_watcher_public_ref(text):
            assert _uses_current_file_watcher_public_numbering(text), (
                f"{path.name} lacks current File Watcher Steps 12–13 numbering"
            )


def test_lkw_platform_proof_indexed_hybrid_ask_web_url_claim_boundary() -> None:
    text = _proof_text()
    assert "indexed evidence branch of production Hybrid Ask" in text
    assert "real RAG ingest" in text
    assert "exact tenant/workspace vector scope" in text
    assert "production Hybrid Ask composition" in text
    assert (
        "test_web_url_end_to_end_real_rag_ask_proof" in text
    ), "Must reference accepted Web URL / real-RAG verification path"
    assert (
        "test_count_exact_workspace_scope_does_not_masquerade_as_tenant_only" in text
    ), "Must reference exact workspace-scope verification path"
    assert (
        "This proof does not establish combined indexed + live evidence, "
        "complete live-provider behavior, or complete Hybrid Ask."
    ) in text
    lower = text.lower()
    assert "complete hybrid ask is complete" not in lower
    assert "hybrid ask is complete" not in lower
    # Positive completion claims without the mixed-evidence qualifier are forbidden.
    assert "complete Hybrid Ask." in text  # limitation sentence only
    assert "Not claimed today:" in text
    not_claimed = text[text.index("Not claimed today:") : text.index("Not claimed today:") + 400]
    assert "Hybrid Ask combining indexed and authorized live evidence" in not_claimed
    assert not_claimed.count("Hybrid Ask;") == 0
