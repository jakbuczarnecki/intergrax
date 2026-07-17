# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LKW_ROOT = _PROJECT_ROOT / "applications" / "local_workspace_application"
_SCRIPTS_DIR = _LKW_ROOT / "scripts"
_PROOF_SCRIPT = _SCRIPTS_DIR / "run-lkw-windows-interaction-proof.py"
_PROOF_BAT = _SCRIPTS_DIR / "run-lkw-windows-interaction-proof.bat"
_ADAPTER_SCRIPT = _SCRIPTS_DIR / "invoke-lkw-interaction.ps1"
_PUBLIC_PLATFORM_PROOF = (
    _PROJECT_ROOT / "docs" / "public-adoption" / "LKW_PLATFORM_PROOF.md"
)
_LIVE_TEST = (
    _LKW_ROOT / "tests" / "interactions" / "test_windows_powershell_interaction_live.py"
)

_LIVE_NODE = "applications/local_workspace_application/tests/interactions/test_windows_powershell_interaction_live.py::test_windows_powershell_adapter_executes_real_lkw_interactions"

_REQUIRED_EVIDENCE_KEYS = (
    "windows_interaction.hosted_ready",
    "windows_interaction.adapter_invoked",
    "windows_interaction.adapter_id",
    "windows_interaction.powershell_runtime",
    "windows_interaction.transport",
    "windows_interaction.intake_endpoint",
    "windows_interaction.interaction_surface",
    "windows_interaction.interaction_channel",
    "windows_interaction.index_executed",
    "windows_interaction.index_state",
    "windows_interaction.index_task_id",
    "windows_interaction.index_run_id",
    "windows_interaction.search_executed",
    "windows_interaction.search_state",
    "windows_interaction.search_task_id",
    "windows_interaction.search_run_id",
    "windows_interaction.task_ids_distinct",
    "windows_interaction.run_ids_distinct",
    "windows_interaction.graceful_stop",
    "windows_interaction.cleanup_verified",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_proof_module():
    module_name = "lkw_windows_interaction_proof"
    spec = importlib.util.spec_from_file_location(module_name, _PROOF_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sample_evidence(module, **overrides):
    base = dict(
        schema_version="lkw.windows_interaction_proof_evidence.v1",
        hosted_ready=True,
        adapter_invoked=True,
        adapter_id="lkw.windows_powershell",
        powershell_runtime="Windows PowerShell",
        transport="http",
        intake_endpoint="/v1/interactions/intake",
        interaction_surface="lab_json",
        interaction_channel="lab",
        index_executed=True,
        index_state="completed",
        index_task_id="task-index-001",
        index_run_id="run-index-001",
        search_executed=True,
        search_state="completed",
        search_task_id="task-search-001",
        search_run_id="run-search-001",
        task_ids_distinct=True,
        run_ids_distinct=True,
        graceful_stop=True,
        cleanup_verified=True,
    )
    base.update(overrides)
    return module.WindowsInteractionProofEvidence(**base)


def _complete_props() -> dict[str, str]:
    return {
        "windows_interaction.hosted_ready": "true",
        "windows_interaction.adapter_invoked": "true",
        "windows_interaction.adapter_id": "lkw.windows_powershell",
        "windows_interaction.powershell_runtime": "Windows PowerShell",
        "windows_interaction.transport": "http",
        "windows_interaction.intake_endpoint": "/v1/interactions/intake",
        "windows_interaction.interaction_surface": "lab_json",
        "windows_interaction.interaction_channel": "lab",
        "windows_interaction.index_executed": "true",
        "windows_interaction.index_state": "completed",
        "windows_interaction.index_task_id": "task-index-001",
        "windows_interaction.index_run_id": "run-index-001",
        "windows_interaction.search_executed": "true",
        "windows_interaction.search_state": "completed",
        "windows_interaction.search_task_id": "task-search-001",
        "windows_interaction.search_run_id": "run-search-001",
        "windows_interaction.task_ids_distinct": "true",
        "windows_interaction.run_ids_distinct": "true",
        "windows_interaction.graceful_stop": "true",
        "windows_interaction.cleanup_verified": "true",
    }


def _write_junit(
    path: Path,
    *,
    name: str,
    props: dict[str, str],
    status: str | None = None,
) -> None:
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<testsuite name="pytest" tests="1" errors="0" failures="0" skipped="0">',
        f'  <testcase classname="interactions" name="{name}" time="0.1">',
    ]
    if status == "failure":
        parts.append('    <failure message="failed">failed</failure>')
    elif status == "error":
        parts.append('    <error message="errored">errored</error>')
    elif status == "skipped":
        parts.append('    <skipped message="skipped"/>')
    if props:
        parts.append("    <properties>")
        for key, value in props.items():
            parts.append(f'      <property name="{key}" value="{value}"/>')
        parts.append("    </properties>")
    parts.append("  </testcase>")
    parts.append("</testsuite>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def test_powershell_adapter_contract_static() -> None:
    text = _read(_ADAPTER_SCRIPT)
    for required in (
        "/v1/interactions/intake",
        "execute=true",
        "Invoke-RestMethod",
        "ConvertTo-Json",
        "windows_powershell",
        "lkw.windows_powershell",
        "local_workspace.windows_interaction_adapter_result.v1",
        "invalid_adapter_input",
        "interaction_request_failed",
    ):
        assert required in text
    for forbidden in (
        "/v1/local_workspace/run",
        "NexusLoop",
        "LocalWorkspaceTaskExecutor",
        "Task(",
        "rag.",
        "MongoClient",
        "pymongo",
        "Invoke-Expression",
        "Start-Process",
    ):
        assert forbidden not in text
    success_block = text.split("Write-Output", 1)[1]
    assert "Message" not in success_block
    assert "MetadataJson" not in success_block
    assert "raw request body" not in success_block.lower()


def test_build_windows_interaction_proof_receipt_maps_live_evidence() -> None:
    proof = _load_proof_module()
    evidence = _sample_evidence(proof)
    receipt = proof.build_windows_interaction_proof_receipt(
        run_id="run-windows-1",
        correlation_id="corr-windows-1",
        evidence=evidence,
        mongo_express_url="http://127.0.0.1:8086",
    )

    assert (
        receipt.proof_id == "local_workspace:platform_windows_interaction:run-windows-1"
    )
    assert receipt.proof_kind == "platform_windows_interaction"
    assert receipt.application_id == "local_workspace"
    assert receipt.result.value == "PASS"
    assert receipt.task_id is None
    assert receipt.run_id == "run-windows-1"
    assert receipt.correlation_id == "corr-windows-1"

    assert receipt.provider_evidence["os_family"] == "windows"
    assert receipt.provider_evidence["os_adapter"] == "lkw.windows_powershell"
    assert receipt.provider_evidence["client_runtime"] == "Windows PowerShell"
    assert receipt.provider_evidence["transport"] == "http"
    assert receipt.provider_evidence["interaction_surface"] == "lab_json"
    assert receipt.provider_evidence["interaction_channel"] == "lab"
    assert receipt.provider_evidence["intake_endpoint"] == "/v1/interactions/intake"
    assert receipt.provider_evidence["intake_service"] == "InteractionIntakeService"
    assert (
        receipt.provider_evidence["execution_boundary"] == "LocalWorkspaceTaskExecutor"
    )
    assert receipt.provider_evidence["orchestrator"] == "NexusLoop"
    assert (
        receipt.provider_evidence["hosted_entrypoint"]
        == "python_-m_local_workspace_application.hosting"
    )
    assert receipt.provider_evidence["evidence_source"] == "pytest_junit_properties"
    assert receipt.provider_evidence["selected_live_tests"] == 1
    assert receipt.provider_evidence["receipt_document_store_provider"] == "mongodb"

    assert receipt.domain_evidence["hosted_ready"] is True
    assert receipt.domain_evidence["adapter_invoked"] is True
    assert receipt.domain_evidence["adapter_id"] == "lkw.windows_powershell"
    assert receipt.domain_evidence["powershell_runtime"] == "Windows PowerShell"
    assert receipt.domain_evidence["transport"] == "http"
    assert receipt.domain_evidence["intake_endpoint"] == "/v1/interactions/intake"
    assert receipt.domain_evidence["interaction_surface"] == "lab_json"
    assert receipt.domain_evidence["interaction_channel"] == "lab"
    assert receipt.domain_evidence["index_executed"] is True
    assert receipt.domain_evidence["index_state"] == "completed"
    assert receipt.domain_evidence["index_task_id"] == "task-index-001"
    assert receipt.domain_evidence["index_run_id"] == "run-index-001"
    assert receipt.domain_evidence["search_executed"] is True
    assert receipt.domain_evidence["search_state"] == "completed"
    assert receipt.domain_evidence["search_task_id"] == "task-search-001"
    assert receipt.domain_evidence["search_run_id"] == "run-search-001"
    assert receipt.domain_evidence["task_ids_distinct"] is True
    assert receipt.domain_evidence["run_ids_distinct"] is True
    assert receipt.domain_evidence["graceful_stop"] is True
    assert receipt.domain_evidence["cleanup_verified"] is True

    assert receipt.guardrails["direct_run_endpoint"] is False
    assert receipt.guardrails["direct_task_construction"] is False
    assert receipt.guardrails["direct_task_executor_call"] is False
    assert receipt.guardrails["direct_nexus_call"] is False
    assert receipt.guardrails["direct_agent_call"] is False
    assert receipt.guardrails["mock_http_server"] is False
    assert receipt.guardrails["http_test_client"] is False
    assert receipt.guardrails["fake_interaction_service"] is False
    assert receipt.guardrails["fake_hosted_application"] is False
    assert receipt.guardrails["new_platform_interaction_adapter"] is False
    assert receipt.guardrails["generic_os_hosting_adapter"] is False
    assert receipt.guardrails["service_installation"] is False
    assert receipt.guardrails["powershell_invocation_via_shell"] is False
    assert receipt.guardrails["manual_evidence_injection"] is False
    assert receipt.guardrails["inmemory_receipt_store"] is False
    assert receipt.guardrails["direct_mongodb_write"] is False
    assert receipt.guardrails["direct_pymongo_from_lkw"] is False
    assert receipt.guardrails["markdown_source_of_truth"] is False

    assert receipt.metadata["proof_runner"] == "run-lkw-windows-interaction-proof.py"
    assert receipt.metadata["receipt_task"] == "LKW.6C"
    assert (
        receipt.metadata["evidence_schema"]
        == "lkw.windows_interaction_proof_evidence.v1"
    )
    assert receipt.metadata["recorded_from_live_run"] is True
    assert receipt.metadata["mongo_express_url"] == "http://127.0.0.1:8086"
    assert receipt.metadata["source_test"] == _LIVE_NODE
    assert receipt.metadata["adapter_script"] == "invoke-lkw-interaction.ps1"


def test_build_windows_interaction_proof_receipt_has_no_credentials() -> None:
    proof = _load_proof_module()
    evidence = _sample_evidence(proof)
    receipt = proof.build_windows_interaction_proof_receipt(
        run_id="run-windows-1",
        correlation_id="corr-windows-1",
        evidence=evidence,
        mongo_express_url="http://127.0.0.1:8086",
    )
    serialized = receipt.model_dump_json()
    assert "mongodb://" not in serialized
    assert "authSource" not in serialized
    assert "intergrax-local-dev-only" not in serialized
    assert "password" not in serialized.lower()
    assert "LKW_MONGODB_ROOT_PASSWORD" not in serialized
    assert "INTERGRAX_MONGODB_URI" not in serialized
    assert "fixture absolute path" not in serialized
    assert "raw message" not in serialized
    assert "raw metadata" not in serialized


def test_parse_windows_interaction_proof_junit_accepts_complete_evidence(
    tmp_path: Path,
) -> None:
    proof = _load_proof_module()
    junit = tmp_path / "ok.xml"
    _write_junit(
        junit,
        name="test_windows_powershell_adapter_executes_real_lkw_interactions",
        props=_complete_props(),
    )
    evidence = proof.parse_windows_interaction_proof_junit(junit)
    assert evidence.hosted_ready is True
    assert evidence.adapter_id == "lkw.windows_powershell"
    assert evidence.index_task_id == "task-index-001"
    assert evidence.search_task_id == "task-search-001"
    assert evidence.index_run_id == "run-index-001"
    assert evidence.search_run_id == "run-search-001"
    assert evidence.task_ids_distinct is True
    assert evidence.run_ids_distinct is True


@pytest.mark.parametrize(
    ("mutate", "expected_fragment"),
    [
        ("missing_testcase", "unexpected_testcase_count"),
        ("unexpected_testcase", "unexpected_testcase"),
        ("failed", "failed_testcase"),
        ("error", "errored_testcase"),
        ("skipped", "skipped_testcase"),
        ("missing_property", "missing_property"),
        ("false_evidence", "false_required_evidence"),
        ("blank_task", "blank_property"),
        ("blank_run", "blank_property"),
        ("same_task", "same_task_ids"),
        ("same_run", "same_run_ids"),
        ("wrong_adapter", "invalid_adapter_id"),
        ("wrong_endpoint", "invalid_endpoint"),
        ("wrong_surface", "invalid_interaction_surface"),
        ("wrong_channel", "invalid_interaction_channel"),
        ("wrong_index_state", "invalid_index_state"),
        ("wrong_search_state", "invalid_search_state"),
        ("malformed", "malformed_junit_xml"),
    ],
)
def test_parse_windows_interaction_proof_junit_rejects_invalid_evidence(
    tmp_path: Path,
    mutate: str,
    expected_fragment: str,
) -> None:
    proof = _load_proof_module()
    junit = tmp_path / f"{mutate}.xml"

    if mutate == "malformed":
        junit.write_text("<not-xml", encoding="utf-8")
        with pytest.raises(proof.WindowsInteractionProofEvidenceError) as exc_info:
            proof.parse_windows_interaction_proof_junit(junit)
        assert expected_fragment in str(exc_info.value)
        return

    if mutate == "missing_testcase":
        junit.write_text(
            '<?xml version="1.0"?><testsuite tests="0"></testsuite>\n',
            encoding="utf-8",
        )
        with pytest.raises(proof.WindowsInteractionProofEvidenceError) as exc_info:
            proof.parse_windows_interaction_proof_junit(junit)
        assert expected_fragment in str(exc_info.value)
        return

    props = _complete_props()
    name = "test_windows_powershell_adapter_executes_real_lkw_interactions"
    status: str | None = None

    if mutate == "unexpected_testcase":
        name = "test_other"
    elif mutate == "failed":
        status = "failure"
    elif mutate == "error":
        status = "error"
    elif mutate == "skipped":
        status = "skipped"
    elif mutate == "missing_property":
        del props["windows_interaction.hosted_ready"]
    elif mutate == "false_evidence":
        props["windows_interaction.hosted_ready"] = "false"
    elif mutate == "blank_task":
        props["windows_interaction.index_task_id"] = "   "
    elif mutate == "blank_run":
        props["windows_interaction.search_run_id"] = "   "
    elif mutate == "same_task":
        props["windows_interaction.search_task_id"] = props[
            "windows_interaction.index_task_id"
        ]
    elif mutate == "same_run":
        props["windows_interaction.search_run_id"] = props[
            "windows_interaction.index_run_id"
        ]
    elif mutate == "wrong_adapter":
        props["windows_interaction.adapter_id"] = "other.adapter"
    elif mutate == "wrong_endpoint":
        props["windows_interaction.intake_endpoint"] = "/v1/local_workspace/run"
    elif mutate == "wrong_surface":
        props["windows_interaction.interaction_surface"] = "slack"
    elif mutate == "wrong_channel":
        props["windows_interaction.interaction_channel"] = "windows"
    elif mutate == "wrong_index_state":
        props["windows_interaction.index_state"] = "failed"
    elif mutate == "wrong_search_state":
        props["windows_interaction.search_state"] = "failed"

    _write_junit(junit, name=name, props=props, status=status)
    with pytest.raises(proof.WindowsInteractionProofEvidenceError) as exc_info:
        proof.parse_windows_interaction_proof_junit(junit)
    assert expected_fragment in str(exc_info.value)


def test_windows_interaction_proof_script_uses_platform_receipt_recording() -> None:
    text = _read(_PROOF_SCRIPT)
    assert _LIVE_NODE in text
    assert "--junitxml" in text
    assert "junit_family=legacy" in text
    assert "ProofReceipt" in text
    assert "record_and_verify_proof_receipt" in text
    assert "create_mongodb_integration" in text
    assert "MongoDBDocumentStoreIntegration" in text
    assert "windows_required" in text
    assert "windows_powershell_unavailable" in text
    assert "windows_interaction_live_test_failed" in text
    assert "windows_interaction_evidence_invalid" in text
    assert "proof_receipt_recording_failed" in text
    assert "DocumentRecord" not in text
    assert "MongoClient" not in text
    assert "insert_one" not in text
    assert "update_one" not in text
    assert "replace_one" not in text
    assert "find_one" not in text
    pymongo_import = "import py" + "mongo"
    pymongo_from = "from py" + "mongo"
    assert pymongo_import not in text
    assert pymongo_from not in text


def test_windows_interaction_proof_pass_prints_after_receipt_verification() -> None:
    text = _read(_PROOF_SCRIPT)
    receipt_record_index = text.index("record_and_verify_proof_receipt")
    pass_output_index = text.index('print("proof_result=PASS")')
    assert receipt_record_index < pass_output_index


def test_windows_interaction_proof_bat_runner_starts_only_mongodb_services() -> None:
    text = _read(_PROOF_BAT)
    assert "docker-compose.yml" in text
    assert "docker-compose.mongodb.yml" in text
    assert "lkw-mongodb" in text
    assert "lkw-mongo-express" in text
    assert "mongodb_container_healthy=true" in text
    assert "mongo_express_available=true" in text
    assert "powershell.exe" in text
    assert "--extra integrations-mongodb" in text

    up_lines = [
        line.strip()
        for line in text.splitlines()
        if "docker compose" in line and " up " in line
    ]
    assert len(up_lines) == 1
    up_line = up_lines[0]
    assert "lkw-mongodb" in up_line
    assert "lkw-mongo-express" in up_line
    for forbidden in (
        "local_workspace",
        "lkw-background-worker",
        "lkw-kafka",
        "lkw-redis",
        "qdrant",
        "elasticsearch",
        "sentry",
    ):
        assert forbidden not in up_line


def test_live_test_emits_required_evidence_keys() -> None:
    text = _read(_LIVE_TEST)
    for key in _REQUIRED_EVIDENCE_KEYS:
        assert key in text
    assert "record_property" in text
    assert "test_windows_powershell_adapter_executes_real_lkw_interactions" in text
    assert "powershell.exe" in text
    assert "local_workspace_application.hosting" in text


def test_public_reviewer_document_contains_windows_interaction_steps() -> None:
    text = _read(_PUBLIC_PLATFORM_PROOF)
    assert "## Step 12 — Run the Windows PowerShell interaction proof" in text
    assert (
        "## Step 13 — Inspect the Windows Interaction ProofReceipt in Mongo Express"
    ) in text
    assert "run-lkw-windows-interaction-proof.bat" in text
    assert "proof_kind=platform_windows_interaction" in text
    shortcut = text.split("## Reviewer shortcut", 1)[1]
    assert "run-lkw-windows-interaction-proof.bat" in shortcut
