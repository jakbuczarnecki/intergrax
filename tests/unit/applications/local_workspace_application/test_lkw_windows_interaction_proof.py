# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LKW_ROOT = _PROJECT_ROOT / "applications" / "local_workspace_application"
_SCRIPTS_DIR = _LKW_ROOT / "scripts"
_PROOF_SCRIPT = _SCRIPTS_DIR / "run-lkw-windows-interaction-proof.py"
_SHARED_PROOF = _SCRIPTS_DIR / "run-lkw-os-interaction-proof.py"
_PROOF_BAT = _SCRIPTS_DIR / "run-lkw-windows-interaction-proof.bat"
_ADAPTER_SCRIPT = _SCRIPTS_DIR / "invoke-lkw-interaction.ps1"
_SHARED_CLIENT = _SCRIPTS_DIR / "invoke-lkw-interaction.py"
_PUBLIC_PLATFORM_PROOF = (
    _PROJECT_ROOT / "docs" / "public-adoption" / "LKW_PLATFORM_PROOF.md"
)
_LIVE_TEST = (
    _LKW_ROOT / "tests" / "interactions" / "test_windows_powershell_interaction_live.py"
)

_LIVE_NODE = (
    "applications/local_workspace_application/tests/interactions/"
    "test_windows_powershell_interaction_live.py::"
    "test_windows_powershell_adapter_executes_real_lkw_interactions"
)

_REQUIRED_EVIDENCE_KEYS = (
    "os_interaction.hosted_ready",
    "os_interaction.adapter_invoked",
    "os_interaction.adapter_id",
    "os_interaction.client_runtime",
    "os_interaction.wrapper_runtime",
    "os_interaction.transport",
    "os_interaction.intake_endpoint",
    "os_interaction.interaction_surface",
    "os_interaction.interaction_channel",
    "os_interaction.index_executed",
    "os_interaction.index_state",
    "os_interaction.index_task_id",
    "os_interaction.index_run_id",
    "os_interaction.search_executed",
    "os_interaction.search_state",
    "os_interaction.search_task_id",
    "os_interaction.search_run_id",
    "os_interaction.task_ids_distinct",
    "os_interaction.run_ids_distinct",
    "os_interaction.graceful_stop",
    "os_interaction.cleanup_verified",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_shared():
    module_name = "lkw_windows_interaction_proof_via_shared"
    spec = importlib.util.spec_from_file_location(module_name, _SHARED_PROOF)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_powershell_adapter_is_thin_wrapper() -> None:
    text = _read(_ADAPTER_SCRIPT)
    assert "invoke-lkw-interaction.py" in text
    assert "--os-family" in text
    assert "windows" in text
    assert "lkw.windows_powershell" in text
    assert "windows_powershell" in text
    assert "IsNullOrWhiteSpace($SessionId)" in text
    assert "IsNullOrWhiteSpace($InteractionId)" in text
    for forbidden in (
        "Invoke-RestMethod",
        "Invoke-WebRequest",
        "ConvertTo-Json",
        "ConvertFrom-Json",
        "Invoke-Expression",
        "Start-Process",
        "/v1/local_workspace/run",
        "MongoClient",
        "pymongo",
    ):
        assert forbidden not in text
    assert _SHARED_CLIENT.is_file()


def _capture_ps1_argv(tmp_path: Path, *ps_args: str) -> list[str]:
    """Capture argv built by the public PS1 before native process launch."""
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("powershell.exe is required")

    dump_path = tmp_path / "argv.json"
    env = os.environ.copy()
    env["LKW_PS1_ARGV_DUMP"] = str(dump_path)

    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_ADAPTER_SCRIPT),
            *ps_args,
        ],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    assert dump_path.is_file(), (
        f"ps1 did not dump argv\n"
        f"exit={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    payload = json.loads(dump_path.read_text(encoding="utf-8-sig"))
    assert isinstance(payload, list)
    return [str(item) for item in payload]


def test_powershell_omits_empty_session_and_interaction_ids(tmp_path: Path) -> None:
    argv = _capture_ps1_argv(
        tmp_path,
        "-Message",
        "hello",
        "-BaseUrl",
        "http://127.0.0.1:65530",
        "-Capability",
        "local.workspace.index",
        "-SessionId",
        "",
        "-InteractionId",
        "",
        "-TimeoutSeconds",
        "1",
    )
    assert any(part.endswith("invoke-lkw-interaction.py") for part in argv)
    assert "--os-family" in argv
    assert "windows" in argv
    assert "--session-id" not in argv
    assert "--interaction-id" not in argv
    assert "--message" in argv
    assert "hello" in argv


def test_powershell_forwards_nonempty_session_and_interaction_ids(
    tmp_path: Path,
) -> None:
    argv = _capture_ps1_argv(
        tmp_path,
        "-Message",
        "hello",
        "-BaseUrl",
        "http://127.0.0.1:65530",
        "-SessionId",
        "sess-1",
        "-InteractionId",
        "ix-1",
        "-TimeoutSeconds",
        "1",
    )
    assert "--session-id" in argv
    assert argv[argv.index("--session-id") + 1] == "sess-1"
    assert "--interaction-id" in argv
    assert argv[argv.index("--interaction-id") + 1] == "ix-1"


def test_powershell_wrapper_does_not_perform_http() -> None:
    text = _read(_ADAPTER_SCRIPT)
    for forbidden in (
        "Invoke-RestMethod",
        "Invoke-WebRequest",
        "System.Net.HttpWebRequest",
        "HttpClient",
        "/v1/interactions/intake",
    ):
        assert forbidden not in text
    assert "invoke-lkw-interaction.py" in text
    assert "ProcessStartInfo" in text
    assert "& $python @argumentList" not in text


def test_powershell_preserves_metadata_json_quotes() -> None:
    """Windows PowerShell must not strip JSON quotes before the Python client."""
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("powershell.exe is required")
    meta = json.dumps(
        {"source_paths": ["C:/tmp/x.txt"], "collection_id": "c1"},
        ensure_ascii=False,
    )
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_ADAPTER_SCRIPT),
            "-Message",
            "hello",
            "-BaseUrl",
            "http://127.0.0.1:65530",
            "-Capability",
            "local.workspace.index",
            "-MetadataJson",
            meta,
        ],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    combined = (completed.stdout or "") + (completed.stderr or "")
    assert "invalid_adapter_input" not in combined
    assert "interaction_request_failed" in combined
    assert completed.returncode == 3


def test_windows_proof_identity_and_receipt_compatibility() -> None:
    proof = _load_shared()
    contract = proof.resolve_os_proof_contract("windows")
    evidence = proof.OSInteractionProofEvidence(
        schema_version="lkw.os_interaction_proof_evidence.v1",
        os_family="windows",
        os_version="10.0",
        architecture="AMD64",
        client_runtime="python",
        wrapper_runtime="windows_powershell",
        adapter_id="lkw.windows_powershell",
        source="windows_powershell",
        transport="http",
        intake_endpoint="/v1/interactions/intake",
        interaction_surface="lab_json",
        interaction_channel="lab",
        hosted_ready=True,
        adapter_invoked=True,
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
    receipt = proof.build_os_interaction_proof_receipt(
        contract=contract,
        run_id="run-windows-1",
        correlation_id="corr-windows-1",
        evidence=evidence,
        mongo_express_url="http://127.0.0.1:8086",
    )
    assert (
        receipt.proof_id == "local_workspace:platform_windows_interaction:run-windows-1"
    )
    assert receipt.proof_kind == "platform_windows_interaction"
    assert receipt.provider_evidence["os_adapter"] == "lkw.windows_powershell"
    assert receipt.provider_evidence["intake_endpoint"] == "/v1/interactions/intake"
    assert receipt.provider_evidence["interaction_surface"] == "lab_json"
    assert receipt.provider_evidence["interaction_channel"] == "lab"
    assert receipt.domain_evidence["powershell_runtime"] == "Windows PowerShell"
    assert receipt.metadata["adapter_script"] == "invoke-lkw-interaction.ps1"
    assert receipt.metadata["source_test"] == _LIVE_NODE


def test_windows_interaction_proof_bat_delegates_to_shared_runner() -> None:
    text = _read(_PROOF_BAT)
    assert "run-lkw-os-interaction-proof.py" in text
    assert "--os-family windows" in text
    assert "--extra integrations-mongodb" in text
    assert "Invoke-WebRequest" not in text
    assert "ConvertFrom-Json" not in text


def test_windows_compatibility_shim_is_thin() -> None:
    text = _read(_PROOF_SCRIPT)
    assert "run-lkw-os-interaction-proof.py" in text
    assert "--os-family" in text
    assert "windows" in text
    assert "ProofReceipt(" not in text
    assert "record_and_verify_proof_receipt" not in text


def test_live_test_emits_required_evidence_keys() -> None:
    helpers = _LKW_ROOT / "tests" / "interactions" / "os_interaction_live_helpers.py"
    text = _read(helpers) + "\n" + _read(_LIVE_TEST)
    for key in _REQUIRED_EVIDENCE_KEYS:
        assert key in text
    assert "test_windows_powershell_adapter_executes_real_lkw_interactions" in text
    assert "local_workspace_application.hosting" in text
    assert "invoke-lkw-interaction.ps1" in text


def test_public_reviewer_document_contains_windows_interaction_steps() -> None:
    text = _read(_PUBLIC_PLATFORM_PROOF)
    assert "# Optional operating-system interaction proofs" in text
    assert (
        "## Windows users — Optional W1: Run the Windows PowerShell interaction proof"
        in text
    )
    assert (
        "## Windows users — Optional W2: Inspect the Windows Interaction ProofReceipt"
        in text
    )
    assert "run-lkw-windows-interaction-proof.bat" in text
    assert "proof_kind=platform_windows_interaction" in text
    assert "os_family=windows" in text
    assert "adapter_id=lkw.windows_powershell" in text
    assert "intake_endpoint=/v1/interactions/intake" in text
