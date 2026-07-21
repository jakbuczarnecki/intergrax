# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS = _PROJECT_ROOT / "applications" / "local_workspace_application" / "scripts"
_SHARED = _SCRIPTS / "run-lkw-os-interaction-proof.py"
_SHIM = _SCRIPTS / "run-lkw-windows-interaction-proof.py"
_BAT = _SCRIPTS / "run-lkw-windows-interaction-proof.bat"
_LINUX_SH = _SCRIPTS / "run-lkw-linux-interaction-proof.sh"
_MACOS_SH = _SCRIPTS / "run-lkw-macos-interaction-proof.sh"

_WINDOWS_NODE = (
    "applications/local_workspace_application/tests/interactions/"
    "test_windows_powershell_interaction_live.py::"
    "test_windows_powershell_adapter_executes_real_lkw_interactions"
)


def _load_shared():
    module_name = "lkw_os_interaction_proof_shared"
    spec = importlib.util.spec_from_file_location(module_name, _SHARED)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _complete_props(os_family: str) -> dict[str, str]:
    contract = {
        "windows": {
            "adapter_id": "lkw.windows_powershell",
            "source": "windows_powershell",
            "wrapper_runtime": "windows_powershell",
        },
        "linux": {
            "adapter_id": "lkw.linux_shell",
            "source": "linux_shell",
            "wrapper_runtime": "posix_sh",
        },
        "macos": {
            "adapter_id": "lkw.macos_shell",
            "source": "macos_shell",
            "wrapper_runtime": "posix_sh",
        },
    }[os_family]
    return {
        "os_interaction.hosted_ready": "true",
        "os_interaction.adapter_invoked": "true",
        "os_interaction.os_family": os_family,
        "os_interaction.os_version": "test-os",
        "os_interaction.architecture": "x86_64",
        "os_interaction.client_runtime": "python",
        "os_interaction.wrapper_runtime": contract["wrapper_runtime"],
        "os_interaction.adapter_id": contract["adapter_id"],
        "os_interaction.source": contract["source"],
        "os_interaction.transport": "http",
        "os_interaction.intake_endpoint": "/v1/interactions/intake",
        "os_interaction.interaction_surface": "lab_json",
        "os_interaction.interaction_channel": "lab",
        "os_interaction.index_executed": "true",
        "os_interaction.index_state": "completed",
        "os_interaction.index_task_id": "task-index-001",
        "os_interaction.index_run_id": "run-index-001",
        "os_interaction.search_executed": "true",
        "os_interaction.search_state": "completed",
        "os_interaction.search_task_id": "task-search-001",
        "os_interaction.search_run_id": "run-search-001",
        "os_interaction.task_ids_distinct": "true",
        "os_interaction.run_ids_distinct": "true",
        "os_interaction.graceful_stop": "true",
        "os_interaction.cleanup_verified": "true",
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


def _sample_evidence(module, os_family: str = "windows", **overrides):
    contract = module.resolve_os_proof_contract(os_family)
    base = dict(
        schema_version="lkw.os_interaction_proof_evidence.v1",
        os_family=os_family,
        os_version="test-os",
        architecture="x86_64",
        client_runtime="python",
        wrapper_runtime=contract.wrapper_runtime,
        adapter_id=contract.adapter_id,
        source=contract.source,
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
    base.update(overrides)
    return module.OSInteractionProofEvidence(**base)


def test_proof_kinds_and_identities_preserved() -> None:
    proof = _load_shared()
    windows = proof.resolve_os_proof_contract("windows")
    linux = proof.resolve_os_proof_contract("linux")
    macos = proof.resolve_os_proof_contract("macos")
    assert windows.proof_kind == "platform_windows_interaction"
    assert linux.proof_kind == "platform_linux_interaction"
    assert macos.proof_kind == "platform_macos_interaction"
    assert (
        proof.build_os_interaction_proof_id(
            proof_kind=windows.proof_kind, run_id="run-1"
        )
        == "local_workspace:platform_windows_interaction:run-1"
    )
    assert (
        proof.build_os_interaction_proof_id(proof_kind=linux.proof_kind, run_id="run-1")
        == "local_workspace:platform_linux_interaction:run-1"
    )
    assert (
        proof.build_os_interaction_proof_id(proof_kind=macos.proof_kind, run_id="run-1")
        == "local_workspace:platform_macos_interaction:run-1"
    )


@pytest.mark.parametrize("os_family", ["windows", "linux", "macos"])
def test_typed_evidence_accepted(tmp_path: Path, os_family: str) -> None:
    proof = _load_shared()
    contract = proof.resolve_os_proof_contract(os_family)
    junit = tmp_path / f"{os_family}.xml"
    _write_junit(
        junit,
        name=contract.expected_testcase_name,
        props=_complete_props(os_family),
    )
    evidence = proof.parse_os_interaction_proof_junit(junit, contract=contract)
    assert evidence.os_family == os_family
    assert evidence.adapter_id == contract.adapter_id
    assert evidence.client_runtime == "python"
    assert evidence.wrapper_runtime == contract.wrapper_runtime


@pytest.mark.parametrize(
    ("mutate", "expected_fragment"),
    [
        ("missing_property", "missing_property"),
        ("false_evidence", "false_required_evidence"),
        ("wrong_adapter", "invalid_adapter_id"),
        ("wrong_source", "invalid_source"),
        ("wrong_wrapper", "invalid_wrapper_runtime"),
        ("wrong_os", "invalid_os_family"),
        ("wrong_test", "unexpected_testcase"),
        ("failed", "failed_testcase"),
        ("error", "errored_testcase"),
        ("skipped", "skipped_testcase"),
        ("same_task", "same_task_ids"),
        ("same_run", "same_run_ids"),
        ("multiple", "unexpected_testcase_count"),
    ],
)
def test_evidence_rejected(
    tmp_path: Path,
    mutate: str,
    expected_fragment: str,
) -> None:
    proof = _load_shared()
    contract = proof.resolve_os_proof_contract("windows")
    junit = tmp_path / f"{mutate}.xml"
    if mutate == "multiple":
        junit.write_text(
            '<?xml version="1.0"?><testsuite>'
            '<testcase name="a"/><testcase name="b"/>'
            "</testsuite>\n",
            encoding="utf-8",
        )
        with pytest.raises(proof.OSInteractionProofEvidenceError) as exc_info:
            proof.parse_os_interaction_proof_junit(junit, contract=contract)
        assert expected_fragment in str(exc_info.value)
        return

    props = _complete_props("windows")
    name = contract.expected_testcase_name
    status: str | None = None
    if mutate == "missing_property":
        del props["os_interaction.hosted_ready"]
    elif mutate == "false_evidence":
        props["os_interaction.hosted_ready"] = "false"
    elif mutate == "wrong_adapter":
        props["os_interaction.adapter_id"] = "other"
    elif mutate == "wrong_source":
        props["os_interaction.source"] = "linux_shell"
    elif mutate == "wrong_wrapper":
        props["os_interaction.wrapper_runtime"] = "posix_sh"
    elif mutate == "wrong_os":
        props["os_interaction.os_family"] = "linux"
    elif mutate == "wrong_test":
        name = "test_other"
    elif mutate == "failed":
        status = "failure"
    elif mutate == "error":
        status = "error"
    elif mutate == "skipped":
        status = "skipped"
    elif mutate == "same_task":
        props["os_interaction.search_task_id"] = props["os_interaction.index_task_id"]
    elif mutate == "same_run":
        props["os_interaction.search_run_id"] = props["os_interaction.index_run_id"]

    _write_junit(junit, name=name, props=props, status=status)
    with pytest.raises(proof.OSInteractionProofEvidenceError) as exc_info:
        proof.parse_os_interaction_proof_junit(junit, contract=contract)
    assert expected_fragment in str(exc_info.value)


def test_receipt_contains_required_fields_and_no_credentials() -> None:
    proof = _load_shared()
    contract = proof.resolve_os_proof_contract("windows")
    evidence = _sample_evidence(proof, "windows")
    receipt = proof.build_os_interaction_proof_receipt(
        contract=contract,
        run_id="run-windows-1",
        correlation_id="corr-1",
        evidence=evidence,
        mongo_express_url="http://127.0.0.1:8086",
    )
    assert receipt.proof_kind == "platform_windows_interaction"
    assert (
        receipt.proof_id == "local_workspace:platform_windows_interaction:run-windows-1"
    )
    assert receipt.provider_evidence["client_runtime"] == "python"
    assert receipt.provider_evidence["wrapper_runtime"] == "windows_powershell"
    assert receipt.provider_evidence["os_version"] == "test-os"
    assert receipt.provider_evidence["architecture"] == "x86_64"
    assert receipt.domain_evidence["powershell_runtime"] == "Windows PowerShell"
    assert receipt.metadata["proof_runner"] == "run-lkw-os-interaction-proof.py"
    assert receipt.metadata["receipt_task"] == "PROOF-PORTABILITY-1C"
    assert receipt.metadata["source_test"] == _WINDOWS_NODE
    serialized = receipt.model_dump_json()
    assert "mongodb://" not in serialized
    assert "password" not in serialized.lower()
    assert "INTERGRAX_MONGODB_URI" not in serialized


def test_runtime_os_mismatch_rejected() -> None:
    proof = _load_shared()
    with pytest.raises(proof.OSInteractionProofEvidenceError) as exc_info:
        proof.validate_runtime_os_matches("linux", runtime_os_family="windows")
    assert "runtime_os_mismatch" in str(exc_info.value)


def test_receipt_recording_failure_produces_overall_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    proof = _load_shared()
    monkeypatch.setattr(
        proof,
        "validate_runtime_os_matches",
        lambda *_a, **_k: "windows",
    )
    monkeypatch.setattr(proof.shutil, "which", lambda name: "x" if name else None)
    monkeypatch.setattr(proof, "prepare_mongodb_stack", lambda **_k: None)
    monkeypatch.setattr(proof, "_run_accepted_live_test", lambda **_k: 0)

    def _parse(*_a, **_k):
        return _sample_evidence(proof, "windows")

    monkeypatch.setattr(proof, "parse_os_interaction_proof_junit", _parse)
    monkeypatch.setattr(
        proof,
        "record_os_interaction_proof_receipt",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("store_down")),
    )
    code = proof.main(
        [
            "--os-family",
            "windows",
            "--run-id",
            "run-x",
            "--correlation-id",
            "corr-x",
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "proof_result=FAIL" in out
    assert "proof_receipt_recording_failed" in out


def test_compatibility_shim_delegates_without_duplicate_implementation() -> None:
    text = _SHIM.read_text(encoding="utf-8")
    assert "run-lkw-os-interaction-proof.py" in text
    assert "--os-family" in text
    assert "windows" in text
    for forbidden in (
        "ProofReceipt(",
        "record_and_verify_proof_receipt",
        "parse_os_interaction_proof_junit",
        "create_mongodb_integration",
        "ET.fromstring",
        "--junitxml",
    ):
        assert forbidden not in text


def test_launchers_delegate_to_shared_runner() -> None:
    bat = _BAT.read_text(encoding="utf-8")
    assert "run-lkw-os-interaction-proof.py" in bat
    assert "--os-family windows" in bat
    assert "Invoke-WebRequest" not in bat
    assert "ConvertFrom-Json" not in bat

    for path, family in ((_LINUX_SH, "linux"), (_MACOS_SH, "macos")):
        text = path.read_text(encoding="utf-8")
        assert "run-lkw-os-interaction-proof.py" in text
        assert f"--os-family {family}" in text
        for forbidden in ("curl", "wget", "jq", "ProofReceipt", "docker compose"):
            assert forbidden not in text


def test_shared_runner_uses_argument_list_subprocess() -> None:
    text = _SHARED.read_text(encoding="utf-8")
    assert "shell=False" in text
    assert "subprocess.run(" in text
    assert "shell=True" not in text
    assert "record_and_verify_proof_receipt" in text
    assert "create_mongodb_integration" in text
