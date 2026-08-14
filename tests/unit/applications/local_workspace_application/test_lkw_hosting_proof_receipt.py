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
_PROOF_SCRIPT = _SCRIPTS_DIR / "run-lkw-hosting-proof.py"
_PROOF_BAT = _SCRIPTS_DIR / "run-lkw-hosting-proof.bat"
_PUBLIC_PLATFORM_PROOF = (
    _PROJECT_ROOT
    / "applications"
    / "local_workspace_application"
    / "docs"
    / "proof"
    / "LKW_PLATFORM_PROOF.md"
)
_FOREGROUND_TEST = _LKW_ROOT / "tests" / "hosting" / "test_hosted_foreground_process.py"
_RESTART_TEST = _LKW_ROOT / "tests" / "hosting" / "test_hosted_restart_live.py"

_LIVE_NODES = (
    "applications/local_workspace_application/tests/hosting/test_hosted_foreground_process.py::test_hosted_foreground_process_ready_index_and_instance_conflict",
    "applications/local_workspace_application/tests/hosting/test_hosted_foreground_process.py::test_hosted_foreground_process_graceful_stop_releases_instance_lock",
    "applications/local_workspace_application/tests/hosting/test_hosted_restart_live.py::test_hosted_lkw_restart_creates_new_instance_and_accepts_work",
)

_REQUIRED_EVIDENCE_KEYS = (
    "hosting.foreground_ready",
    "hosting.real_index_before_restart",
    "hosting.instance_conflict_verified",
    "hosting.first_process_remained_ready",
    "hosting.foreground_clean_stop",
    "hosting.foreground_shutdown_reason",
    "hosting.replacement_process_ready",
    "hosting.instance_lock_released",
    "hosting.replacement_clean_stop",
    "hosting.restart_requested",
    "hosting.first_instance_id",
    "hosting.second_instance_id",
    "hosting.instance_id_changed",
    "hosting.first_attempt_exit_kind",
    "hosting.first_attempt_cleanup_verified",
    "hosting.first_lease_released",
    "hosting.first_context_closed",
    "hosting.stopped_events_verified",
    "hosting.restart_events_verified",
    "hosting.second_instance_ready",
    "hosting.real_index_after_restart",
    "hosting.profile_digest",
    "hosting.definition_digest",
    "hosting.profile_digest_preserved",
    "hosting.definition_digest_preserved",
    "hosting.final_exit_kind",
    "hosting.final_cleanup_verified",
    "hosting.final_lease_released",
    "hosting.final_context_closed",
    "hosting.final_lock_reacquired",
)

_DIGEST = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_proof_module():
    module_name = "lkw_hosting_proof"
    spec = importlib.util.spec_from_file_location(module_name, _PROOF_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sample_evidence(module, **overrides):
    base = dict(
        schema_version="lkw.application_hosting_proof_evidence.v1",
        foreground_ready=True,
        real_index_before_restart=True,
        instance_conflict_verified=True,
        first_process_remained_ready=True,
        foreground_clean_stop=True,
        foreground_shutdown_reason="signal.sigterm",
        replacement_process_ready=True,
        instance_lock_released=True,
        replacement_clean_stop=True,
        restart_requested=True,
        first_instance_id="00000000-0000-4000-8000-000000000001",
        second_instance_id="00000000-0000-4000-8000-000000000002",
        instance_id_changed=True,
        first_attempt_exit_kind="restart_requested",
        first_attempt_cleanup_verified=True,
        first_lease_released=True,
        first_context_closed=True,
        stopped_events_verified=True,
        restart_events_verified=True,
        second_instance_ready=True,
        real_index_after_restart=True,
        profile_digest=_DIGEST,
        definition_digest=_DIGEST_B,
        profile_digest_preserved=True,
        definition_digest_preserved=True,
        final_exit_kind="clean_stop",
        final_cleanup_verified=True,
        final_lease_released=True,
        final_context_closed=True,
        final_lock_reacquired=True,
    )
    base.update(overrides)
    return module.HostingProofEvidence(**base)


def _write_junit(
    path: Path, *, cases: list[tuple[str, dict[str, str], str | None]]
) -> None:
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<testsuite name="pytest" tests="3" errors="0" failures="0" skipped="0">',
    ]
    for name, props, status in cases:
        parts.append(f'  <testcase classname="hosting" name="{name}" time="0.1">')
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


def _complete_props() -> dict[str, dict[str, str]]:
    return {
        "test_hosted_foreground_process_ready_index_and_instance_conflict": {
            "hosting.foreground_ready": "true",
            "hosting.real_index_before_restart": "true",
            "hosting.instance_conflict_verified": "true",
            "hosting.first_process_remained_ready": "true",
        },
        "test_hosted_foreground_process_graceful_stop_releases_instance_lock": {
            "hosting.foreground_clean_stop": "true",
            "hosting.foreground_shutdown_reason": "signal.sigbreak",
            "hosting.replacement_process_ready": "true",
            "hosting.instance_lock_released": "true",
            "hosting.replacement_clean_stop": "true",
        },
        "test_hosted_lkw_restart_creates_new_instance_and_accepts_work": {
            "hosting.restart_requested": "true",
            "hosting.first_instance_id": "00000000-0000-4000-8000-000000000001",
            "hosting.second_instance_id": "00000000-0000-4000-8000-000000000002",
            "hosting.instance_id_changed": "true",
            "hosting.first_attempt_exit_kind": "restart_requested",
            "hosting.first_attempt_cleanup_verified": "true",
            "hosting.first_lease_released": "true",
            "hosting.first_context_closed": "true",
            "hosting.stopped_events_verified": "true",
            "hosting.restart_events_verified": "true",
            "hosting.second_instance_ready": "true",
            "hosting.real_index_after_restart": "true",
            "hosting.profile_digest": _DIGEST,
            "hosting.definition_digest": _DIGEST_B,
            "hosting.profile_digest_preserved": "true",
            "hosting.definition_digest_preserved": "true",
            "hosting.final_exit_kind": "clean_stop",
            "hosting.final_cleanup_verified": "true",
            "hosting.final_lease_released": "true",
            "hosting.final_context_closed": "true",
            "hosting.final_lock_reacquired": "true",
        },
    }


def test_build_application_hosting_proof_receipt_maps_live_evidence() -> None:
    proof = _load_proof_module()
    evidence = _sample_evidence(proof)
    receipt = proof.build_application_hosting_proof_receipt(
        run_id="run-hosting-1",
        correlation_id="corr-hosting-1",
        evidence=evidence,
        mongo_express_url="http://127.0.0.1:8086",
    )

    assert (
        receipt.proof_id == "local_workspace:platform_application_hosting:run-hosting-1"
    )
    assert receipt.proof_kind == "platform_application_hosting"
    assert receipt.application_id == "local_workspace"
    assert receipt.result.value == "PASS"
    assert receipt.task_id is None
    assert receipt.run_id == "run-hosting-1"
    assert receipt.correlation_id == "corr-hosting-1"

    assert (
        receipt.provider_evidence["hosting_platform"] == "intergrax_application_hosting"
    )
    assert (
        receipt.provider_evidence["foreground_entrypoint"]
        == "python_-m_local_workspace_application.hosting"
    )
    assert receipt.provider_evidence["foreground_execution"] == "real_subprocess"
    assert receipt.provider_evidence["runtime_surface"] == "fastapi_uvicorn"
    assert receipt.provider_evidence["supervisor"] == "HostedApplicationSupervisor"
    assert receipt.provider_evidence["engine"] == "HostedApplicationEngine"
    assert (
        receipt.provider_evidence["instance_guard"]
        == "FileHostedApplicationInstanceGuard"
    )
    assert receipt.provider_evidence["evidence_source"] == "pytest_junit_properties"
    assert receipt.provider_evidence["selected_live_tests"] == 3
    assert receipt.provider_evidence["receipt_document_store_provider"] == "mongodb"

    assert receipt.domain_evidence["foreground_ready"] is True
    assert receipt.domain_evidence["real_index_before_restart"] is True
    assert receipt.domain_evidence["instance_conflict_verified"] is True
    assert receipt.domain_evidence["first_process_remained_ready"] is True
    assert receipt.domain_evidence["foreground_clean_stop"] is True
    assert receipt.domain_evidence["foreground_shutdown_reason"] == "signal.sigterm"
    assert receipt.domain_evidence["replacement_process_ready"] is True
    assert receipt.domain_evidence["instance_lock_released"] is True
    assert receipt.domain_evidence["replacement_clean_stop"] is True
    assert receipt.domain_evidence["restart_requested"] is True
    assert (
        receipt.domain_evidence["first_instance_id"]
        == "00000000-0000-4000-8000-000000000001"
    )
    assert (
        receipt.domain_evidence["second_instance_id"]
        == "00000000-0000-4000-8000-000000000002"
    )
    assert receipt.domain_evidence["instance_id_changed"] is True
    assert receipt.domain_evidence["first_attempt_exit_kind"] == "restart_requested"
    assert receipt.domain_evidence["first_attempt_cleanup_verified"] is True
    assert receipt.domain_evidence["first_lease_released"] is True
    assert receipt.domain_evidence["first_context_closed"] is True
    assert receipt.domain_evidence["stopped_events_verified"] is True
    assert receipt.domain_evidence["restart_events_verified"] is True
    assert receipt.domain_evidence["second_instance_ready"] is True
    assert receipt.domain_evidence["real_index_after_restart"] is True
    assert receipt.domain_evidence["profile_digest"] == _DIGEST
    assert receipt.domain_evidence["definition_digest"] == _DIGEST_B
    assert receipt.domain_evidence["profile_digest_preserved"] is True
    assert receipt.domain_evidence["definition_digest_preserved"] is True
    assert receipt.domain_evidence["final_exit_kind"] == "clean_stop"
    assert receipt.domain_evidence["final_cleanup_verified"] is True
    assert receipt.domain_evidence["final_lease_released"] is True
    assert receipt.domain_evidence["final_context_closed"] is True
    assert receipt.domain_evidence["final_lock_reacquired"] is True

    assert receipt.guardrails["mock_hosting_runtime"] is False
    assert receipt.guardrails["fake_supervisor"] is False
    assert receipt.guardrails["fake_engine"] is False
    assert receipt.guardrails["fake_instance_guard"] is False
    assert receipt.guardrails["http_test_client"] is False
    assert receipt.guardrails["direct_runtime_stop"] is False
    assert receipt.guardrails["direct_engine_stop"] is False
    assert receipt.guardrails["restart_http_endpoint"] is False
    assert receipt.guardrails["production_test_hook"] is False
    assert receipt.guardrails["manual_evidence_injection"] is False
    assert receipt.guardrails["inmemory_receipt_store"] is False
    assert receipt.guardrails["direct_mongodb_write"] is False
    assert receipt.guardrails["direct_pymongo_from_lkw"] is False
    assert receipt.guardrails["markdown_source_of_truth"] is False

    assert receipt.metadata["proof_runner"] == "run-lkw-hosting-proof.py"
    assert receipt.metadata["receipt_task"] == "APP-HOST-8E"
    assert (
        receipt.metadata["evidence_schema"]
        == "lkw.application_hosting_proof_evidence.v1"
    )
    assert receipt.metadata["recorded_from_live_run"] is True
    assert receipt.metadata["mongo_express_url"] == "http://127.0.0.1:8086"
    assert receipt.metadata["source_tests"] == list(_LIVE_NODES)


def test_build_application_hosting_proof_receipt_has_no_credentials() -> None:
    proof = _load_proof_module()
    evidence = _sample_evidence(proof)
    receipt = proof.build_application_hosting_proof_receipt(
        run_id="run-hosting-1",
        correlation_id="corr-hosting-1",
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
    assert "TemporaryDirectory" not in serialized
    assert "/tmp/" not in serialized
    assert "\\\\Temp\\\\" not in serialized


def test_parse_hosting_proof_junit_accepts_complete_evidence(tmp_path: Path) -> None:
    proof = _load_proof_module()
    props = _complete_props()
    junit = tmp_path / "ok.xml"
    _write_junit(
        junit,
        cases=[
            (
                "test_hosted_foreground_process_ready_index_and_instance_conflict",
                props[
                    "test_hosted_foreground_process_ready_index_and_instance_conflict"
                ],
                None,
            ),
            (
                "test_hosted_foreground_process_graceful_stop_releases_instance_lock",
                props[
                    "test_hosted_foreground_process_graceful_stop_releases_instance_lock"
                ],
                None,
            ),
            (
                "test_hosted_lkw_restart_creates_new_instance_and_accepts_work",
                props["test_hosted_lkw_restart_creates_new_instance_and_accepts_work"],
                None,
            ),
        ],
    )
    evidence = proof.parse_hosting_proof_junit(junit)
    assert evidence.foreground_ready is True
    assert evidence.foreground_shutdown_reason == "signal.sigbreak"
    assert evidence.first_instance_id == "00000000-0000-4000-8000-000000000001"
    assert evidence.second_instance_id == "00000000-0000-4000-8000-000000000002"
    assert evidence.profile_digest == _DIGEST
    assert evidence.definition_digest == _DIGEST_B
    assert evidence.final_exit_kind == "clean_stop"


@pytest.mark.parametrize(
    ("mutate", "expected_fragment"),
    [
        ("missing_testcase", "unexpected_testcase_count"),
        ("failed", "failed_testcase"),
        ("error", "errored_testcase"),
        ("skipped", "skipped_testcase"),
        ("missing_property", "missing_property"),
        ("blank_instance", "blank_property"),
        ("same_instance", "instance_ids_not_changed"),
        ("bad_profile", "invalid_profile_digest"),
        ("bad_definition", "invalid_definition_digest"),
        ("false_evidence", "false_required_evidence"),
        ("conflict", "conflicting_property"),
        ("malformed", "malformed_junit_xml"),
    ],
)
def test_parse_hosting_proof_junit_rejects_invalid_evidence(
    tmp_path: Path,
    mutate: str,
    expected_fragment: str,
) -> None:
    proof = _load_proof_module()
    props = _complete_props()
    junit = tmp_path / f"{mutate}.xml"

    if mutate == "malformed":
        junit.write_text("<not-xml", encoding="utf-8")
        with pytest.raises(proof.HostingProofEvidenceError) as exc_info:
            proof.parse_hosting_proof_junit(junit)
        assert expected_fragment in str(exc_info.value)
        return

    names = list(props.keys())
    cases: list[tuple[str, dict[str, str], str | None]] = [
        (names[0], dict(props[names[0]]), None),
        (names[1], dict(props[names[1]]), None),
        (names[2], dict(props[names[2]]), None),
    ]

    if mutate == "missing_testcase":
        cases = cases[:2]
    elif mutate == "failed":
        cases[0] = (cases[0][0], cases[0][1], "failure")
    elif mutate == "error":
        cases[1] = (cases[1][0], cases[1][1], "error")
    elif mutate == "skipped":
        cases[2] = (cases[2][0], cases[2][1], "skipped")
    elif mutate == "missing_property":
        del cases[0][1]["hosting.foreground_ready"]
    elif mutate == "blank_instance":
        cases[2][1]["hosting.first_instance_id"] = "   "
    elif mutate == "same_instance":
        cases[2][1]["hosting.second_instance_id"] = cases[2][1][
            "hosting.first_instance_id"
        ]
    elif mutate == "bad_profile":
        cases[2][1]["hosting.profile_digest"] = "not-a-digest"
    elif mutate == "bad_definition":
        cases[2][1]["hosting.definition_digest"] = "sha256:ZZ"
    elif mutate == "false_evidence":
        cases[0][1]["hosting.foreground_ready"] = "false"
    elif mutate == "conflict":
        cases[1][1]["hosting.foreground_ready"] = "false"

    _write_junit(junit, cases=cases)
    with pytest.raises(proof.HostingProofEvidenceError) as exc_info:
        proof.parse_hosting_proof_junit(junit)
    assert expected_fragment in str(exc_info.value)


def test_hosting_proof_script_uses_platform_receipt_recording() -> None:
    text = _read(_PROOF_SCRIPT)
    for node in _LIVE_NODES:
        assert node in text
    assert "--junitxml" in text
    assert "junit_family=legacy" in text
    assert "ProofReceipt" in text
    assert "record_and_verify_proof_receipt" in text
    assert "create_mongodb_integration" in text
    assert "MongoDBDocumentStoreIntegration" in text
    assert "proof_receipt_recorded=true" in text
    assert "proof_receipt_verified=true" in text
    assert "proof_receipt_query_verified=true" in text
    assert "proof_receipt_recording_failed" in text
    assert "hosting_proof_tests_failed" in text
    assert "hosting_proof_evidence_invalid" in text
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


def test_hosting_proof_pass_prints_after_receipt_verification() -> None:
    text = _read(_PROOF_SCRIPT)
    receipt_record_index = text.index("record_and_verify_proof_receipt")
    pass_output_index = text.index('print("proof_result=PASS")')
    assert receipt_record_index < pass_output_index


def test_hosting_proof_bat_runner_starts_only_mongodb_services() -> None:
    text = _read(_PROOF_BAT)
    assert "docker-compose.yml" in text
    assert "docker-compose.mongodb.yml" in text
    assert "lkw-mongodb" in text
    assert "lkw-mongo-express" in text
    assert "mongodb_container_healthy=true" in text
    assert "mongo_express_available=true" in text
    assert "--project applications/local_workspace_application" in text
    assert "INTERGRAX_MONGODB_URI" in text
    assert "INTERGRAX_MONGODB_DATABASE" in text
    assert "INTERGRAX_MONGODB_COLLECTION" in text

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


def test_accepted_hosting_tests_emit_required_evidence_keys() -> None:
    foreground = _read(_FOREGROUND_TEST)
    restart = _read(_RESTART_TEST)
    combined = foreground + "\n" + restart
    for key in _REQUIRED_EVIDENCE_KEYS:
        assert key in combined
    assert "record_property" in foreground
    assert "record_property" in restart
    assert (
        "test_hosted_foreground_process_ready_index_and_instance_conflict" in foreground
    )
    assert (
        "test_hosted_foreground_process_graceful_stop_releases_instance_lock"
        in foreground
    )
    assert "test_hosted_lkw_restart_creates_new_instance_and_accepts_work" in restart


def test_public_reviewer_document_contains_hosting_steps() -> None:
    text = _read(_PUBLIC_PLATFORM_PROOF)
    assert "## Step 10 — Run the Application Hosting proof" in text
    assert (
        "## Step 11 — Inspect the Application Hosting ProofReceipt in Mongo Express"
        in text
    )
    assert "run-lkw-hosting-proof.bat" in text
    assert "proof_kind=platform_application_hosting" in text
    assert "proof_tests_passed=3" in text
    assert "instance_conflict_verified=true" in text
    assert "instance_lock_released=true" in text
    assert "instance_id_changed=true" in text
    assert "profile_digest_preserved=true" in text
    assert "definition_digest_preserved=true" in text
    assert "real_index_after_restart=true" in text
    assert "final_lock_reacquired=true" in text
    assert "proof_receipt_recorded=true" in text
    assert "proof_receipt_verified=true" in text
    assert "proof_receipt_query_verified=true" in text
    assert "markdown_source_of_truth=false" in text
    assert "MongoDB ProofReceipt is the authoritative" in text
    step10_idx = text.index("## Step 10 — Run the Application Hosting proof")
    completion_idx = text.index("## Core Platform Proof completion")
    assert step10_idx < completion_idx
    assert (
        text.index(
            "## Step 11 — Inspect the Application Hosting ProofReceipt in Mongo Express"
        )
        < completion_idx
    )
    core_shortcuts = text.split("## Core reviewer shortcuts", 1)[1].split(
        "## Optional Windows reviewer shortcut", 1
    )[0]
    assert "run-lkw-hosting-proof.bat" in core_shortcuts
    assert "platform_application_hosting" in core_shortcuts
    assert "run-lkw-windows-interaction-proof.bat" not in core_shortcuts
