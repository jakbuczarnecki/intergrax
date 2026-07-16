#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LKW Application Hosting proof runner with ProofReceipt recording (APP-HOST-8E)."""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from intergrax.integrations.providers.document_store.mongodb.bundle import (
    create_mongodb_integration,
)
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    MongoDBDocumentStoreIntegration,
)
from intergrax.proofs.receipts.contracts import ProofReceipt, ProofReceiptResult
from intergrax.proofs.receipts.recording import (
    ProofReceiptVerificationError,
    record_and_verify_proof_receipt,
)

_APPLICATION_ID = "local_workspace"
_PROOF_KIND = "platform_application_hosting"
_PROOF_RUNNER = "run-lkw-hosting-proof.py"
_RECEIPT_TASK = "APP-HOST-8E"
_EVIDENCE_SCHEMA = "lkw.application_hosting_proof_evidence.v1"
_DEFAULT_MONGO_EXPRESS_URL = "http://127.0.0.1:8086"
_PYTEST_TIMEOUT_SECONDS = 240
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHUTDOWN_REASONS = frozenset({"signal.sigterm", "signal.sigbreak"})

_REPO_ROOT = Path(__file__).resolve().parents[3]

_LIVE_TEST_NODES = (
    "applications/local_workspace_application/tests/hosting/test_hosted_foreground_process.py::test_hosted_foreground_process_ready_index_and_instance_conflict",
    "applications/local_workspace_application/tests/hosting/test_hosted_foreground_process.py::test_hosted_foreground_process_graceful_stop_releases_instance_lock",
    "applications/local_workspace_application/tests/hosting/test_hosted_restart_live.py::test_hosted_lkw_restart_creates_new_instance_and_accepts_work",
)

_EXPECTED_TESTCASE_NAMES = frozenset(
    {
        "test_hosted_foreground_process_ready_index_and_instance_conflict",
        "test_hosted_foreground_process_graceful_stop_releases_instance_lock",
        "test_hosted_lkw_restart_creates_new_instance_and_accepts_work",
    }
)

_REQUIRED_TRUE_KEYS = (
    "hosting.foreground_ready",
    "hosting.real_index_before_restart",
    "hosting.instance_conflict_verified",
    "hosting.first_process_remained_ready",
    "hosting.foreground_clean_stop",
    "hosting.replacement_process_ready",
    "hosting.instance_lock_released",
    "hosting.replacement_clean_stop",
    "hosting.restart_requested",
    "hosting.instance_id_changed",
    "hosting.first_attempt_cleanup_verified",
    "hosting.first_lease_released",
    "hosting.first_context_closed",
    "hosting.stopped_events_verified",
    "hosting.restart_events_verified",
    "hosting.second_instance_ready",
    "hosting.real_index_after_restart",
    "hosting.profile_digest_preserved",
    "hosting.definition_digest_preserved",
    "hosting.final_cleanup_verified",
    "hosting.final_lease_released",
    "hosting.final_context_closed",
    "hosting.final_lock_reacquired",
)


class HostingProofEvidenceError(ValueError):
    """Raised when JUnit hosting proof evidence is missing or invalid."""


@dataclass(frozen=True, slots=True)
class HostingProofEvidence:
    """Validated hosting proof evidence collected from accepted live tests."""

    schema_version: str
    foreground_ready: bool
    real_index_before_restart: bool
    instance_conflict_verified: bool
    first_process_remained_ready: bool
    foreground_clean_stop: bool
    foreground_shutdown_reason: str
    replacement_process_ready: bool
    instance_lock_released: bool
    replacement_clean_stop: bool
    restart_requested: bool
    first_instance_id: str
    second_instance_id: str
    instance_id_changed: bool
    first_attempt_exit_kind: str
    first_attempt_cleanup_verified: bool
    first_lease_released: bool
    first_context_closed: bool
    stopped_events_verified: bool
    restart_events_verified: bool
    second_instance_ready: bool
    real_index_after_restart: bool
    profile_digest: str
    definition_digest: str
    profile_digest_preserved: bool
    definition_digest_preserved: bool
    final_exit_kind: str
    final_cleanup_verified: bool
    final_lease_released: bool
    final_context_closed: bool
    final_lock_reacquired: bool


def build_application_hosting_proof_id(run_id: str) -> str:
    """Stable proof receipt identity for an application-hosting proof run."""
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must not be blank")
    return f"{_APPLICATION_ID}:{_PROOF_KIND}:{normalized_run_id}"


def build_application_hosting_proof_receipt(
    *,
    run_id: str,
    correlation_id: str,
    evidence: HostingProofEvidence,
    mongo_express_url: str,
) -> ProofReceipt:
    """Build a structured ProofReceipt from validated hosting proof evidence."""
    return ProofReceipt(
        proof_id=build_application_hosting_proof_id(run_id),
        proof_kind=_PROOF_KIND,
        application_id=_APPLICATION_ID,
        result=ProofReceiptResult.PASS,
        run_id=run_id,
        correlation_id=correlation_id,
        task_id=None,
        provider_evidence={
            "hosting_platform": "intergrax_application_hosting",
            "foreground_entrypoint": "python_-m_local_workspace_application.hosting",
            "foreground_execution": "real_subprocess",
            "runtime_surface": "fastapi_uvicorn",
            "supervisor": "HostedApplicationSupervisor",
            "engine": "HostedApplicationEngine",
            "instance_guard": "FileHostedApplicationInstanceGuard",
            "evidence_source": "pytest_junit_properties",
            "selected_live_tests": 3,
            "receipt_document_store_provider": "mongodb",
        },
        domain_evidence={
            "foreground_ready": evidence.foreground_ready,
            "real_index_before_restart": evidence.real_index_before_restart,
            "instance_conflict_verified": evidence.instance_conflict_verified,
            "first_process_remained_ready": evidence.first_process_remained_ready,
            "foreground_clean_stop": evidence.foreground_clean_stop,
            "foreground_shutdown_reason": evidence.foreground_shutdown_reason,
            "replacement_process_ready": evidence.replacement_process_ready,
            "instance_lock_released": evidence.instance_lock_released,
            "replacement_clean_stop": evidence.replacement_clean_stop,
            "restart_requested": evidence.restart_requested,
            "first_instance_id": evidence.first_instance_id,
            "second_instance_id": evidence.second_instance_id,
            "instance_id_changed": evidence.instance_id_changed,
            "first_attempt_exit_kind": evidence.first_attempt_exit_kind,
            "first_attempt_cleanup_verified": evidence.first_attempt_cleanup_verified,
            "first_lease_released": evidence.first_lease_released,
            "first_context_closed": evidence.first_context_closed,
            "stopped_events_verified": evidence.stopped_events_verified,
            "restart_events_verified": evidence.restart_events_verified,
            "second_instance_ready": evidence.second_instance_ready,
            "real_index_after_restart": evidence.real_index_after_restart,
            "profile_digest": evidence.profile_digest,
            "definition_digest": evidence.definition_digest,
            "profile_digest_preserved": evidence.profile_digest_preserved,
            "definition_digest_preserved": evidence.definition_digest_preserved,
            "final_exit_kind": evidence.final_exit_kind,
            "final_cleanup_verified": evidence.final_cleanup_verified,
            "final_lease_released": evidence.final_lease_released,
            "final_context_closed": evidence.final_context_closed,
            "final_lock_reacquired": evidence.final_lock_reacquired,
        },
        guardrails={
            "mock_hosting_runtime": False,
            "fake_supervisor": False,
            "fake_engine": False,
            "fake_instance_guard": False,
            "http_test_client": False,
            "direct_runtime_stop": False,
            "direct_engine_stop": False,
            "restart_http_endpoint": False,
            "production_test_hook": False,
            "manual_evidence_injection": False,
            "inmemory_receipt_store": False,
            "direct_mongodb_write": False,
            "direct_pymongo_from_lkw": False,
            "markdown_source_of_truth": False,
        },
        metadata={
            "proof_runner": _PROOF_RUNNER,
            "receipt_task": _RECEIPT_TASK,
            "evidence_schema": _EVIDENCE_SCHEMA,
            "recorded_from_live_run": True,
            "mongo_express_url": mongo_express_url,
            "source_tests": list(_LIVE_TEST_NODES),
            "operating_system": platform.system(),
            "python_version": platform.python_version(),
        },
    )


def _resolve_host_mongodb_uri() -> str | None:
    explicit = os.environ.get("INTERGRAX_MONGODB_URI", "").strip()
    if explicit:
        return explicit

    username = (
        os.environ.get("LKW_MONGODB_ROOT_USERNAME", "intergrax").strip() or "intergrax"
    )
    password = (
        os.environ.get("LKW_MONGODB_ROOT_PASSWORD", "intergrax-local-dev-only").strip()
        or "intergrax-local-dev-only"
    )
    database = (
        os.environ.get("LKW_MONGODB_DATABASE", "intergrax_proofs").strip()
        or "intergrax_proofs"
    )
    host_port = os.environ.get("LKW_MONGODB_HOST_PORT", "27018").strip() or "27018"
    return f"mongodb://{username}:{password}@127.0.0.1:{host_port}/{database}?authSource=admin"


def ensure_mongodb_env() -> None:
    """Populate host-visible MongoDB provider environment for platform resolution."""
    if not os.environ.get("INTERGRAX_MONGODB_URI", "").strip():
        resolved = _resolve_host_mongodb_uri()
        if resolved:
            os.environ["INTERGRAX_MONGODB_URI"] = resolved
    if not os.environ.get("INTERGRAX_MONGODB_DATABASE", "").strip():
        os.environ["INTERGRAX_MONGODB_DATABASE"] = (
            os.environ.get("LKW_MONGODB_DATABASE", "intergrax_proofs").strip()
            or "intergrax_proofs"
        )
    if not os.environ.get("INTERGRAX_MONGODB_COLLECTION", "").strip():
        os.environ["INTERGRAX_MONGODB_COLLECTION"] = (
            os.environ.get("LKW_MONGODB_COLLECTION", "proof_receipts").strip()
            or "proof_receipts"
        )


def resolve_mongodb_document_store():
    """Resolve MongoDB DocumentStore through the platform provider factory."""
    ensure_mongodb_env()
    bundle = create_mongodb_integration()
    integration = bundle.document_store
    if not isinstance(integration, MongoDBDocumentStoreIntegration):
        raise TypeError("integration_not_mongodb_document_store")
    store = integration.as_document_store()
    if store is None:
        raise RuntimeError("document_store_adapter_unresolved")
    return integration, store


def record_application_hosting_proof_receipt(
    receipt: ProofReceipt,
) -> tuple[ProofReceipt, MongoDBDocumentStoreIntegration]:
    """Persist and verify an application-hosting proof receipt through the platform store."""
    integration, document_store = resolve_mongodb_document_store()
    verified = record_and_verify_proof_receipt(
        receipt, document_store, owns_document_store=True
    )
    return verified, integration


def _collect_properties(testcase: ET.Element) -> dict[str, str]:
    properties: dict[str, str] = {}
    for props in testcase.findall("properties"):
        for prop in props.findall("property"):
            name = prop.attrib.get("name", "").strip()
            value = prop.attrib.get("value", "")
            if not name:
                continue
            if name in properties and properties[name] != value:
                raise HostingProofEvidenceError(f"conflicting_property:{name}")
            properties[name] = value
    return properties


def _require_true(properties: dict[str, str], key: str) -> bool:
    value = properties.get(key)
    if value is None:
        raise HostingProofEvidenceError(f"missing_property:{key}")
    if value != "true":
        raise HostingProofEvidenceError(f"false_required_evidence:{key}")
    return True


def _require_text(properties: dict[str, str], key: str) -> str:
    value = properties.get(key)
    if value is None:
        raise HostingProofEvidenceError(f"missing_property:{key}")
    normalized = value.strip()
    if not normalized:
        raise HostingProofEvidenceError(f"blank_property:{key}")
    return normalized


def parse_hosting_proof_junit(junit_xml: str | Path) -> HostingProofEvidence:
    """Parse and validate hosting proof evidence from a legacy JUnit XML report."""
    path = Path(junit_xml)
    if not path.is_file():
        raise HostingProofEvidenceError("missing_junit_file")
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        raise HostingProofEvidenceError("malformed_junit_xml") from exc

    testcases = root.findall(".//testcase")
    if len(testcases) != 3:
        raise HostingProofEvidenceError(f"unexpected_testcase_count:{len(testcases)}")

    names: list[str] = []
    merged: dict[str, str] = {}
    for testcase in testcases:
        name = testcase.attrib.get("name", "").strip()
        if not name:
            raise HostingProofEvidenceError("missing_testcase_name")
        names.append(name)
        if testcase.find("failure") is not None:
            raise HostingProofEvidenceError(f"failed_testcase:{name}")
        if testcase.find("error") is not None:
            raise HostingProofEvidenceError(f"errored_testcase:{name}")
        if testcase.find("skipped") is not None:
            raise HostingProofEvidenceError(f"skipped_testcase:{name}")
        case_props = _collect_properties(testcase)
        for key, value in case_props.items():
            if key in merged and merged[key] != value:
                raise HostingProofEvidenceError(f"conflicting_property:{key}")
            merged[key] = value

    name_set = set(names)
    if name_set != _EXPECTED_TESTCASE_NAMES:
        missing = sorted(_EXPECTED_TESTCASE_NAMES - name_set)
        unexpected = sorted(name_set - _EXPECTED_TESTCASE_NAMES)
        if missing:
            raise HostingProofEvidenceError(f"missing_testcase:{','.join(missing)}")
        if unexpected:
            raise HostingProofEvidenceError(
                f"unexpected_testcase:{','.join(unexpected)}"
            )
        raise HostingProofEvidenceError("duplicate_expected_testcase")

    for key in _REQUIRED_TRUE_KEYS:
        _require_true(merged, key)

    shutdown_reason = _require_text(merged, "hosting.foreground_shutdown_reason")
    if shutdown_reason not in _SHUTDOWN_REASONS:
        raise HostingProofEvidenceError("invalid_foreground_shutdown_reason")

    first_instance_id = _require_text(merged, "hosting.first_instance_id")
    second_instance_id = _require_text(merged, "hosting.second_instance_id")
    if first_instance_id == second_instance_id:
        raise HostingProofEvidenceError("instance_ids_not_changed")

    first_attempt_exit_kind = _require_text(merged, "hosting.first_attempt_exit_kind")
    if first_attempt_exit_kind != "restart_requested":
        raise HostingProofEvidenceError("invalid_first_attempt_exit_kind")

    final_exit_kind = _require_text(merged, "hosting.final_exit_kind")
    if final_exit_kind != "clean_stop":
        raise HostingProofEvidenceError("invalid_final_exit_kind")

    profile_digest = _require_text(merged, "hosting.profile_digest")
    if _DIGEST_RE.fullmatch(profile_digest) is None:
        raise HostingProofEvidenceError("invalid_profile_digest")

    definition_digest = _require_text(merged, "hosting.definition_digest")
    if _DIGEST_RE.fullmatch(definition_digest) is None:
        raise HostingProofEvidenceError("invalid_definition_digest")

    return HostingProofEvidence(
        schema_version=_EVIDENCE_SCHEMA,
        foreground_ready=True,
        real_index_before_restart=True,
        instance_conflict_verified=True,
        first_process_remained_ready=True,
        foreground_clean_stop=True,
        foreground_shutdown_reason=shutdown_reason,
        replacement_process_ready=True,
        instance_lock_released=True,
        replacement_clean_stop=True,
        restart_requested=True,
        first_instance_id=first_instance_id,
        second_instance_id=second_instance_id,
        instance_id_changed=True,
        first_attempt_exit_kind=first_attempt_exit_kind,
        first_attempt_cleanup_verified=True,
        first_lease_released=True,
        first_context_closed=True,
        stopped_events_verified=True,
        restart_events_verified=True,
        second_instance_ready=True,
        real_index_after_restart=True,
        profile_digest=profile_digest,
        definition_digest=definition_digest,
        profile_digest_preserved=True,
        definition_digest_preserved=True,
        final_exit_kind=final_exit_kind,
        final_cleanup_verified=True,
        final_lease_released=True,
        final_context_closed=True,
        final_lock_reacquired=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the accepted APP-HOST-8C/8D LKW hosting live tests and record a "
            "platform Application Hosting ProofReceipt."
        ),
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id for the proof receipt (generated when omitted).",
    )
    parser.add_argument(
        "--correlation-id",
        default="",
        help="Optional correlation id (defaults to run_id).",
    )
    parser.add_argument(
        "--mongo-express",
        default=os.environ.get("LKW_MONGO_EXPRESS_URL", _DEFAULT_MONGO_EXPRESS_URL),
        help="Mongo Express URL for reviewer hints (default: http://127.0.0.1:8086).",
    )
    return parser.parse_args()


def _fail(reason: str, **fields: object) -> int:
    print("proof_result=FAIL")
    print(f"failure_reason={reason}")
    print("proof_receipt_recorded=false")
    print("proof_receipt_verified=false")
    print("proof_receipt_query_verified=false")
    for key, value in fields.items():
        print(f"{key}={value}")
    return 1


def _fail_receipt_recording(error: BaseException) -> int:
    print("proof_result=FAIL")
    print("failure_reason=proof_receipt_recording_failed")
    print("proof_workload_result=PASS")
    print("proof_receipt_recorded=false")
    print("proof_receipt_verified=false")
    print("proof_receipt_query_verified=false")
    print(f"receipt_error={type(error).__name__}")
    return 1


def _print_pass_output(
    *,
    evidence: HostingProofEvidence,
    verified_receipt: ProofReceipt,
    mongo_express_url: str,
    correlation_id: str,
) -> None:
    print("proof_result=PASS")
    print(f"proof_kind={_PROOF_KIND}")
    print("proof_tests_passed=3")
    print("foreground_ready=true")
    print("real_index_before_restart=true")
    print("instance_conflict_verified=true")
    print("first_process_remained_ready=true")
    print("foreground_clean_stop=true")
    print(f"foreground_shutdown_reason={evidence.foreground_shutdown_reason}")
    print("replacement_process_ready=true")
    print("instance_lock_released=true")
    print("replacement_clean_stop=true")
    print("restart_requested=true")
    print(f"first_instance_id={evidence.first_instance_id}")
    print(f"second_instance_id={evidence.second_instance_id}")
    print("instance_id_changed=true")
    print("first_attempt_exit_kind=restart_requested")
    print("first_attempt_cleanup_verified=true")
    print("first_lease_released=true")
    print("first_context_closed=true")
    print("stopped_events_verified=true")
    print("restart_events_verified=true")
    print("second_instance_ready=true")
    print("real_index_after_restart=true")
    print(f"profile_digest={evidence.profile_digest}")
    print(f"definition_digest={evidence.definition_digest}")
    print("profile_digest_preserved=true")
    print("definition_digest_preserved=true")
    print("final_exit_kind=clean_stop")
    print("final_cleanup_verified=true")
    print("final_lease_released=true")
    print("final_context_closed=true")
    print("final_lock_reacquired=true")
    print("proof_receipt_recorded=true")
    print("proof_receipt_verified=true")
    print("proof_receipt_query_verified=true")
    print("proof_receipt_store=platform")
    print(f"document_store_provider={MONGODB_DOCUMENT_STORE_PROVIDER_ID}")
    print(f"proof_receipt_id={verified_receipt.proof_id}")
    print(f"proof_receipt_run_id={verified_receipt.run_id}")
    print(f"proof_receipt_result={verified_receipt.result.value}")
    print(f"correlation_id={correlation_id}")
    print(f"mongo_express_url={mongo_express_url}")
    print("inmemory_receipt_store=false")
    print("direct_mongodb_write=false")
    print("direct_pymongo_from_lkw=false")
    print("markdown_source_of_truth=false")
    print("manual_evidence_injection=false")


def _run_accepted_hosting_tests(junit_path: Path, basetemp: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        *_LIVE_TEST_NODES,
        "-q",
        "-o",
        "junit_family=legacy",
        f"--junitxml={junit_path}",
        f"--basetemp={basetemp}",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(_REPO_ROOT),
            check=False,
            timeout=_PYTEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return -1
    return int(completed.returncode)


def main() -> int:
    args = _parse_args()
    run_id = args.run_id.strip() or f"lkw-hosting-{uuid.uuid4().hex[:12]}"
    correlation_id = args.correlation_id.strip() or run_id
    mongo_express_url = args.mongo_express.strip() or _DEFAULT_MONGO_EXPRESS_URL

    with tempfile.TemporaryDirectory(prefix="lkw-hosting-proof-") as temp_dir:
        temp_root = Path(temp_dir)
        junit_path = temp_root / "hosting-proof-junit.xml"
        basetemp = temp_root / "pytest-basetemp"
        basetemp.mkdir(parents=True, exist_ok=True)

        returncode = _run_accepted_hosting_tests(junit_path, basetemp)
        if returncode != 0:
            return _fail(
                "hosting_proof_tests_failed",
                pytest_returncode=returncode,
            )

        try:
            evidence = parse_hosting_proof_junit(junit_path)
        except HostingProofEvidenceError:
            return _fail("hosting_proof_evidence_invalid")

        receipt = build_application_hosting_proof_receipt(
            run_id=run_id,
            correlation_id=correlation_id,
            evidence=evidence,
            mongo_express_url=mongo_express_url,
        )

        try:
            verified_receipt, _integration = record_application_hosting_proof_receipt(
                receipt
            )
        except (
            ProofReceiptVerificationError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return _fail_receipt_recording(exc)

        _print_pass_output(
            evidence=evidence,
            verified_receipt=verified_receipt,
            mongo_express_url=mongo_express_url,
            correlation_id=correlation_id,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
