#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LKW Windows PowerShell interaction proof runner with ProofReceipt (LKW.6C)."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
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
_PROOF_KIND = "platform_windows_interaction"
_PROOF_RUNNER = "run-lkw-windows-interaction-proof.py"
_RECEIPT_TASK = "LKW.6C"
_EVIDENCE_SCHEMA = "lkw.windows_interaction_proof_evidence.v1"
_DEFAULT_MONGO_EXPRESS_URL = "http://127.0.0.1:8086"
_PYTEST_TIMEOUT_SECONDS = 180
_ADAPTER_ID = "lkw.windows_powershell"
_INTAKE_ENDPOINT = "/v1/interactions/intake"
_INTERACTION_SURFACE = "lab_json"
_INTERACTION_CHANNEL = "lab"
_POWERSHELL_RUNTIME = "Windows PowerShell"

_REPO_ROOT = Path(__file__).resolve().parents[3]

_LIVE_TEST_NODE = "applications/local_workspace_application/tests/interactions/test_windows_powershell_interaction_live.py::test_windows_powershell_adapter_executes_real_lkw_interactions"
_EXPECTED_TESTCASE_NAME = (
    "test_windows_powershell_adapter_executes_real_lkw_interactions"
)

_REQUIRED_TRUE_KEYS = (
    "windows_interaction.hosted_ready",
    "windows_interaction.adapter_invoked",
    "windows_interaction.index_executed",
    "windows_interaction.search_executed",
    "windows_interaction.task_ids_distinct",
    "windows_interaction.run_ids_distinct",
    "windows_interaction.graceful_stop",
    "windows_interaction.cleanup_verified",
)


class WindowsInteractionProofEvidenceError(ValueError):
    """Raised when JUnit Windows interaction proof evidence is missing or invalid."""


@dataclass(frozen=True, slots=True)
class WindowsInteractionProofEvidence:
    """Validated Windows interaction proof evidence from the accepted live test."""

    schema_version: str
    hosted_ready: bool
    adapter_invoked: bool
    adapter_id: str
    powershell_runtime: str
    transport: str
    intake_endpoint: str
    interaction_surface: str
    interaction_channel: str
    index_executed: bool
    index_state: str
    index_task_id: str
    index_run_id: str
    search_executed: bool
    search_state: str
    search_task_id: str
    search_run_id: str
    task_ids_distinct: bool
    run_ids_distinct: bool
    graceful_stop: bool
    cleanup_verified: bool


def build_windows_interaction_proof_id(run_id: str) -> str:
    """Stable proof receipt identity for a Windows interaction proof run."""
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must not be blank")
    return f"{_APPLICATION_ID}:{_PROOF_KIND}:{normalized_run_id}"


def build_windows_interaction_proof_receipt(
    *,
    run_id: str,
    correlation_id: str,
    evidence: WindowsInteractionProofEvidence,
    mongo_express_url: str,
) -> ProofReceipt:
    """Build a structured ProofReceipt from validated Windows interaction evidence."""
    return ProofReceipt(
        proof_id=build_windows_interaction_proof_id(run_id),
        proof_kind=_PROOF_KIND,
        application_id=_APPLICATION_ID,
        result=ProofReceiptResult.PASS,
        run_id=run_id,
        correlation_id=correlation_id,
        task_id=None,
        provider_evidence={
            "os_family": "windows",
            "os_adapter": _ADAPTER_ID,
            "client_runtime": _POWERSHELL_RUNTIME,
            "transport": "http",
            "interaction_surface": _INTERACTION_SURFACE,
            "interaction_channel": _INTERACTION_CHANNEL,
            "intake_endpoint": _INTAKE_ENDPOINT,
            "intake_service": "InteractionIntakeService",
            "execution_boundary": "LocalWorkspaceTaskExecutor",
            "orchestrator": "NexusLoop",
            "hosted_entrypoint": "python_-m_local_workspace_application.hosting",
            "evidence_source": "pytest_junit_properties",
            "selected_live_tests": 1,
            "receipt_document_store_provider": "mongodb",
        },
        domain_evidence={
            "hosted_ready": evidence.hosted_ready,
            "adapter_invoked": evidence.adapter_invoked,
            "adapter_id": evidence.adapter_id,
            "powershell_runtime": evidence.powershell_runtime,
            "transport": evidence.transport,
            "intake_endpoint": evidence.intake_endpoint,
            "interaction_surface": evidence.interaction_surface,
            "interaction_channel": evidence.interaction_channel,
            "index_executed": evidence.index_executed,
            "index_state": evidence.index_state,
            "index_task_id": evidence.index_task_id,
            "index_run_id": evidence.index_run_id,
            "search_executed": evidence.search_executed,
            "search_state": evidence.search_state,
            "search_task_id": evidence.search_task_id,
            "search_run_id": evidence.search_run_id,
            "task_ids_distinct": evidence.task_ids_distinct,
            "run_ids_distinct": evidence.run_ids_distinct,
            "graceful_stop": evidence.graceful_stop,
            "cleanup_verified": evidence.cleanup_verified,
        },
        guardrails={
            "direct_run_endpoint": False,
            "direct_task_construction": False,
            "direct_task_executor_call": False,
            "direct_nexus_call": False,
            "direct_agent_call": False,
            "mock_http_server": False,
            "http_test_client": False,
            "fake_interaction_service": False,
            "fake_hosted_application": False,
            "new_platform_interaction_adapter": False,
            "generic_os_hosting_adapter": False,
            "service_installation": False,
            "powershell_invocation_via_shell": False,
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
            "source_test": _LIVE_TEST_NODE,
            "adapter_script": "invoke-lkw-interaction.ps1",
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
    return (
        f"mongodb://{username}:{password}@127.0.0.1:{host_port}/"
        f"{database}?authSource=admin"
    )


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


def record_windows_interaction_proof_receipt(
    receipt: ProofReceipt,
) -> tuple[ProofReceipt, MongoDBDocumentStoreIntegration]:
    """Persist and verify a Windows interaction proof receipt through the platform store."""
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
                raise WindowsInteractionProofEvidenceError(
                    f"conflicting_property:{name}"
                )
            properties[name] = value
    return properties


def _require_true(properties: dict[str, str], key: str) -> bool:
    value = properties.get(key)
    if value is None:
        raise WindowsInteractionProofEvidenceError(f"missing_property:{key}")
    if value != "true":
        raise WindowsInteractionProofEvidenceError(f"false_required_evidence:{key}")
    return True


def _require_text(properties: dict[str, str], key: str) -> str:
    value = properties.get(key)
    if value is None:
        raise WindowsInteractionProofEvidenceError(f"missing_property:{key}")
    normalized = value.strip()
    if not normalized:
        raise WindowsInteractionProofEvidenceError(f"blank_property:{key}")
    return normalized


def parse_windows_interaction_proof_junit(
    junit_xml: str | Path,
) -> WindowsInteractionProofEvidence:
    """Parse and validate Windows interaction proof evidence from legacy JUnit XML."""
    path = Path(junit_xml)
    if not path.is_file():
        raise WindowsInteractionProofEvidenceError("missing_junit_file")
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        raise WindowsInteractionProofEvidenceError("malformed_junit_xml") from exc

    testcases = root.findall(".//testcase")
    if len(testcases) != 1:
        raise WindowsInteractionProofEvidenceError(
            f"unexpected_testcase_count:{len(testcases)}"
        )

    testcase = testcases[0]
    name = testcase.attrib.get("name", "").strip()
    if not name:
        raise WindowsInteractionProofEvidenceError("missing_testcase_name")
    if name != _EXPECTED_TESTCASE_NAME:
        raise WindowsInteractionProofEvidenceError(f"unexpected_testcase:{name}")
    if testcase.find("failure") is not None:
        raise WindowsInteractionProofEvidenceError(f"failed_testcase:{name}")
    if testcase.find("error") is not None:
        raise WindowsInteractionProofEvidenceError(f"errored_testcase:{name}")
    if testcase.find("skipped") is not None:
        raise WindowsInteractionProofEvidenceError(f"skipped_testcase:{name}")

    properties = _collect_properties(testcase)
    for key in _REQUIRED_TRUE_KEYS:
        _require_true(properties, key)

    adapter_id = _require_text(properties, "windows_interaction.adapter_id")
    if adapter_id != _ADAPTER_ID:
        raise WindowsInteractionProofEvidenceError("invalid_adapter_id")

    powershell_runtime = _require_text(
        properties, "windows_interaction.powershell_runtime"
    )
    if powershell_runtime != _POWERSHELL_RUNTIME:
        raise WindowsInteractionProofEvidenceError("invalid_powershell_runtime")

    transport = _require_text(properties, "windows_interaction.transport")
    if transport != "http":
        raise WindowsInteractionProofEvidenceError("invalid_transport")

    intake_endpoint = _require_text(properties, "windows_interaction.intake_endpoint")
    if intake_endpoint != _INTAKE_ENDPOINT:
        raise WindowsInteractionProofEvidenceError("invalid_endpoint")

    interaction_surface = _require_text(
        properties, "windows_interaction.interaction_surface"
    )
    if interaction_surface != _INTERACTION_SURFACE:
        raise WindowsInteractionProofEvidenceError("invalid_interaction_surface")

    interaction_channel = _require_text(
        properties, "windows_interaction.interaction_channel"
    )
    if interaction_channel != _INTERACTION_CHANNEL:
        raise WindowsInteractionProofEvidenceError("invalid_interaction_channel")

    index_state = _require_text(properties, "windows_interaction.index_state")
    if index_state != "completed":
        raise WindowsInteractionProofEvidenceError("invalid_index_state")

    search_state = _require_text(properties, "windows_interaction.search_state")
    if search_state != "completed":
        raise WindowsInteractionProofEvidenceError("invalid_search_state")

    index_task_id = _require_text(properties, "windows_interaction.index_task_id")
    index_run_id = _require_text(properties, "windows_interaction.index_run_id")
    search_task_id = _require_text(properties, "windows_interaction.search_task_id")
    search_run_id = _require_text(properties, "windows_interaction.search_run_id")

    if index_task_id == search_task_id:
        raise WindowsInteractionProofEvidenceError("same_task_ids")
    if index_run_id == search_run_id:
        raise WindowsInteractionProofEvidenceError("same_run_ids")

    return WindowsInteractionProofEvidence(
        schema_version=_EVIDENCE_SCHEMA,
        hosted_ready=True,
        adapter_invoked=True,
        adapter_id=adapter_id,
        powershell_runtime=powershell_runtime,
        transport=transport,
        intake_endpoint=intake_endpoint,
        interaction_surface=interaction_surface,
        interaction_channel=interaction_channel,
        index_executed=True,
        index_state=index_state,
        index_task_id=index_task_id,
        index_run_id=index_run_id,
        search_executed=True,
        search_state=search_state,
        search_task_id=search_task_id,
        search_run_id=search_run_id,
        task_ids_distinct=True,
        run_ids_distinct=True,
        graceful_stop=True,
        cleanup_verified=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the accepted LKW.6C Windows PowerShell interaction live test "
            "and record a platform Windows Interaction ProofReceipt."
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
    evidence: WindowsInteractionProofEvidence,
    verified_receipt: ProofReceipt,
    mongo_express_url: str,
    correlation_id: str,
) -> None:
    print("proof_result=PASS")
    print(f"proof_kind={_PROOF_KIND}")
    print("proof_tests_passed=1")
    print("os_family=windows")
    print("adapter_invoked=true")
    print(f"adapter_id={evidence.adapter_id}")
    print(f"powershell_runtime={evidence.powershell_runtime}")
    print(f"transport={evidence.transport}")
    print(f"intake_endpoint={evidence.intake_endpoint}")
    print(f"interaction_surface={evidence.interaction_surface}")
    print(f"interaction_channel={evidence.interaction_channel}")
    print("hosted_ready=true")
    print("index_executed=true")
    print(f"index_state={evidence.index_state}")
    print(f"index_task_id={evidence.index_task_id}")
    print(f"index_run_id={evidence.index_run_id}")
    print("search_executed=true")
    print(f"search_state={evidence.search_state}")
    print(f"search_task_id={evidence.search_task_id}")
    print(f"search_run_id={evidence.search_run_id}")
    print("task_ids_distinct=true")
    print("run_ids_distinct=true")
    print("graceful_stop=true")
    print("cleanup_verified=true")
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
    print("direct_run_endpoint=false")
    print("direct_task_executor_call=false")
    print("direct_nexus_call=false")
    print("fake_interaction_service=false")
    print("new_platform_interaction_adapter=false")
    print("generic_os_hosting_adapter=false")
    print("service_installation=false")
    print("manual_evidence_injection=false")
    print("inmemory_receipt_store=false")
    print("direct_mongodb_write=false")
    print("direct_pymongo_from_lkw=false")
    print("markdown_source_of_truth=false")


def _run_accepted_live_test(junit_path: Path, basetemp: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        _LIVE_TEST_NODE,
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
    if os.name != "nt":
        return _fail("windows_required")
    if not shutil.which("powershell.exe"):
        return _fail("windows_powershell_unavailable")

    run_id = args.run_id.strip() or f"lkw-windows-interaction-{uuid.uuid4().hex[:12]}"
    correlation_id = args.correlation_id.strip() or run_id
    mongo_express_url = args.mongo_express.strip() or _DEFAULT_MONGO_EXPRESS_URL

    with tempfile.TemporaryDirectory(
        prefix="lkw-windows-interaction-proof-"
    ) as temp_dir:
        temp_root = Path(temp_dir)
        junit_path = temp_root / "windows-interaction-proof-junit.xml"
        basetemp = temp_root / "pytest-basetemp"
        basetemp.mkdir(parents=True, exist_ok=True)

        returncode = _run_accepted_live_test(junit_path, basetemp)
        if returncode != 0:
            return _fail(
                "windows_interaction_live_test_failed",
                pytest_returncode=returncode,
            )

        try:
            evidence = parse_windows_interaction_proof_junit(junit_path)
        except WindowsInteractionProofEvidenceError:
            return _fail("windows_interaction_evidence_invalid")

        receipt = build_windows_interaction_proof_receipt(
            run_id=run_id,
            correlation_id=correlation_id,
            evidence=evidence,
            mongo_express_url=mongo_express_url,
        )

        try:
            verified_receipt, _integration = record_windows_interaction_proof_receipt(
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
