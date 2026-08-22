#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Shared cross-platform LKW OS interaction proof runner.

Thin OS launchers select ``--os-family``. Proof orchestration, evidence
validation, MongoDB stack preparation, and ProofReceipt recording live here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
_PROOF_RUNNER = "run-lkw-os-interaction-proof.py"
_RECEIPT_TASK = "PROOF-PORTABILITY-1C"
_EVIDENCE_SCHEMA = "lkw.os_interaction_proof_evidence.v1"
_DEFAULT_MONGO_EXPRESS_URL = "http://127.0.0.1:8086"
_PYTEST_TIMEOUT_SECONDS = 180
_INTAKE_ENDPOINT = "/v1/interactions/intake"
_INTERACTION_SURFACE = "lab_json"
_INTERACTION_CHANNEL = "lab"
_CLIENT_RUNTIME = "python"
_PROP_PREFIX = "os_interaction"
_MONGODB_STACK_MANAGED = "managed"
_MONGODB_STACK_EXTERNAL = "external"
_MONGODB_STACK_CHOICES = (_MONGODB_STACK_MANAGED, _MONGODB_STACK_EXTERNAL)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from lkw_proof_compose_lifecycle import (  # noqa: E402
    finalize_exit_code_with_teardown,
    run_terminal_compose_teardown,
    teardown_known_stack,
)
_APP_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_DIR.parent.parent
_DOCKER_DIR = _APP_DIR / "docker"
_BASE_COMPOSE = _DOCKER_DIR / "docker-compose.yml"
_MONGODB_COMPOSE = _DOCKER_DIR / "docker-compose.mongodb.yml"
_COMPOSE_PROJECT = "lkw-os-interaction-proof"
_OS_INTERACTION_STACK_ID = _COMPOSE_PROJECT


@dataclass(frozen=True, slots=True)
class OsInteractionProofContract:
    os_family: str
    proof_kind: str
    adapter_id: str
    source: str
    wrapper_runtime: str
    adapter_script: str
    live_test_node: str
    expected_testcase_name: str
    run_id_prefix: str


OS_PROOF_CONTRACTS: Mapping[str, OsInteractionProofContract] = {
    "windows": OsInteractionProofContract(
        os_family="windows",
        proof_kind="platform_windows_interaction",
        adapter_id="lkw.windows_powershell",
        source="windows_powershell",
        wrapper_runtime="windows_powershell",
        adapter_script="invoke-lkw-interaction.ps1",
        live_test_node=(
            "applications/local_workspace_application/tests/interactions/"
            "test_windows_powershell_interaction_live.py::"
            "test_windows_powershell_adapter_executes_real_lkw_interactions"
        ),
        expected_testcase_name=(
            "test_windows_powershell_adapter_executes_real_lkw_interactions"
        ),
        run_id_prefix="lkw-windows-interaction",
    ),
    "linux": OsInteractionProofContract(
        os_family="linux",
        proof_kind="platform_linux_interaction",
        adapter_id="lkw.linux_shell",
        source="linux_shell",
        wrapper_runtime="posix_sh",
        adapter_script="invoke-lkw-interaction-linux.sh",
        live_test_node=(
            "applications/local_workspace_application/tests/interactions/"
            "test_linux_shell_interaction_live.py::"
            "test_linux_shell_adapter_executes_real_lkw_interactions"
        ),
        expected_testcase_name=(
            "test_linux_shell_adapter_executes_real_lkw_interactions"
        ),
        run_id_prefix="lkw-linux-interaction",
    ),
    "macos": OsInteractionProofContract(
        os_family="macos",
        proof_kind="platform_macos_interaction",
        adapter_id="lkw.macos_shell",
        source="macos_shell",
        wrapper_runtime="posix_sh",
        adapter_script="invoke-lkw-interaction-macos.sh",
        live_test_node=(
            "applications/local_workspace_application/tests/interactions/"
            "test_macos_shell_interaction_live.py::"
            "test_macos_shell_adapter_executes_real_lkw_interactions"
        ),
        expected_testcase_name=(
            "test_macos_shell_adapter_executes_real_lkw_interactions"
        ),
        run_id_prefix="lkw-macos-interaction",
    ),
}

_REQUIRED_TRUE_KEYS = (
    f"{_PROP_PREFIX}.hosted_ready",
    f"{_PROP_PREFIX}.adapter_invoked",
    f"{_PROP_PREFIX}.index_executed",
    f"{_PROP_PREFIX}.search_executed",
    f"{_PROP_PREFIX}.task_ids_distinct",
    f"{_PROP_PREFIX}.run_ids_distinct",
    f"{_PROP_PREFIX}.graceful_stop",
    f"{_PROP_PREFIX}.cleanup_verified",
)


class OSInteractionProofEvidenceError(ValueError):
    """Raised when JUnit OS interaction proof evidence is missing or invalid."""


@dataclass(frozen=True, slots=True)
class OSInteractionProofEvidence:
    """Validated OS interaction proof evidence from the accepted live test."""

    schema_version: str
    os_family: str
    os_version: str
    architecture: str
    client_runtime: str
    wrapper_runtime: str
    adapter_id: str
    source: str
    transport: str
    intake_endpoint: str
    interaction_surface: str
    interaction_channel: str
    hosted_ready: bool
    adapter_invoked: bool
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


def detect_runtime_os_family(system_name: str | None = None) -> str:
    name = (system_name if system_name is not None else platform.system()).strip()
    mapping = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}
    detected = mapping.get(name)
    if detected is None:
        raise OSInteractionProofEvidenceError("unsupported_operating_system")
    return detected


def resolve_os_proof_contract(os_family: str) -> OsInteractionProofContract:
    contract = OS_PROOF_CONTRACTS.get(os_family.strip())
    if contract is None:
        raise OSInteractionProofEvidenceError("unsupported_os_family")
    return contract


def validate_runtime_os_matches(
    declared_os_family: str,
    *,
    runtime_os_family: str | None = None,
) -> str:
    actual = (
        runtime_os_family
        if runtime_os_family is not None
        else detect_runtime_os_family()
    )
    if actual != declared_os_family.strip():
        raise OSInteractionProofEvidenceError("runtime_os_mismatch")
    return actual


def build_os_interaction_proof_id(*, proof_kind: str, run_id: str) -> str:
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must not be blank")
    return f"{_APPLICATION_ID}:{proof_kind}:{normalized_run_id}"


def build_os_interaction_proof_receipt(
    *,
    contract: OsInteractionProofContract,
    run_id: str,
    correlation_id: str,
    evidence: OSInteractionProofEvidence,
    mongo_express_url: str,
) -> ProofReceipt:
    """Build a structured ProofReceipt from validated OS interaction evidence."""
    if evidence.os_family != contract.os_family:
        raise OSInteractionProofEvidenceError("invalid_os_family")
    if evidence.adapter_id != contract.adapter_id:
        raise OSInteractionProofEvidenceError("invalid_adapter_id")
    if evidence.source != contract.source:
        raise OSInteractionProofEvidenceError("invalid_source")
    if evidence.wrapper_runtime != contract.wrapper_runtime:
        raise OSInteractionProofEvidenceError("invalid_wrapper_runtime")

    domain_evidence: dict[str, Any] = {
        "os_family": evidence.os_family,
        "os_version": evidence.os_version,
        "architecture": evidence.architecture,
        "client_runtime": evidence.client_runtime,
        "wrapper_runtime": evidence.wrapper_runtime,
        "hosted_ready": evidence.hosted_ready,
        "adapter_invoked": evidence.adapter_invoked,
        "adapter_id": evidence.adapter_id,
        "source": evidence.source,
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
    }
    # Windows semantic compatibility: retain powershell_runtime for reviewers.
    if contract.os_family == "windows":
        domain_evidence["powershell_runtime"] = "Windows PowerShell"

    execution_profile = _execution_profile_from_env()
    domain_evidence.update(execution_profile)

    provider_evidence: dict[str, Any] = {
        "os_family": evidence.os_family,
        "os_version": evidence.os_version,
        "architecture": evidence.architecture,
        "os_adapter": evidence.adapter_id,
        "client_runtime": evidence.client_runtime,
        "wrapper_runtime": evidence.wrapper_runtime,
        "source": evidence.source,
        "transport": evidence.transport,
        "interaction_surface": evidence.interaction_surface,
        "interaction_channel": evidence.interaction_channel,
        "intake_endpoint": evidence.intake_endpoint,
        "intake_service": "InteractionIntakeService",
        "execution_boundary": "LocalWorkspaceTaskExecutor",
        "orchestrator": "NexusLoop",
        "hosted_entrypoint": "python_-m_local_workspace_application.hosting",
        "evidence_source": "pytest_junit_properties",
        "selected_live_tests": 1,
        "receipt_document_store_provider": "mongodb",
    }
    provider_evidence.update(execution_profile)

    return ProofReceipt(
        proof_id=build_os_interaction_proof_id(
            proof_kind=contract.proof_kind, run_id=run_id
        ),
        proof_kind=contract.proof_kind,
        application_id=_APPLICATION_ID,
        result=ProofReceiptResult.PASS,
        run_id=run_id,
        correlation_id=correlation_id,
        task_id=None,
        provider_evidence=provider_evidence,
        domain_evidence=domain_evidence,
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
            "source_test": contract.live_test_node,
            "adapter_script": contract.adapter_script,
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


def _execution_profile_from_env() -> dict[str, str]:
    """Optional execution-profile fields for container certification receipts."""
    profile: dict[str, str] = {}
    for key, env_name in (
        ("execution_environment", "LKW_EXECUTION_ENVIRONMENT"),
        ("container_runtime", "LKW_CONTAINER_RUNTIME"),
        ("certification_profile", "LKW_CERTIFICATION_PROFILE"),
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            profile[key] = value
    return profile


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


def verify_mongodb_reachable_via_platform() -> None:
    """Fail closed when the configured MongoDB provider cannot be reached."""
    ensure_mongodb_env()
    uri = os.environ.get("INTERGRAX_MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("external_mongodb_uri_required")
    integration, store = resolve_mongodb_document_store()
    try:
        store.get("__lkw_certification_probe__", "reachability")
    except Exception as exc:  # noqa: BLE001 - fail closed with typed reason
        raise RuntimeError("external_mongodb_unreachable") from exc
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
        _ = integration


def prepare_external_mongodb(*, mongo_express_url: str) -> None:
    """Use outer-compose MongoDB; never invoke Docker."""
    _ = mongo_express_url
    if not os.environ.get("INTERGRAX_MONGODB_URI", "").strip():
        raise RuntimeError("external_mongodb_uri_required")
    ensure_mongodb_env()
    verify_mongodb_reachable_via_platform()


def record_os_interaction_proof_receipt(
    receipt: ProofReceipt,
) -> tuple[ProofReceipt, MongoDBDocumentStoreIntegration]:
    """Persist and verify an OS interaction proof receipt through the platform store."""
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
                raise OSInteractionProofEvidenceError(f"conflicting_property:{name}")
            properties[name] = value
    return properties


def _require_true(properties: dict[str, str], key: str) -> bool:
    value = properties.get(key)
    if value is None:
        raise OSInteractionProofEvidenceError(f"missing_property:{key}")
    if value != "true":
        raise OSInteractionProofEvidenceError(f"false_required_evidence:{key}")
    return True


def _require_text(properties: dict[str, str], key: str) -> str:
    value = properties.get(key)
    if value is None:
        raise OSInteractionProofEvidenceError(f"missing_property:{key}")
    normalized = value.strip()
    if not normalized:
        raise OSInteractionProofEvidenceError(f"blank_property:{key}")
    return normalized


def parse_os_interaction_proof_junit(
    junit_xml: str | Path,
    *,
    contract: OsInteractionProofContract,
) -> OSInteractionProofEvidence:
    """Parse and validate OS interaction proof evidence from legacy JUnit XML."""
    path = Path(junit_xml)
    if not path.is_file():
        raise OSInteractionProofEvidenceError("missing_junit_file")
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        raise OSInteractionProofEvidenceError("malformed_junit_xml") from exc

    testcases = root.findall(".//testcase")
    if len(testcases) != 1:
        raise OSInteractionProofEvidenceError(
            f"unexpected_testcase_count:{len(testcases)}"
        )

    testcase = testcases[0]
    name = testcase.attrib.get("name", "").strip()
    if not name:
        raise OSInteractionProofEvidenceError("missing_testcase_name")
    if name != contract.expected_testcase_name:
        raise OSInteractionProofEvidenceError(f"unexpected_testcase:{name}")
    if testcase.find("failure") is not None:
        raise OSInteractionProofEvidenceError(f"failed_testcase:{name}")
    if testcase.find("error") is not None:
        raise OSInteractionProofEvidenceError(f"errored_testcase:{name}")
    if testcase.find("skipped") is not None:
        raise OSInteractionProofEvidenceError(f"skipped_testcase:{name}")

    properties = _collect_properties(testcase)
    for key in _REQUIRED_TRUE_KEYS:
        _require_true(properties, key)

    os_family = _require_text(properties, f"{_PROP_PREFIX}.os_family")
    if os_family != contract.os_family:
        raise OSInteractionProofEvidenceError("invalid_os_family")

    adapter_id = _require_text(properties, f"{_PROP_PREFIX}.adapter_id")
    if adapter_id != contract.adapter_id:
        raise OSInteractionProofEvidenceError("invalid_adapter_id")

    source = _require_text(properties, f"{_PROP_PREFIX}.source")
    if source != contract.source:
        raise OSInteractionProofEvidenceError("invalid_source")

    wrapper_runtime = _require_text(properties, f"{_PROP_PREFIX}.wrapper_runtime")
    if wrapper_runtime != contract.wrapper_runtime:
        raise OSInteractionProofEvidenceError("invalid_wrapper_runtime")

    client_runtime = _require_text(properties, f"{_PROP_PREFIX}.client_runtime")
    if client_runtime != _CLIENT_RUNTIME:
        raise OSInteractionProofEvidenceError("invalid_client_runtime")

    transport = _require_text(properties, f"{_PROP_PREFIX}.transport")
    if transport != "http":
        raise OSInteractionProofEvidenceError("invalid_transport")

    intake_endpoint = _require_text(properties, f"{_PROP_PREFIX}.intake_endpoint")
    if intake_endpoint != _INTAKE_ENDPOINT:
        raise OSInteractionProofEvidenceError("invalid_endpoint")

    interaction_surface = _require_text(
        properties, f"{_PROP_PREFIX}.interaction_surface"
    )
    if interaction_surface != _INTERACTION_SURFACE:
        raise OSInteractionProofEvidenceError("invalid_interaction_surface")

    interaction_channel = _require_text(
        properties, f"{_PROP_PREFIX}.interaction_channel"
    )
    if interaction_channel != _INTERACTION_CHANNEL:
        raise OSInteractionProofEvidenceError("invalid_interaction_channel")

    index_state = _require_text(properties, f"{_PROP_PREFIX}.index_state")
    if index_state != "completed":
        raise OSInteractionProofEvidenceError("invalid_index_state")

    search_state = _require_text(properties, f"{_PROP_PREFIX}.search_state")
    if search_state != "completed":
        raise OSInteractionProofEvidenceError("invalid_search_state")

    os_version = _require_text(properties, f"{_PROP_PREFIX}.os_version")
    architecture = _require_text(properties, f"{_PROP_PREFIX}.architecture")
    index_task_id = _require_text(properties, f"{_PROP_PREFIX}.index_task_id")
    index_run_id = _require_text(properties, f"{_PROP_PREFIX}.index_run_id")
    search_task_id = _require_text(properties, f"{_PROP_PREFIX}.search_task_id")
    search_run_id = _require_text(properties, f"{_PROP_PREFIX}.search_run_id")

    if index_task_id == search_task_id:
        raise OSInteractionProofEvidenceError("same_task_ids")
    if index_run_id == search_run_id:
        raise OSInteractionProofEvidenceError("same_run_ids")

    return OSInteractionProofEvidence(
        schema_version=_EVIDENCE_SCHEMA,
        os_family=os_family,
        os_version=os_version,
        architecture=architecture,
        client_runtime=client_runtime,
        wrapper_runtime=wrapper_runtime,
        adapter_id=adapter_id,
        source=source,
        transport=transport,
        intake_endpoint=intake_endpoint,
        interaction_surface=interaction_surface,
        interaction_channel=interaction_channel,
        hosted_ready=True,
        adapter_invoked=True,
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


def _compose_args(compose_files: Sequence[Path]) -> list[str]:
    args = ["docker", "compose", "-p", _COMPOSE_PROJECT]
    for path in compose_files:
        args.extend(["-f", str(path)])
    return args


def _run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        shell=False,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _compose_ps_json(
    compose_files: Sequence[Path],
    service: str,
    *,
    cwd: Path,
) -> list[dict[str, Any]]:
    completed = _run_command(
        [*_compose_args(compose_files), "ps", "--format", "json", service],
        cwd=cwd,
        timeout=60,
    )
    if completed.returncode != 0:
        return []
    text = completed.stdout.strip()
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            rows.extend(item for item in parsed if isinstance(item, dict))
        elif isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _wait_for_compose_health(
    compose_files: Sequence[Path],
    service: str,
    *,
    cwd: Path,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for row in _compose_ps_json(compose_files, service, cwd=cwd):
            health = str(row.get("Health") or "").strip().lower()
            if health == "healthy":
                return
        time.sleep(2)
    raise RuntimeError("mongodb_health_timeout")


def _wait_for_http_reachable(url: str, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if 200 <= int(response.status) < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise RuntimeError("mongo_express_unreachable")


def prepare_mongodb_stack(
    *,
    mongo_express_url: str,
    compose_ownership_state: list[bool] | None = None,
) -> None:
    """Start MongoDB + Mongo Express for receipt recording (shared across OS launchers)."""
    if shutil.which("docker") is None:
        raise RuntimeError("docker_not_available")
    compose_files = [_BASE_COMPOSE, _MONGODB_COMPOSE]
    config = _run_command(
        [*_compose_args(compose_files), "config"],
        cwd=_REPO_ROOT,
        timeout=120,
    )
    if config.returncode != 0:
        raise RuntimeError("compose_overlay_invalid")
    if compose_ownership_state is not None:
        compose_ownership_state.append(True)
    up = _run_command(
        [
            *_compose_args(compose_files),
            "up",
            "-d",
            "lkw-mongodb",
            "lkw-mongo-express",
        ],
        cwd=_REPO_ROOT,
        timeout=300,
    )
    if up.returncode != 0:
        raise RuntimeError("compose_up_failed")
    _wait_for_compose_health(
        compose_files, "lkw-mongodb", cwd=_REPO_ROOT, timeout_seconds=180
    )
    _wait_for_http_reachable(mongo_express_url, timeout_seconds=120)
    ensure_mongodb_env()


def prepare_mongodb(
    *,
    stack: str,
    mongo_express_url: str,
    compose_ownership_state: list[bool] | None = None,
) -> None:
    """Prepare MongoDB according to managed/external stack ownership."""
    normalized = stack.strip().lower()
    if normalized == _MONGODB_STACK_EXTERNAL:
        prepare_external_mongodb(mongo_express_url=mongo_express_url)
        return
    if normalized != _MONGODB_STACK_MANAGED:
        raise RuntimeError("invalid_mongodb_stack")
    prepare_mongodb_stack(
        mongo_express_url=mongo_express_url,
        compose_ownership_state=compose_ownership_state,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the accepted LKW OS interaction live test and record a "
            "platform OS Interaction ProofReceipt."
        ),
    )
    parser.add_argument(
        "--os-family",
        required=True,
        choices=sorted(OS_PROOF_CONTRACTS.keys()),
        help="Declared OS family; must match the actual runtime OS.",
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
    parser.add_argument(
        "--mongodb-stack",
        default=_MONGODB_STACK_MANAGED,
        choices=list(_MONGODB_STACK_CHOICES),
        help=(
            "managed: start MongoDB Compose overlay (default). "
            "external: require INTERGRAX_MONGODB_URI; never invoke Docker."
        ),
    )
    return parser.parse_args(argv)


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
    contract: OsInteractionProofContract,
    evidence: OSInteractionProofEvidence,
    verified_receipt: ProofReceipt,
    mongo_express_url: str,
    correlation_id: str,
) -> None:
    print("proof_result=PASS")
    print(f"proof_kind={contract.proof_kind}")
    print("proof_tests_passed=1")
    print(f"os_family={evidence.os_family}")
    print(f"os_version={evidence.os_version}")
    print(f"architecture={evidence.architecture}")
    print("adapter_invoked=true")
    print(f"adapter_id={evidence.adapter_id}")
    print(f"source={evidence.source}")
    print(f"client_runtime={evidence.client_runtime}")
    print(f"wrapper_runtime={evidence.wrapper_runtime}")
    if contract.os_family == "windows":
        print("powershell_runtime=Windows PowerShell")
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


def _run_accepted_live_test(
    *,
    contract: OsInteractionProofContract,
    junit_path: Path,
    basetemp: Path,
) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        contract.live_test_node,
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
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return -1
    return int(completed.returncode)


def teardown_owned_compose_stack() -> None:
    teardown_known_stack(_OS_INTERACTION_STACK_ID, cwd=_REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        contract = resolve_os_proof_contract(args.os_family)
        validate_runtime_os_matches(contract.os_family)
    except OSInteractionProofEvidenceError as exc:
        return _fail(str(exc))

    if contract.os_family == "windows" and not shutil.which("powershell.exe"):
        return _fail("windows_powershell_unavailable")

    run_id = args.run_id.strip() or f"{contract.run_id_prefix}-{uuid.uuid4().hex[:12]}"
    correlation_id = args.correlation_id.strip() or run_id
    mongo_express_url = args.mongo_express.strip() or _DEFAULT_MONGO_EXPRESS_URL
    compose_ownership_entered = False
    compose_ownership_state: list[bool] = []
    functional_pass = False
    exit_code = 1

    try:
        try:
            prepare_mongodb(
                stack=str(args.mongodb_stack),
                mongo_express_url=mongo_express_url,
                compose_ownership_state=compose_ownership_state,
            )
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            compose_ownership_entered = bool(compose_ownership_state)
            exit_code = _fail(
                "mongodb_stack_prepare_failed",
                stack_error=type(exc).__name__,
                mongodb_stack=str(args.mongodb_stack),
                stack_detail=str(exc),
            )
            return exit_code
        compose_ownership_entered = bool(compose_ownership_state)

        with tempfile.TemporaryDirectory(prefix="lkw-os-interaction-proof-") as temp_dir:
            temp_root = Path(temp_dir)
            junit_path = temp_root / "os-interaction-proof-junit.xml"
            basetemp = temp_root / "pytest-basetemp"
            basetemp.mkdir(parents=True, exist_ok=True)

            returncode = _run_accepted_live_test(
                contract=contract,
                junit_path=junit_path,
                basetemp=basetemp,
            )
            if returncode != 0:
                exit_code = _fail(
                    "os_interaction_live_test_failed",
                    pytest_returncode=returncode,
                    os_family=contract.os_family,
                )
                return exit_code

            try:
                evidence = parse_os_interaction_proof_junit(junit_path, contract=contract)
            except OSInteractionProofEvidenceError:
                exit_code = _fail("os_interaction_evidence_invalid")
                return exit_code

            receipt = build_os_interaction_proof_receipt(
                contract=contract,
                run_id=run_id,
                correlation_id=correlation_id,
                evidence=evidence,
                mongo_express_url=mongo_express_url,
            )

            try:
                verified_receipt, _integration = record_os_interaction_proof_receipt(
                    receipt
                )
            except (
                ProofReceiptVerificationError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                exit_code = _fail_receipt_recording(exc)
                return exit_code

            _print_pass_output(
                contract=contract,
                evidence=evidence,
                verified_receipt=verified_receipt,
                mongo_express_url=mongo_express_url,
                correlation_id=correlation_id,
            )
            functional_pass = True
            exit_code = 0
    finally:
        compose_ownership_entered = compose_ownership_entered or bool(compose_ownership_state)
        teardown_outcome = run_terminal_compose_teardown(
            compose_ownership_entered=compose_ownership_entered,
            teardown_fn=teardown_owned_compose_stack,
        )
        if functional_pass and teardown_outcome.result == "FAIL":
            exit_code = _fail("proof_teardown_failed")
        else:
            exit_code = finalize_exit_code_with_teardown(
                functional_pass=functional_pass,
                functional_exit_code=exit_code,
                teardown_outcome=teardown_outcome,
            )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
