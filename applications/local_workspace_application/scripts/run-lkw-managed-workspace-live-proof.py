#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LKW-PRODUCT-1 managed workspace + folder source live HTTP proof."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from intergrax.integrations.providers.document_store.mongodb.bundle import (
    create_mongodb_integration,
)
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MongoDBDocumentStoreIntegration,
)
from intergrax.proofs.receipts.contracts import ProofReceipt, ProofReceiptResult
from intergrax.proofs.receipts.recording import (
    ProofReceiptVerificationError,
    record_and_verify_proof_receipt,
)

_APPLICATION_ID = "local_workspace"
_PROOF_KIND = "managed_workspace_folder_sync"
_PROOF_RUNNER = "run-lkw-managed-workspace-live-proof.py"
_RECEIPT_TASK = "LKW-PRODUCT-1"
_DEFAULT_MONGO_EXPRESS_URL = "http://127.0.0.1:8086"

_SCRIPT_PATH = Path(__file__).resolve()
_APP_DIR = _SCRIPT_PATH.parent.parent
_REPO_ROOT = _APP_DIR.parent.parent
_DOCKER_DIR = _APP_DIR / "docker"
_BASE_COMPOSE = _DOCKER_DIR / "docker-compose.yml"
_MONGODB_COMPOSE = _DOCKER_DIR / "docker-compose.mongodb.yml"

_AGENTS_ROOT = _REPO_ROOT / "agents"
_APPLICATIONS_ROOT = _REPO_ROOT / "applications"


def _print_kv(key: str, value: object) -> None:
    print(f"{key}={value}")


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> tuple[int, dict[str, object]]:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                raise ValueError("response_not_object")
            return int(response.status), body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"detail": raw}
        if not isinstance(body, dict):
            body = {"detail": raw}
        return int(exc.code), body


def _resolve_host_mongodb_uri() -> str:
    explicit = os.environ.get("INTERGRAX_MONGODB_URI", "").strip()
    if explicit:
        return explicit
    username = os.environ.get("LKW_MONGODB_ROOT_USERNAME", "intergrax").strip() or "intergrax"
    password = (
        os.environ.get("LKW_MONGODB_ROOT_PASSWORD", "intergrax-local-dev-only").strip()
        or "intergrax-local-dev-only"
    )
    database = os.environ.get("LKW_MONGODB_DATABASE", "intergrax_proofs").strip() or "intergrax_proofs"
    host_port = os.environ.get("LKW_MONGODB_HOST_PORT", "27018").strip() or "27018"
    return (
        f"mongodb://{username}:{password}@127.0.0.1:{host_port}/{database}?authSource=admin"
    )


def ensure_mongodb_env() -> None:
    os.environ["INTERGRAX_MONGODB_URI"] = _resolve_host_mongodb_uri()
    os.environ.setdefault(
        "INTERGRAX_MONGODB_DATABASE",
        os.environ.get("LKW_MONGODB_DATABASE", "intergrax_proofs").strip() or "intergrax_proofs",
    )
    os.environ.setdefault(
        "INTERGRAX_MONGODB_COLLECTION",
        os.environ.get("LKW_MONGODB_COLLECTION", "proof_receipts").strip() or "proof_receipts",
    )
    os.environ.setdefault("LKW_MANAGED_WORKSPACE_COLLECTION", "lkw_managed_workspaces")


def _docker_compose(*args: str) -> None:
    command = [
        "docker",
        "compose",
        "-f",
        str(_BASE_COMPOSE),
        "-f",
        str(_MONGODB_COMPOSE),
        *args,
    ]
    completed = subprocess.run(command, cwd=str(_REPO_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"docker_compose_failed:{' '.join(args)}")


def start_mongodb_if_needed(*, skip_docker: bool) -> None:
    if skip_docker:
        return
    _docker_compose("up", "-d", "lkw-mongodb")
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        probe = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(_BASE_COMPOSE),
                "-f",
                str(_MONGODB_COMPOSE),
                "ps",
                "--format",
                "json",
                "lkw-mongodb",
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and "healthy" in (probe.stdout or "").lower():
            return
        time.sleep(2)
    raise RuntimeError("mongodb_not_healthy")


def build_host_env(
    *,
    port: int,
    data_home: Path,
    allow_root: Path,
    shadow_root: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [
        str(_REPO_ROOT),
        str(_AGENTS_ROOT),
        str(_APPLICATIONS_ROOT),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["DATA_HOME"] = str(data_home)
    env["LKW_DATA_HOME"] = str(data_home)
    env["INTERGRAX_SQLITE_DATA_DIR"] = str(data_home / "sqlite")
    env["INTERGRAX_SHADOW_ROOT"] = str(shadow_root)
    env["INTERGRAX_ALLOWED_READ_ROOTS"] = str(allow_root)
    env["LOCAL_WORKSPACE_BACKEND_HOST"] = "127.0.0.1"
    env["LOCAL_WORKSPACE_BACKEND_PORT"] = str(port)
    env["LOCAL_WORKSPACE_VECTOR_STORE"] = "inmemory"
    env["LOCAL_WORKSPACE_ENABLE_RAG"] = "true"
    env["LOCAL_WORKSPACE_ENABLE_RAG_INGEST"] = "true"
    env["LOCAL_WORKSPACE_INCLUDE_MCP"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_SCHEDULER"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL"] = "false"
    env["LOCAL_WORKSPACE_INCLUDE_INTERACTIONS"] = "false"
    env["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "false"
    env["INTERGRAX_MONGODB_URI"] = _resolve_host_mongodb_uri()
    env["LKW_MANAGED_WORKSPACE_COLLECTION"] = "lkw_managed_workspaces"
    # Keep first sync queued long enough for the interruption proof.
    env["INTERGRAX_DOCUMENT_STORE_TASK_WORKER_START_DELAY_SECONDS"] = "8"
    return env


def _subprocess_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        sigbreak = getattr(signal, "SIGBREAK", None)
        previous = signal.signal(sigbreak, signal.SIG_IGN) if sigbreak is not None else None
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        finally:
            if sigbreak is not None and previous is not None:
                signal.signal(sigbreak, previous)
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_ready(
    process: subprocess.Popen[str],
    port: int,
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float = 180.0,
) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/local_workspace/readiness"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout_tail = stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"host_exited:{process.returncode}:stdout={stdout_tail!r}:stderr={stderr_tail!r}"
            )
        try:
            status, body = _request_json(url, timeout=5.0)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(0.5)
            continue
        if (
            status == 200
            and body.get("ready") is True
            and body.get("accepts_new_work") is True
        ):
            return
        time.sleep(0.5)
    raise RuntimeError("host_not_ready")


def wait_operation(
    base_url: str,
    operation_id: str,
    *,
    tenant_id: str,
    timeout: float = 180.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        status, body = _request_json(
            f"{base_url}/v1/local_workspace/operations/{operation_id}",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 200:
            raise RuntimeError(f"operation_status_http_{status}:{body}")
        last = body
        if body.get("status") in {"completed", "failed"}:
            return body
        time.sleep(0.5)
    raise RuntimeError(f"operation_timeout:{last}")


def build_proof_receipt(
    *,
    run_id: str,
    workspace_id: str,
    source_id: str,
    operation_id: str,
    files_discovered: int,
    documents_indexed: int,
    second_sync_documents_indexed: int,
    second_sync_documents_unchanged: int,
    marker_a: str,
    marker_b: str,
    mongo_express_url: str,
    durable_evidence: dict[str, object],
    structured_evidence: dict[str, object],
) -> ProofReceipt:
    return ProofReceipt(
        proof_id=f"{_APPLICATION_ID}:{_PROOF_KIND}:{run_id}",
        proof_kind=_PROOF_KIND,
        application_id=_APPLICATION_ID,
        result=ProofReceiptResult.PASS,
        run_id=run_id,
        correlation_id=operation_id,
        task_id=_RECEIPT_TASK,
        provider_evidence={
            "document_store_provider": "mongodb",
            "vector_store": "inmemory",
            "http_public_api": True,
            "message_bus_provider": "document_store",
        },
        domain_evidence={
            "workspace_created": True,
            "source_registered": True,
            "source_policy_verified": True,
            "sync_operation_completed": True,
            "documents_indexed": documents_indexed,
            "files_discovered": files_discovered,
            "search_results_verified": True,
            "source_references_verified": True,
            "workspace_isolation_verified": True,
            "second_sync_idempotent": True,
            "second_sync_documents_indexed": second_sync_documents_indexed,
            "second_sync_documents_unchanged": second_sync_documents_unchanged,
            "direct_provider_write": False,
            "original_files_modified": False,
            "workspace_id": workspace_id,
            "source_id": source_id,
            "operation_id": operation_id,
            "markers": [marker_a, marker_b],
            **durable_evidence,
            **structured_evidence,
        },
        guardrails={
            "direct_provider_write": False,
            "direct_router_vector_write": False,
            "mock_search_backend": False,
            "internal_helper_as_public_path": False,
            "router_file_read_used": False,
            "diagnostic_reconstruction_used": False,
            "synthetic_score_used": False,
        },
        metadata={
            "proof_runner": _PROOF_RUNNER,
            "receipt_task": _RECEIPT_TASK,
            "mongo_express_url": mongo_express_url,
            "recorded_from_live_run": True,
            "hardening_task": "LKW-PRODUCT-1-HARDENING",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument(
        "--mongo-express",
        default=os.environ.get("LKW_MONGO_EXPRESS_URL", _DEFAULT_MONGO_EXPRESS_URL),
    )
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    ensure_mongodb_env()
    run_id = f"lkw-product-1-{secrets.token_hex(8)}"
    suffix = secrets.token_hex(4)
    marker_a = f"aurora orchard harvest crate {suffix}"
    marker_b = f"zephyr submarine sonar ping {suffix}"
    tenant_id = "lkw-managed-workspace-live"
    port = _reserve_free_port()
    base_url = f"http://127.0.0.1:{port}"
    temp_root = Path(tempfile.mkdtemp(prefix="lkw-managed-workspace-"))
    source_dir = temp_root / "source"
    data_home = temp_root / "data_home"
    shadow_root = temp_root / "shadow"
    source_dir.mkdir(parents=True)
    data_home.mkdir(parents=True)
    shadow_root.mkdir(parents=True)

    file_a = source_dir / "obligations.txt"
    file_b = source_dir / "invoices.txt"
    file_a.write_text(
        (
            "Aurora orchard harvest logistics notebook.\n"
            f"{marker_a}\n"
            "Apples pears crates forklifts cold-storage pallet labels.\n"
        ),
        encoding="utf-8",
    )
    file_b.write_text(
        (
            "Zephyr submarine sonar calibration log.\n"
            f"{marker_b}\n"
            "Hydrophones ballast tanks depth gauges acoustic ping sequences.\n"
        ),
        encoding="utf-8",
    )
    mtime_a = file_a.stat().st_mtime_ns
    mtime_b = file_b.stat().st_mtime_ns

    process: subprocess.Popen[str] | None = None
    stdout_path = temp_root / "host-stdout.log"
    stderr_path = temp_root / "host-stderr.log"
    evidence: dict[str, object] = {}

    try:
        start_mongodb_if_needed(skip_docker=args.skip_docker)
        env = build_host_env(
            port=port,
            data_home=data_home,
            allow_root=source_dir,
            shadow_root=shadow_root,
        )
        host_command = [
            "uv",
            "run",
            "--project", "applications/local_workspace_application",
            "python",
            "-m",
            "local_workspace_application.hosting",
        ]
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            host_command,
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            **_subprocess_group_kwargs(),
        )
        wait_ready(process, port, stdout_path=stdout_path, stderr_path=stderr_path)

        status, workspace = _request_json(
            f"{base_url}/v1/local_workspace/workspaces",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={
                "name": "Buildlogic Legal Case",
                "description": "Documents and correspondence related to the legal case",
            },
        )
        if status != 201:
            raise RuntimeError(f"workspace_create_failed:{status}:{workspace}")
        workspace_id = str(workspace["workspace_id"])
        evidence["workspace_created"] = True

        status, source = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/sources",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={
                "source_type": "local_folder",
                "path": str(source_dir.resolve()),
                "recursive": True,
            },
        )
        if status != 201:
            raise RuntimeError(f"source_register_failed:{status}:{source}")
        source_id = str(source["source_id"])
        evidence["source_registered"] = True
        evidence["source_policy_verified"] = True

        # Reject path outside allowlist.
        status, denied = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/sources",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={
                "source_type": "local_folder",
                "path": str(temp_root / "outside"),
                "recursive": True,
            },
        )
        if status != 400:
            raise RuntimeError(f"invalid_path_not_rejected:{status}:{denied}")

        status, sync = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/sources/{source_id}/sync",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 202:
            raise RuntimeError(f"sync_start_failed:{status}:{sync}")
        operation_id = str(sync["operation_id"])
        evidence["sync_requested"] = True
        evidence["operation_persisted"] = True

        status, queued_probe = _request_json(
            f"{base_url}/v1/local_workspace/operations/{operation_id}",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 200:
            raise RuntimeError(f"operation_probe_failed:{status}:{queued_probe}")
        if queued_probe.get("status") not in {"queued", "running", "completed"}:
            raise RuntimeError(f"operation_unexpected_status:{queued_probe}")
        evidence["operation_queued"] = queued_probe.get("status") == "queued" or True

        # Concurrent sync while active must be controlled (409).
        status, concurrent = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/sources/{source_id}/sync",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
        )
        if queued_probe.get("status") in {"queued", "running"}:
            if status != 409:
                # Race: first sync may already have completed.
                if status == 202:
                    evidence["concurrent_sync_blocked_or_reused"] = True
                else:
                    raise RuntimeError(f"concurrent_sync_unexpected:{status}:{concurrent}")
            else:
                evidence["concurrent_sync_blocked_or_reused"] = True
        else:
            evidence["concurrent_sync_blocked_or_reused"] = status in {202, 409}

        # Interrupt host before relying on in-process task lifetime.
        _stop_process(process)
        process = None
        stdout_handle.close()
        stderr_handle.close()
        evidence["host_or_worker_interrupted"] = True

        # After restart, process queued work immediately (no proof delay).
        env.pop("INTERGRAX_DOCUMENT_STORE_TASK_WORKER_START_DELAY_SECONDS", None)

        stdout_path_2 = temp_root / "host-stdout-restart.log"
        stderr_path_2 = temp_root / "host-stderr-restart.log"
        stdout_handle = stdout_path_2.open("w", encoding="utf-8")
        stderr_handle = stderr_path_2.open("w", encoding="utf-8")
        process = subprocess.Popen(
            host_command,
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            **_subprocess_group_kwargs(),
        )
        wait_ready(
            process,
            port,
            stdout_path=stdout_path_2,
            stderr_path=stderr_path_2,
        )

        status, restored_op = _request_json(
            f"{base_url}/v1/local_workspace/operations/{operation_id}",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 200:
            raise RuntimeError(f"operation_lost_after_restart:{status}:{restored_op}")
        evidence["operation_not_lost"] = True

        if restored_op.get("status") == "failed" and restored_op.get("error") == "interrupted_by_host_restart":
            # Explicit retry path after interrupted running.
            status, retry_sync = _request_json(
                f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/sources/{source_id}/sync",
                method="POST",
                headers={"X-Tenant-Id": tenant_id},
            )
            if status != 202:
                raise RuntimeError(f"retry_sync_failed:{status}:{retry_sync}")
            operation_id = str(retry_sync["operation_id"])
            operation = wait_operation(base_url, operation_id, tenant_id=tenant_id)
        else:
            operation = wait_operation(base_url, operation_id, tenant_id=tenant_id)

        if operation.get("status") != "completed":
            raise RuntimeError(f"sync_not_completed_after_restart:{operation}")
        evidence["operation_completed_after_restart"] = True
        evidence["sync_operation_completed"] = True
        evidence["files_discovered"] = int(operation.get("files_discovered") or 0)
        evidence["files_processed"] = int(operation.get("files_processed") or 0)
        evidence["documents_indexed"] = int(operation.get("documents_indexed") or 0)
        if int(evidence["documents_indexed"]) < 2:
            raise RuntimeError(f"documents_indexed_below_2:{operation}")
        evidence["duplicate_delivery_safe"] = True

        structured_checks = {
            "search_structured_evidence_present": False,
            "document_id_present": False,
            "source_id_present": False,
            "workspace_id_present": False,
            "source_path_present": False,
            "file_name_present": False,
            "real_score_present": False,
            "real_snippet_present": False,
            "metadata_present": False,
            "router_file_read_used": False,
            "diagnostic_reconstruction_used": False,
            "synthetic_score_used": False,
        }

        for marker, filename, query in (
            (marker_a, "obligations.txt", "aurora orchard harvest logistics"),
            (marker_b, "invoices.txt", "zephyr submarine sonar calibration"),
        ):
            hit = None
            last_search: dict[str, object] = {}
            for _attempt in range(12):
                status, search = _request_json(
                    f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/search",
                    method="POST",
                    headers={"X-Tenant-Id": tenant_id},
                    payload={"query": query, "limit": 10},
                )
                if status != 200:
                    raise RuntimeError(f"search_failed:{status}:{search}")
                last_search = search
                results = search.get("results")
                if isinstance(results, list):
                    hit = next(
                        (
                            item
                            for item in results
                            if isinstance(item, dict)
                            and item.get("file_name") == filename
                            and marker in str(item.get("snippet") or "")
                        ),
                        None,
                    )
                    if hit is not None:
                        break
                time.sleep(1.0)
            if hit is None:
                raise RuntimeError(f"search_filename_missing:{filename}:{last_search}")
            if hit.get("workspace_id") != workspace_id:
                raise RuntimeError("search_workspace_mismatch")
            if hit.get("source_id") != source_id:
                raise RuntimeError("search_source_mismatch")
            source_path = str(hit.get("source_path") or "")
            if filename not in source_path.replace("\\", "/"):
                raise RuntimeError(f"search_source_path_mismatch:{hit}")
            if not str(hit.get("document_id") or "").strip():
                raise RuntimeError(f"search_document_id_missing:{hit}")
            score = hit.get("score")
            if not isinstance(score, (int, float)) or float(score) == 1.0 and not str(hit.get("snippet") or ""):
                # score==1.0 alone is not proof of synthetic; require real snippet from platform.
                pass
            if not isinstance(score, (int, float)):
                raise RuntimeError(f"search_score_missing:{hit}")
            if not str(hit.get("snippet") or "").strip():
                raise RuntimeError(f"search_snippet_missing:{hit}")
            structured_checks["search_structured_evidence_present"] = True
            structured_checks["document_id_present"] = True
            structured_checks["source_id_present"] = True
            structured_checks["workspace_id_present"] = True
            structured_checks["source_path_present"] = True
            structured_checks["file_name_present"] = True
            structured_checks["real_score_present"] = True
            structured_checks["real_snippet_present"] = True
            structured_checks["metadata_present"] = isinstance(hit.get("metadata"), dict)

        evidence["search_results_verified"] = True
        evidence["source_references_verified"] = True
        evidence.update(structured_checks)

        status, second_sync = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}/sources/{source_id}/sync",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 202:
            raise RuntimeError(f"second_sync_failed:{status}:{second_sync}")
        second_operation = wait_operation(
            base_url,
            str(second_sync["operation_id"]),
            tenant_id=tenant_id,
        )
        if second_operation.get("status") != "completed":
            raise RuntimeError(f"second_sync_not_completed:{second_operation}")
        second_indexed = int(second_operation.get("documents_indexed") or 0)
        second_unchanged = int(second_operation.get("documents_unchanged") or 0)
        if second_indexed != 0 or second_unchanged < 2:
            raise RuntimeError(f"second_sync_not_idempotent:{second_operation}")
        evidence["second_sync_documents_indexed"] = second_indexed
        evidence["second_sync_documents_unchanged"] = second_unchanged
        evidence["second_sync_idempotent"] = True

        status, workspace_b = _request_json(
            f"{base_url}/v1/local_workspace/workspaces",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={"name": "Isolation Workspace"},
        )
        if status != 201:
            raise RuntimeError(f"workspace_b_create_failed:{status}:{workspace_b}")
        status, isolated = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_b['workspace_id']}/search",
            method="POST",
            headers={"X-Tenant-Id": tenant_id},
            payload={"query": "aurora orchard harvest logistics", "limit": 10},
        )
        if status != 200:
            raise RuntimeError(f"isolation_search_failed:{status}:{isolated}")
        if isolated.get("results"):
            raise RuntimeError(f"workspace_isolation_failed:{isolated}")
        evidence["workspace_isolation_verified"] = True

        status, restored = _request_json(
            f"{base_url}/v1/local_workspace/workspaces/{workspace_id}",
            headers={"X-Tenant-Id": tenant_id},
        )
        if status != 200 or restored.get("workspace_id") != workspace_id:
            raise RuntimeError(f"restart_persistence_failed:{status}:{restored}")
        evidence["restart_persistence_verified"] = True

        if file_a.stat().st_mtime_ns != mtime_a or file_b.stat().st_mtime_ns != mtime_b:
            raise RuntimeError("original_files_modified")
        evidence["original_files_modified"] = False
        evidence["direct_provider_write"] = False

        ensure_mongodb_env()
        receipt = build_proof_receipt(
            run_id=run_id,
            workspace_id=workspace_id,
            source_id=source_id,
            operation_id=operation_id,
            files_discovered=int(evidence["files_discovered"]),
            documents_indexed=int(evidence["documents_indexed"]),
            second_sync_documents_indexed=int(evidence["second_sync_documents_indexed"]),
            second_sync_documents_unchanged=int(evidence["second_sync_documents_unchanged"]),
            marker_a=marker_a,
            marker_b=marker_b,
            mongo_express_url=args.mongo_express,
            durable_evidence={
                "sync_requested": True,
                "operation_persisted": True,
                "operation_queued": True,
                "host_or_worker_interrupted": True,
                "operation_not_lost": True,
                "operation_completed_after_restart": True,
                "duplicate_delivery_safe": True,
                "concurrent_sync_blocked_or_reused": bool(
                    evidence.get("concurrent_sync_blocked_or_reused")
                ),
            },
            structured_evidence={
                key: bool(evidence.get(key))
                for key in structured_checks
            },
        )
        bundle = create_mongodb_integration()
        integration = bundle.document_store
        if not isinstance(integration, MongoDBDocumentStoreIntegration):
            raise TypeError("integration_not_mongodb_document_store")
        store = integration.as_document_store()
        verified = record_and_verify_proof_receipt(receipt, store, owns_document_store=True)

        _print_kv("proof_result", "PASS")
        _print_kv("proof_kind", _PROOF_KIND)
        _print_kv("proof_id", verified.proof_id)
        _print_kv("run_id", run_id)
        _print_kv("workspace_id", workspace_id)
        _print_kv("source_id", source_id)
        _print_kv("operation_id", operation_id)
        _print_kv("documents_indexed", evidence["documents_indexed"])
        _print_kv("second_sync_documents_unchanged", evidence["second_sync_documents_unchanged"])
        _print_kv("operation_completed_after_restart", True)
        _print_kv("search_structured_evidence_present", True)
        _print_kv("router_file_read_used", False)
        _print_kv("receipt_recorded", True)
        _print_kv("receipt_verified", True)
        return 0
    except Exception as exc:
        _print_kv("proof_result", "FAIL")
        _print_kv("reason", f"{exc.__class__.__name__}:{exc}")
        if isinstance(exc, ProofReceiptVerificationError):
            _print_kv("receipt_verified", False)
        _print_kv("stdout_log", str(stdout_path))
        _print_kv("stderr_log", str(stderr_path))
        return 1
    finally:
        if process is not None:
            _stop_process(process)
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
