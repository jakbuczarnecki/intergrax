#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""In-container LKW Linux Docker runtime certification runner.

Executes the Linux Application Hosting Proof (``application-hosting`` phase of
the shared core-platform runner) then the Linux Optional OS Interaction Proof,
both with ``--mongodb-stack external``. Emits one structured JSON summary only
after both proofs pass.

This profile does **not** execute the full multi-phase Linux Core Platform
Proof.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

_SCRIPT_DIR = Path(__file__).resolve().parent
_APP_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_DIR.parent.parent

_CORE_PLATFORM_RUNNER_PY = _SCRIPT_DIR / "run-lkw-core-platform-proof.py"
_INTERACTION_PROOF_SH = _SCRIPT_DIR / "run-lkw-linux-interaction-proof.sh"
_APPLICATION_HOSTING_PROOF_KIND = "platform_application_hosting"
_INTERACTION_PROOF_KIND = "platform_linux_interaction"
_CERT_PROFILE = "linux_docker_runtime"
_KV_LINE = re.compile(r"^([A-Za-z0-9_.-]+)=(.*)$")


class CertificationInsideError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _parse_kv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _KV_LINE.match(line.strip())
        if match is None:
            continue
        values[match.group(1)] = match.group(2)
    return values


def _require_true(values: Mapping[str, str], key: str) -> None:
    if values.get(key) != "true":
        raise CertificationInsideError(f"false_or_missing:{key}")


def _require_value(values: Mapping[str, str], key: str, expected: str) -> None:
    actual = values.get(key)
    if actual != expected:
        raise CertificationInsideError(f"unexpected_{key}:{actual!r}")


def _require_nonblank(values: Mapping[str, str], key: str) -> str:
    value = str(values.get(key, "")).strip()
    if not value:
        raise CertificationInsideError(f"blank_{key}")
    return value


def detect_linux_runtime() -> dict[str, str]:
    system = platform.system()
    if system != "Linux":
        raise CertificationInsideError(f"non_linux_runtime:{system}")
    return {
        "platform_system": system,
        "os_version": platform.version(),
        "kernel_release": platform.release(),
        "architecture": platform.machine(),
    }


def _run(command: list[str], *, cwd: Path, env: Mapping[str, str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=dict(env),
        shell=False,
        check=False,
        text=True,
        capture_output=True,
    )
    combined = (completed.stdout or "") + (
        "\n" + completed.stderr if completed.stderr else ""
    )
    return int(completed.returncode), combined


def run_application_hosting_proof(*, env: Mapping[str, str]) -> dict[str, Any]:
    """Run the shared core-platform runner ``application-hosting`` phase only.

    This is not the full multi-phase Core Platform Proof.
    """
    code, output = _run(
        [
            sys.executable,
            str(_CORE_PLATFORM_RUNNER_PY),
            "--os-family",
            "linux",
            "--wrapper-id",
            "linux_sh",
            "--phase",
            "application-hosting",
            "--mongodb-stack",
            "external",
        ],
        cwd=_REPO_ROOT,
        env=env,
    )
    values = _parse_kv(output)
    if code != 0:
        raise CertificationInsideError(
            "application_hosting_proof_failed:"
            f"{values.get('failure_reason', 'nonzero_exit')}"
        )
    _require_value(values, "core_proof_result", "PASS")
    _require_value(values, "proof_kind", _APPLICATION_HOSTING_PROOF_KIND)
    _require_true(values, "proof_receipt_recorded")
    _require_true(values, "proof_receipt_verified")
    _require_true(values, "proof_receipt_query_verified")
    proof_id = _require_nonblank(values, "proof_id")
    run_id = _require_nonblank(values, "run_id")
    correlation_id = _require_nonblank(values, "correlation_id")
    return {
        "proof_kind": _APPLICATION_HOSTING_PROOF_KIND,
        "certified_scope": "application_hosting_phase",
        "full_core_platform_proof": False,
        "proof_id": proof_id,
        "run_id": run_id,
        "correlation_id": correlation_id,
        "result": "PASS",
        "receipt_recorded": True,
        "receipt_verified": True,
        "receipt_query_verified": True,
        "raw_kv": values,
    }


def run_interaction_proof(*, env: Mapping[str, str]) -> dict[str, Any]:
    if not _INTERACTION_PROOF_SH.is_file():
        raise CertificationInsideError("linux_interaction_launcher_missing")
    code, output = _run(
        [
            "sh",
            str(_INTERACTION_PROOF_SH),
            "--mongodb-stack",
            "external",
        ],
        cwd=_REPO_ROOT,
        env=env,
    )
    values = _parse_kv(output)
    if code != 0:
        reason = values.get("failure_reason", "nonzero_exit")
        detail = output.strip().splitlines()[-5:] if output.strip() else []
        suffix = ";".join(detail)[:400]
        raise CertificationInsideError(
            f"interaction_proof_failed:{reason}:{suffix}"
        )
    _require_value(values, "proof_result", "PASS")
    _require_value(values, "proof_kind", _INTERACTION_PROOF_KIND)
    _require_value(values, "adapter_id", "lkw.linux_shell")
    _require_value(values, "source", "linux_shell")
    _require_value(values, "wrapper_runtime", "posix_sh")
    _require_value(values, "client_runtime", "python")
    _require_true(values, "proof_receipt_recorded")
    _require_true(values, "proof_receipt_verified")
    _require_true(values, "proof_receipt_query_verified")
    proof_id = _require_nonblank(values, "proof_receipt_id")
    run_id = _require_nonblank(values, "proof_receipt_run_id")
    correlation_id = _require_nonblank(values, "correlation_id")
    return {
        "proof_kind": _INTERACTION_PROOF_KIND,
        "proof_id": proof_id,
        "run_id": run_id,
        "correlation_id": correlation_id,
        "adapter_id": "lkw.linux_shell",
        "source": "linux_shell",
        "client_runtime": "python",
        "wrapper_runtime": "posix_sh",
        "result": "PASS",
        "receipt_recorded": True,
        "receipt_verified": True,
        "receipt_query_verified": True,
        "raw_kv": values,
    }


def build_summary(
    *,
    runtime: Mapping[str, str],
    application_hosting: Mapping[str, Any],
    interaction: Mapping[str, Any],
) -> dict[str, Any]:
    source_commit = os.environ.get("LKW_CERTIFICATION_SOURCE_COMMIT", "").strip()
    return {
        "schema_version": "lkw.linux_docker_certification_inside.v1",
        "certification_result": "PASS",
        "certification_profile": _CERT_PROFILE,
        "execution_environment": "container",
        "execution_os_family": "linux",
        "os_version": runtime["os_version"],
        "kernel_release": runtime["kernel_release"],
        "architecture": runtime["architecture"],
        "containerized": True,
        "container_runtime": "docker",
        "client_runtime": "python",
        "wrapper_runtime": "posix_sh",
        "source_commit": source_commit,
        "full_core_platform_proof_certified": False,
        "application_hosting_proof": {
            "proof_kind": application_hosting["proof_kind"],
            "certified_scope": application_hosting["certified_scope"],
            "full_core_platform_proof": False,
            "proof_id": application_hosting["proof_id"],
            "run_id": application_hosting["run_id"],
            "correlation_id": application_hosting["correlation_id"],
            "result": application_hosting["result"],
            "receipt_recorded": application_hosting["receipt_recorded"],
            "receipt_verified": application_hosting["receipt_verified"],
            "receipt_query_verified": application_hosting["receipt_query_verified"],
        },
        "interaction_proof": {
            "proof_kind": interaction["proof_kind"],
            "proof_id": interaction["proof_id"],
            "run_id": interaction["run_id"],
            "correlation_id": interaction["correlation_id"],
            "adapter_id": interaction["adapter_id"],
            "source": interaction["source"],
            "client_runtime": interaction["client_runtime"],
            "wrapper_runtime": interaction["wrapper_runtime"],
            "result": interaction["result"],
            "receipt_recorded": interaction["receipt_recorded"],
            "receipt_verified": interaction["receipt_verified"],
            "receipt_query_verified": interaction["receipt_query_verified"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    _ = argv
    env = os.environ.copy()
    env.setdefault("LKW_EXECUTION_ENVIRONMENT", "container")
    env.setdefault("LKW_CONTAINER_RUNTIME", "docker")
    env.setdefault("LKW_CERTIFICATION_PROFILE", _CERT_PROFILE)
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        runtime = detect_linux_runtime()
        if not env.get("INTERGRAX_MONGODB_URI", "").strip():
            raise CertificationInsideError("external_mongodb_uri_required")
        application_hosting = run_application_hosting_proof(env=env)
        interaction = run_interaction_proof(env=env)
        summary = build_summary(
            runtime=runtime,
            application_hosting=application_hosting,
            interaction=interaction,
        )
    except CertificationInsideError as exc:
        failure = {
            "schema_version": "lkw.linux_docker_certification_inside.v1",
            "certification_result": "FAIL",
            "certification_profile": _CERT_PROFILE,
            "failure_reason": exc.reason,
            "full_core_platform_proof_certified": False,
            "certification_id": f"fail-{uuid.uuid4().hex[:12]}",
        }
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 1

    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
