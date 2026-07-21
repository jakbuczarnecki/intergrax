#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Native Windows LKW runtime certification orchestrator.

Certifies Windows Application Hosting Proof and Windows Optional OS
Interaction Proof on a real native Windows host through preserved public
Windows entrypoints and the shared Python proof runners.

Profile: ``windows_native_runtime``

This profile does **not** re-execute the full multi-phase Core Platform Proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_APP_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_DIR.parent.parent
_DOCKER_DIR = _APP_DIR / "docker"
_BASE_COMPOSE = _DOCKER_DIR / "docker-compose.yml"
_MONGODB_COMPOSE = _DOCKER_DIR / "docker-compose.mongodb.yml"

_HOSTING_BAT = _SCRIPT_DIR / "run-lkw-core-platform-proof-windows.bat"
_INTERACTION_BAT = _SCRIPT_DIR / "run-lkw-windows-interaction-proof.bat"
_EVIDENCE_PATH = (
    _REPO_ROOT / "docs/public-adoption/evidence/LKW_WINDOWS_NATIVE_CERTIFICATION.json"
)

_EXPECTED_PARENT = "6b71a841c894728766fd6f574c9cd53ad12ec5f9"
_CERT_PROFILE = "windows_native_runtime"
_APPLICATION_HOSTING_PROOF_KIND = "platform_application_hosting"
_INTERACTION_PROOF_KIND = "platform_windows_interaction"
_ADAPTER_ID = "lkw.windows_powershell"
_SOURCE = "windows_powershell"
_CLIENT_RUNTIME = "python"
_WRAPPER_RUNTIME = "windows_powershell"
_POWERSHELL_RUNTIME = "Windows PowerShell"

_KV_LINE = re.compile(r"^([A-Za-z0-9_.-]+)=(.*)$")
_SECRET_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "uri",
        "mongodb_uri",
        "intergrax_mongodb_uri",
        "access_token",
        "authorization",
    }
)
_SECRET_PATTERN = re.compile(
    r"(?i)(mongodb://[^\s\"']+|password\s*[:=]\s*\S+|token\s*[:=]\s*\S+|"
    r"authorization\s*[:=]\s*\S+)"
)

# Untracked paths that must contribute content to the source fingerprint.
_FINGERPRINT_PATH_PREFIXES = (
    "applications/local_workspace_application/scripts/run-lkw-windows-native-certification",
    "docs/public-adoption/evidence/LKW_WINDOWS_NATIVE_CERTIFICATION.json",
    "docs/public-adoption/LKW_PLATFORM_PROOF.md",
    "applications/local_workspace_application/docs/ARCHITECTURE.md",
    "applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md",
    "tests/unit/applications/local_workspace_application/test_lkw_windows_native_certification.py",
    "tests/unit/docs/test_lkw_platform_proof_contract.py",
)


class CertificationOrchestratorError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _run(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        env=None if env is None else dict(env),
        shell=False,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def parse_kv_output(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _KV_LINE.match(line.strip())
        if match is None:
            continue
        values[match.group(1)] = match.group(2)
    return values


def _require_true(values: Mapping[str, str], key: str) -> None:
    if values.get(key) != "true":
        raise CertificationOrchestratorError(f"false_or_missing:{key}")


def _require_value(values: Mapping[str, str], key: str, expected: str) -> None:
    actual = values.get(key)
    if actual != expected:
        raise CertificationOrchestratorError(f"unexpected_{key}:{actual!r}")


def _require_nonblank(values: Mapping[str, str], key: str) -> str:
    value = str(values.get(key, "")).strip()
    if not value:
        raise CertificationOrchestratorError(f"blank_{key}")
    return value


def require_native_windows() -> dict[str, str]:
    system = platform.system()
    if system != "Windows":
        raise CertificationOrchestratorError(f"non_windows_runtime:{system}")
    if os.name != "nt":
        raise CertificationOrchestratorError(f"non_nt_os_name:{os.name}")
    return {
        "execution_os_family": "windows",
        "execution_os_version": platform.version(),
        "execution_kernel_release": platform.release(),
        "execution_architecture": platform.machine(),
        "python_version": platform.python_version(),
    }


def require_powershell() -> str:
    path = shutil.which("powershell.exe") or shutil.which("powershell")
    if path is None:
        raise CertificationOrchestratorError("powershell_unavailable")
    probe = _run(
        [
            path,
            "-NoProfile",
            "-Command",
            "$PSVersionTable.PSEdition",
        ],
        cwd=_REPO_ROOT,
        timeout=30,
    )
    edition = (probe.stdout or "").strip()
    if probe.returncode != 0 or not edition:
        # powershell.exe present is enough; edition probe is best-effort.
        return _POWERSHELL_RUNTIME
    if edition.lower() not in {"desktop", "core"}:
        raise CertificationOrchestratorError(f"unexpected_powershell_edition:{edition}")
    # Frozen evidence label remains Windows PowerShell for this adapter contract.
    return _POWERSHELL_RUNTIME


def require_docker() -> None:
    if shutil.which("docker") is None:
        raise CertificationOrchestratorError("docker_unavailable")
    probe = _run(["docker", "version"], cwd=_REPO_ROOT, timeout=60)
    if probe.returncode != 0:
        raise CertificationOrchestratorError("docker_unavailable")


def inspect_docker_engine() -> dict[str, str]:
    os_type = _run(
        ["docker", "info", "--format", "{{.OSType}}"],
        cwd=_REPO_ROOT,
        timeout=60,
    )
    if os_type.returncode != 0:
        raise CertificationOrchestratorError("docker_info_failed")
    engine_os = (os_type.stdout or "").strip().lower() or "unavailable"
    arch = _run(
        ["docker", "info", "--format", "{{.Architecture}}"],
        cwd=_REPO_ROOT,
        timeout=60,
    )
    version = _run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        cwd=_REPO_ROOT,
        timeout=60,
    )
    return {
        "docker_engine_os": engine_os,
        "docker_engine_architecture": (arch.stdout or "").strip() or "unavailable",
        "docker_engine_version": (version.stdout or "").strip() or "unavailable",
    }


def git_rev_parse_head() -> str:
    completed = _run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, timeout=30)
    if completed.returncode != 0:
        raise CertificationOrchestratorError("git_rev_parse_failed")
    head = (completed.stdout or "").strip()
    if not head:
        raise CertificationOrchestratorError("git_rev_parse_failed")
    return head


def _path_is_fingerprint_relevant(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    return any(normalized.startswith(prefix) for prefix in _FINGERPRINT_PATH_PREFIXES)


def git_diff_sha256() -> tuple[bool, str]:
    """Fingerprint tracked diffs plus relevant untracked certification files."""
    completed = _run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=_REPO_ROOT,
        timeout=120,
    )
    if completed.returncode not in (0, 1):
        raise CertificationOrchestratorError("git_diff_failed")
    status = _run(
        ["git", "status", "--porcelain", "-u"],
        cwd=_REPO_ROOT,
        timeout=60,
    )
    if status.returncode != 0:
        raise CertificationOrchestratorError("git_status_failed")

    hasher = hashlib.sha256()
    hasher.update((completed.stdout or "").encode("utf-8", errors="surrogateescape"))
    hasher.update(b"\n--STATUS--\n")
    hasher.update((status.stdout or "").encode("utf-8", errors="surrogateescape"))

    for line in (status.stdout or "").splitlines():
        if len(line) < 4:
            continue
        # Untracked: "?? path"
        if not line.startswith("?? "):
            continue
        rel = line[3:].strip().strip('"')
        if not _path_is_fingerprint_relevant(rel):
            continue
        path = _REPO_ROOT / rel
        hasher.update(b"\n--UNTRACKED--\n")
        hasher.update(rel.encode("utf-8", errors="surrogateescape"))
        hasher.update(b"\n")
        if path.is_file():
            hasher.update(path.read_bytes())
        else:
            hasher.update(b"<missing-or-dir>")

    dirty = bool((completed.stdout or "").strip() or (status.stdout or "").strip())
    return dirty, hasher.hexdigest()


def invoke_public_bat(bat_path: Path, *extra: str) -> tuple[int, str]:
    """Invoke a public Windows ``.bat`` entrypoint via ``cmd.exe``."""
    if not bat_path.is_file():
        raise CertificationOrchestratorError(f"launcher_missing:{bat_path.name}")
    completed = _run(
        ["cmd.exe", "/c", str(bat_path), *extra],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=3600,
    )
    combined = (completed.stdout or "") + (
        "\n" + completed.stderr if completed.stderr else ""
    )
    return int(completed.returncode), combined


def validate_application_hosting_kv(values: Mapping[str, str]) -> dict[str, Any]:
    if not values:
        raise CertificationOrchestratorError("malformed_application_hosting_output")
    if values.get("core_proof_result") == "SKIP" or values.get("result") == "SKIP":
        raise CertificationOrchestratorError("application_hosting_proof_skipped")
    if values.get("core_proof_result") != "PASS" and values.get("result") != "PASS":
        raise CertificationOrchestratorError(
            f"application_hosting_proof_failed:"
            f"{values.get('failure_reason', 'not_pass')}"
        )
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
    }


def validate_interaction_kv(values: Mapping[str, str]) -> dict[str, Any]:
    if not values:
        raise CertificationOrchestratorError("malformed_interaction_output")
    if values.get("proof_result") == "SKIP":
        raise CertificationOrchestratorError("interaction_proof_skipped")
    if values.get("proof_result") != "PASS":
        raise CertificationOrchestratorError(
            f"interaction_proof_failed:{values.get('failure_reason', 'not_pass')}"
        )
    _require_value(values, "proof_kind", _INTERACTION_PROOF_KIND)
    _require_value(values, "adapter_id", _ADAPTER_ID)
    _require_value(values, "source", _SOURCE)
    _require_value(values, "client_runtime", _CLIENT_RUNTIME)
    _require_value(values, "wrapper_runtime", _WRAPPER_RUNTIME)
    _require_value(values, "powershell_runtime", _POWERSHELL_RUNTIME)
    _require_value(values, "os_family", "windows")
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
        "adapter_id": _ADAPTER_ID,
        "source": _SOURCE,
        "client_runtime": _CLIENT_RUNTIME,
        "wrapper_runtime": _WRAPPER_RUNTIME,
        "powershell_runtime": _POWERSHELL_RUNTIME,
        "result": "PASS",
        "receipt_recorded": True,
        "receipt_verified": True,
        "receipt_query_verified": True,
    }


def validate_certification_blocks(
    *,
    application_hosting: Mapping[str, Any],
    interaction: Mapping[str, Any],
) -> None:
    if "core_proof" in application_hosting and "proof_kind" not in application_hosting:
        raise CertificationOrchestratorError("core_proof_not_accepted")
    if not isinstance(application_hosting, dict) or not application_hosting:
        raise CertificationOrchestratorError("missing_application_hosting_proof")
    if not isinstance(interaction, dict) or not interaction:
        raise CertificationOrchestratorError("missing_interaction_proof")
    if application_hosting.get("proof_kind") != _APPLICATION_HOSTING_PROOF_KIND:
        raise CertificationOrchestratorError(
            "unexpected_application_hosting_proof_kind"
        )
    if application_hosting.get("certified_scope") != "application_hosting_phase":
        raise CertificationOrchestratorError(
            "unexpected_application_hosting_certified_scope"
        )
    if application_hosting.get("full_core_platform_proof") is not False:
        raise CertificationOrchestratorError(
            "application_hosting_must_not_claim_full_core"
        )
    if interaction.get("proof_kind") != _INTERACTION_PROOF_KIND:
        raise CertificationOrchestratorError("unexpected_interaction_proof_kind")
    if interaction.get("adapter_id") != _ADAPTER_ID:
        raise CertificationOrchestratorError("unexpected_adapter_id")
    if interaction.get("source") != _SOURCE:
        raise CertificationOrchestratorError("unexpected_source")
    if interaction.get("client_runtime") != _CLIENT_RUNTIME:
        raise CertificationOrchestratorError("unexpected_client_runtime")
    if interaction.get("wrapper_runtime") != _WRAPPER_RUNTIME:
        raise CertificationOrchestratorError("unexpected_wrapper_runtime")
    if interaction.get("powershell_runtime") != _POWERSHELL_RUNTIME:
        raise CertificationOrchestratorError("unexpected_powershell_runtime")
    for block_name, block in (
        ("application_hosting", application_hosting),
        ("interaction", interaction),
    ):
        if block.get("result") != "PASS":
            raise CertificationOrchestratorError(f"{block_name}_proof_failed")
        for flag in (
            "receipt_recorded",
            "receipt_verified",
            "receipt_query_verified",
        ):
            if block.get(flag) is not True:
                raise CertificationOrchestratorError(
                    f"{block_name}_false_receipt_flag:{flag}"
                )
        for key in ("proof_id", "run_id", "correlation_id"):
            if not str(block.get(key, "")).strip():
                raise CertificationOrchestratorError(f"{block_name}_blank_{key}")


def run_application_hosting_proof() -> dict[str, Any]:
    code, output = invoke_public_bat(
        _HOSTING_BAT,
        "--phase",
        "application-hosting",
    )
    values = parse_kv_output(output)
    if code != 0:
        raise CertificationOrchestratorError(
            "application_hosting_proof_failed:"
            f"{values.get('failure_reason', f'nonzero_exit:{code}')}"
        )
    return validate_application_hosting_kv(values)


def run_interaction_proof() -> dict[str, Any]:
    code, output = invoke_public_bat(_INTERACTION_BAT)
    values = parse_kv_output(output)
    if code != 0:
        raise CertificationOrchestratorError(
            "interaction_proof_failed:"
            f"{values.get('failure_reason', f'nonzero_exit:{code}')}"
        )
    return validate_interaction_kv(values)


def scrub_secrets(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if str(key).strip().lower() in _SECRET_KEYS:
                continue
            if str(key).strip().lower().endswith("_uri"):
                continue
            cleaned[str(key)] = scrub_secrets(value)
        return cleaned
    if isinstance(payload, list):
        return [scrub_secrets(item) for item in payload]
    if isinstance(payload, str):
        if _SECRET_PATTERN.search(payload):
            return "[redacted]"
        return payload
    return payload


def build_evidence(
    *,
    runtime: Mapping[str, str],
    docker: Mapping[str, str],
    powershell_runtime: str,
    application_hosting: Mapping[str, Any],
    interaction: Mapping[str, Any],
    source_commit: str,
    source_tree_dirty: bool,
    source_tree_diff_sha256: str,
) -> dict[str, Any]:
    validate_certification_blocks(
        application_hosting=application_hosting,
        interaction=interaction,
    )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evidence = {
        "schema_version": "lkw.windows_native_certification.v1",
        "certification_id": f"lkw-windows-native-{uuid.uuid4().hex[:12]}",
        "certification_profile": _CERT_PROFILE,
        "certification_result": "PASS",
        "certified_at_utc": now,
        "certification_source_commit": source_commit,
        "certification_commit_parent": _EXPECTED_PARENT,
        "final_documentation_commit": "pending_pre_commit",
        "source_tree_dirty": source_tree_dirty,
        "source_tree_diff_sha256": source_tree_diff_sha256,
        "execution_environment": "native_host",
        "execution_os_family": "windows",
        "execution_os_version": runtime["execution_os_version"],
        "execution_kernel_release": runtime["execution_kernel_release"],
        "execution_architecture": runtime["execution_architecture"],
        "python_version": runtime["python_version"],
        "powershell_runtime": powershell_runtime,
        "docker_engine_os": docker["docker_engine_os"],
        "docker_engine_architecture": docker["docker_engine_architecture"],
        "docker_engine_version": docker["docker_engine_version"],
        "application_hosting_proof": {
            "proof_kind": application_hosting["proof_kind"],
            "certified_scope": application_hosting["certified_scope"],
            "full_core_platform_proof": False,
            "proof_id": application_hosting["proof_id"],
            "run_id": application_hosting["run_id"],
            "correlation_id": application_hosting["correlation_id"],
            "result": "PASS",
            "receipt_recorded": True,
            "receipt_verified": True,
            "receipt_query_verified": True,
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
            "powershell_runtime": interaction["powershell_runtime"],
            "result": "PASS",
            "receipt_recorded": True,
            "receipt_verified": True,
            "receipt_query_verified": True,
        },
        "full_core_platform_proof_certified_by_this_run": False,
        "native_windows_host_certified": True,
        "limitations": (
            "The run occurred on a native Windows host. "
            "The Application Hosting phase "
            "(proof_kind=platform_application_hosting) was live-certified. "
            "The Windows Optional OS Interaction Proof "
            "(proof_kind=platform_windows_interaction) was live-certified "
            "through the shared Python client and shared OS proof runner. "
            "This run did not re-execute every full multi-phase Core Platform "
            "Proof component. Windows Service installation was not tested. "
            "Windows desktop or tray integration was not tested. "
            "Linux native-host and macOS certification are outside this run."
        ),
        "reproduction_command": (
            "applications\\local_workspace_application\\scripts\\"
            "run-lkw-windows-native-certification.bat"
        ),
    }
    cleaned = scrub_secrets(evidence)
    if cleaned.get("certification_result") != "PASS":
        raise CertificationOrchestratorError("evidence_not_pass")
    if "core_proof" in cleaned:
        raise CertificationOrchestratorError("core_proof_not_accepted")
    serialized = json.dumps(cleaned, sort_keys=True)
    if "mongodb://" in serialized.lower():
        raise CertificationOrchestratorError("secret_leaked_in_evidence")
    return cleaned


def write_evidence(evidence: Mapping[str, Any]) -> Path:
    if evidence.get("certification_result") != "PASS":
        raise CertificationOrchestratorError("refusing_non_pass_evidence")
    _EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    _EVIDENCE_PATH.write_text(text, encoding="utf-8")
    return _EVIDENCE_PATH


def compose_cleanup_args() -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(_BASE_COMPOSE),
        "-f",
        str(_MONGODB_COMPOSE),
        "down",
        "--remove-orphans",
        "-v",
    ]


def cleanup_managed_compose() -> bool:
    if not _BASE_COMPOSE.is_file() or not _MONGODB_COMPOSE.is_file():
        return True
    down = _run(compose_cleanup_args(), cwd=_REPO_ROOT, timeout=300)
    return down.returncode == 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Native Windows certification for LKW Application Hosting and "
            "Windows Optional OS Interaction proofs."
        ),
    )
    parser.add_argument(
        "--pre-commit-certification",
        action="store_true",
        help=(
            "Allow dirty source tree and fingerprint git diff; required when "
            "evidence will be committed in the same change."
        ),
    )
    parser.add_argument(
        "--expected-source-commit",
        default=_EXPECTED_PARENT,
        help="Required HEAD commit (frozen base for pre-commit mode).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cleanup_ok = True
    managed_started = False
    exit_code = 1
    try:
        runtime = require_native_windows()
        powershell_runtime = require_powershell()
        require_docker()
        docker = inspect_docker_engine()

        head = git_rev_parse_head()
        dirty, diff_sha = git_diff_sha256()
        if args.pre_commit_certification:
            if head != args.expected_source_commit:
                raise CertificationOrchestratorError(
                    f"unexpected_parent_for_pre_commit:{head}"
                )
        else:
            if head != args.expected_source_commit:
                raise CertificationOrchestratorError(f"source_commit_mismatch:{head}")
            if dirty:
                raise CertificationOrchestratorError("source_tree_dirty_without_flag")

        managed_started = True
        application_hosting = run_application_hosting_proof()
        interaction = run_interaction_proof()
        validate_certification_blocks(
            application_hosting=application_hosting,
            interaction=interaction,
        )

        evidence = build_evidence(
            runtime=runtime,
            docker=docker,
            powershell_runtime=powershell_runtime,
            application_hosting=application_hosting,
            interaction=interaction,
            source_commit=head,
            source_tree_dirty=dirty if args.pre_commit_certification else False,
            source_tree_diff_sha256=diff_sha,
        )
        path = write_evidence(evidence)
        print("certification_result=PASS", flush=True)
        print(f"evidence_file={path}", flush=True)
        print(f"certification_profile={_CERT_PROFILE}", flush=True)
        print("full_core_platform_proof_certified_by_this_run=false", flush=True)
        print("native_windows_host_certified=true", flush=True)
        exit_code = 0
    except CertificationOrchestratorError as exc:
        print("certification_result=FAIL", flush=True)
        print(f"failure_reason={exc.reason}", flush=True)
        exit_code = 1
    except subprocess.TimeoutExpired:
        print("certification_result=FAIL", flush=True)
        print("failure_reason=command_timeout", flush=True)
        exit_code = 1
    finally:
        if managed_started:
            if cleanup_managed_compose():
                print("compose_cleanup_result=PASS", flush=True)
            else:
                cleanup_ok = False
                print("compose_cleanup_result=FAIL", flush=True)
        if not cleanup_ok:
            return 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
