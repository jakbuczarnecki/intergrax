# © Artur Czarnecki. All rights reserved.
"""Generate the LKW cross-platform certification matrix (PROOF-PORTABILITY-1D-MATRIX).

Aggregates existing receipt-backed certification artifacts. Does not run live
proofs or create ProofReceipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "lkw.platform_certification_matrix.v1"
MATRIX_STATUS_VALID = "VALID"

WINDOWS_SOURCE_REL = Path(
    "docs/project/maintainers/public-adoption/evidence/LKW_WINDOWS_NATIVE_CERTIFICATION.json"
)
LINUX_SOURCE_REL = Path(
    "docs/project/maintainers/public-adoption/evidence/LKW_LINUX_DOCKER_CERTIFICATION.json"
)
MATRIX_JSON_REL = Path(
    "docs/project/maintainers/public-adoption/evidence/LKW_PLATFORM_CERTIFICATION_MATRIX.json"
)
MATRIX_MD_REL = Path("docs/project/maintainers/public-adoption/LKW_PLATFORM_CERTIFICATION_MATRIX.md")

WINDOWS_SCHEMA = "lkw.windows_native_certification.v1"
LINUX_SCHEMA = "lkw.linux_docker_certification.v1"

PROFILE_WINDOWS = "windows_native_runtime"
PROFILE_LINUX_DOCKER = "linux_docker_runtime"
PROFILE_LINUX_NATIVE = "linux_native_runtime"
PROFILE_MACOS = "macos_native_runtime"

HOSTING_PROOF_KIND = "platform_application_hosting"
HOSTING_SCOPE = "application_hosting_phase"
WINDOWS_INTERACTION_KIND = "platform_windows_interaction"
LINUX_INTERACTION_KIND = "platform_linux_interaction"

SECRET_NEEDLES = (
    "password",
    "secret",
    "api_key",
    "apikey",
    "token=",
    "mongodb://",
    "mongodb+srv://",
    "connection_string",
    "private_key",
)


class MatrixGenerationError(RuntimeError):
    """Fail-closed matrix generation / validation error."""


def find_repo_root(start: Path | None = None) -> Path:
    cursor = (start or Path(__file__).resolve()).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / ".git").exists() and (candidate / "applications").is_dir():
            return candidate
    raise MatrixGenerationError("repository_root_not_found")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_rev_parse_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MatrixGenerationError(f"git_rev_parse_failed:{result.stderr.strip()}")
    commit = result.stdout.strip()
    if not commit:
        raise MatrixGenerationError("git_rev_parse_empty")
    return commit


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise MatrixGenerationError(f"{label}_not_object")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatrixGenerationError(f"{label}_blank_or_missing")
    return value


def _require_bool(value: Any, label: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise MatrixGenerationError(f"{label}_not_bool")
    if expected is not None and value is not expected:
        raise MatrixGenerationError(f"{label}_expected_{expected}")
    return value


def _require_true(value: Any, label: str) -> bool:
    return _require_bool(value, label, expected=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MatrixGenerationError(f"missing_source:{path.as_posix()}")
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MatrixGenerationError(f"malformed_json:{path.as_posix()}:{exc}") from exc
    if not isinstance(data, dict):
        raise MatrixGenerationError(f"source_not_object:{path.as_posix()}")
    return data


def _extract_hosting(evidence: Mapping[str, Any], label: str) -> dict[str, Any]:
    block = _require_mapping(
        evidence.get("application_hosting_proof"),
        f"{label}.application_hosting_proof",
    )
    proof_kind = _require_str(block.get("proof_kind"), f"{label}.hosting.proof_kind")
    if proof_kind != HOSTING_PROOF_KIND:
        raise MatrixGenerationError(f"{label}.hosting.proof_kind_mismatch")
    scope = _require_str(
        block.get("certified_scope"), f"{label}.hosting.certified_scope"
    )
    if scope != HOSTING_SCOPE:
        raise MatrixGenerationError(f"{label}.hosting.certified_scope_mismatch")
    full_core = block.get("full_core_platform_proof")
    if full_core is not False:
        raise MatrixGenerationError(f"{label}.hosting.full_core_platform_proof_rejected")
    proof_id = _require_str(block.get("proof_id"), f"{label}.hosting.proof_id")
    run_id = _require_str(block.get("run_id"), f"{label}.hosting.run_id")
    correlation_id = _require_str(
        block.get("correlation_id"), f"{label}.hosting.correlation_id"
    )
    result = _require_str(block.get("result"), f"{label}.hosting.result")
    if result != "PASS":
        raise MatrixGenerationError(f"{label}.hosting.result_not_pass")
    _require_true(block.get("receipt_recorded"), f"{label}.hosting.receipt_recorded")
    _require_true(block.get("receipt_verified"), f"{label}.hosting.receipt_verified")
    _require_true(
        block.get("receipt_query_verified"),
        f"{label}.hosting.receipt_query_verified",
    )
    return {
        "proof_kind": proof_kind,
        "certified_scope": scope,
        "full_core_platform_proof": False,
        "proof_id": proof_id,
        "run_id": run_id,
        "correlation_id": correlation_id,
        "result": result,
        "receipt_recorded": True,
        "receipt_verified": True,
        "receipt_query_verified": True,
    }


def _extract_interaction(
    evidence: Mapping[str, Any],
    *,
    label: str,
    expected_kind: str,
    expected_adapter: str,
    expected_source: str,
    expected_wrapper: str,
    expected_powershell: str | None,
) -> dict[str, Any]:
    block = _require_mapping(
        evidence.get("interaction_proof"), f"{label}.interaction_proof"
    )
    proof_kind = _require_str(block.get("proof_kind"), f"{label}.interaction.proof_kind")
    if proof_kind != expected_kind:
        raise MatrixGenerationError(f"{label}.interaction.proof_kind_mismatch")
    adapter_id = _require_str(block.get("adapter_id"), f"{label}.interaction.adapter_id")
    if adapter_id != expected_adapter:
        raise MatrixGenerationError(f"{label}.interaction.adapter_id_mismatch")
    source = _require_str(block.get("source"), f"{label}.interaction.source")
    if source != expected_source:
        raise MatrixGenerationError(f"{label}.interaction.source_mismatch")
    client_runtime = _require_str(
        block.get("client_runtime"), f"{label}.interaction.client_runtime"
    )
    if client_runtime != "python":
        raise MatrixGenerationError(f"{label}.interaction.client_runtime_mismatch")
    wrapper_runtime = _require_str(
        block.get("wrapper_runtime"), f"{label}.interaction.wrapper_runtime"
    )
    if wrapper_runtime != expected_wrapper:
        raise MatrixGenerationError(f"{label}.interaction.wrapper_runtime_mismatch")
    powershell_runtime = block.get("powershell_runtime", None)
    if expected_powershell is None:
        if powershell_runtime not in (None,):
            # Linux source may omit the field; normalize to null.
            powershell_runtime = None
    else:
        powershell_runtime = _require_str(
            powershell_runtime, f"{label}.interaction.powershell_runtime"
        )
        if powershell_runtime != expected_powershell:
            raise MatrixGenerationError(
                f"{label}.interaction.powershell_runtime_mismatch"
            )
    proof_id = _require_str(block.get("proof_id"), f"{label}.interaction.proof_id")
    run_id = _require_str(block.get("run_id"), f"{label}.interaction.run_id")
    correlation_id = _require_str(
        block.get("correlation_id"), f"{label}.interaction.correlation_id"
    )
    result = _require_str(block.get("result"), f"{label}.interaction.result")
    if result != "PASS":
        raise MatrixGenerationError(f"{label}.interaction.result_not_pass")
    _require_true(
        block.get("receipt_recorded"), f"{label}.interaction.receipt_recorded"
    )
    _require_true(
        block.get("receipt_verified"), f"{label}.interaction.receipt_verified"
    )
    _require_true(
        block.get("receipt_query_verified"),
        f"{label}.interaction.receipt_query_verified",
    )
    return {
        "proof_kind": proof_kind,
        "proof_id": proof_id,
        "run_id": run_id,
        "correlation_id": correlation_id,
        "adapter_id": adapter_id,
        "source": source,
        "client_runtime": client_runtime,
        "wrapper_runtime": wrapper_runtime,
        "powershell_runtime": powershell_runtime,
        "result": result,
        "receipt_recorded": True,
        "receipt_verified": True,
        "receipt_query_verified": True,
    }


def validate_windows_evidence(evidence: Mapping[str, Any]) -> None:
    schema = _require_str(evidence.get("schema_version"), "windows.schema_version")
    if schema != WINDOWS_SCHEMA:
        raise MatrixGenerationError("windows.schema_version_mismatch")
    result = _require_str(
        evidence.get("certification_result"), "windows.certification_result"
    )
    if result != "PASS":
        raise MatrixGenerationError("windows.certification_result_not_pass")
    profile = _require_str(
        evidence.get("certification_profile"), "windows.certification_profile"
    )
    if profile != PROFILE_WINDOWS:
        raise MatrixGenerationError("windows.certification_profile_mismatch")
    env = _require_str(
        evidence.get("execution_environment"), "windows.execution_environment"
    )
    if env != "native_host":
        raise MatrixGenerationError("windows.execution_environment_mismatch")
    os_family = _require_str(
        evidence.get("execution_os_family"), "windows.execution_os_family"
    )
    if os_family != "windows":
        raise MatrixGenerationError("windows.execution_os_family_mismatch")
    _require_bool(
        evidence.get("native_windows_host_certified"),
        "windows.native_windows_host_certified",
        expected=True,
    )
    if evidence.get("full_core_platform_proof_certified_by_this_run") is not False:
        raise MatrixGenerationError("windows.full_core_claim_rejected")


def validate_linux_docker_evidence(evidence: Mapping[str, Any]) -> None:
    schema = _require_str(evidence.get("schema_version"), "linux.schema_version")
    if schema != LINUX_SCHEMA:
        raise MatrixGenerationError("linux.schema_version_mismatch")
    result = _require_str(
        evidence.get("certification_result"), "linux.certification_result"
    )
    if result != "PASS":
        raise MatrixGenerationError("linux.certification_result_not_pass")
    profile = _require_str(
        evidence.get("certification_profile"), "linux.certification_profile"
    )
    if profile != PROFILE_LINUX_DOCKER:
        raise MatrixGenerationError("linux.certification_profile_mismatch")
    env = _require_str(
        evidence.get("execution_environment"), "linux.execution_environment"
    )
    if env != "container":
        raise MatrixGenerationError("linux.execution_environment_mismatch")
    os_family = _require_str(
        evidence.get("execution_os_family"), "linux.execution_os_family"
    )
    if os_family != "linux":
        raise MatrixGenerationError("linux.execution_os_family_mismatch")
    _require_bool(
        evidence.get("native_linux_host_certified"),
        "linux.native_linux_host_certified",
        expected=False,
    )
    if evidence.get("full_core_platform_proof_certified") is not False:
        raise MatrixGenerationError("linux.full_core_claim_rejected")


def _uncertified_profile(
    *,
    profile_id: str,
    operating_system: str,
    execution_environment: str,
    limitations: list[str],
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "operating_system": operating_system,
        "execution_environment": execution_environment,
        "implementation_status": "implemented",
        "application_hosting_status": "not_live_certified",
        "os_interaction_status": "not_live_certified",
        "full_core_platform_proof_certified_by_profile": False,
        "native_host_certified": False,
        "evidence_available": False,
        "evidence_source": None,
        "certification_result": "NOT_CERTIFIED",
        "certified_at_utc": None,
        "certification_source_commit": None,
        "proofs": {},
        "limitations": limitations,
    }


def _certified_profile(
    *,
    profile_id: str,
    operating_system: str,
    execution_environment: str,
    native_host_certified: bool,
    evidence: Mapping[str, Any],
    evidence_source: str,
    hosting: Mapping[str, Any],
    interaction: Mapping[str, Any],
    limitations: list[str],
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "operating_system": operating_system,
        "execution_environment": execution_environment,
        "implementation_status": "implemented",
        "application_hosting_status": "live_certified",
        "os_interaction_status": "live_certified",
        "full_core_platform_proof_certified_by_profile": False,
        "native_host_certified": native_host_certified,
        "evidence_available": True,
        "evidence_source": evidence_source,
        "certification_result": _require_str(
            evidence.get("certification_result"), f"{profile_id}.certification_result"
        ),
        "certified_at_utc": _require_str(
            evidence.get("certified_at_utc"), f"{profile_id}.certified_at_utc"
        ),
        "certification_source_commit": _require_str(
            evidence.get("certification_source_commit"),
            f"{profile_id}.certification_source_commit",
        ),
        "proofs": {
            "application_hosting": dict(hosting),
            "os_interaction": dict(interaction),
        },
        "limitations": limitations,
    }


def _source_artifact_entry(
    *,
    path: Path,
    evidence: Mapping[str, Any],
    sha256: str,
) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "schema_version": evidence["schema_version"],
        "certification_profile": evidence["certification_profile"],
        "certification_result": evidence["certification_result"],
        "certified_at_utc": evidence["certified_at_utc"],
        "certification_source_commit": evidence["certification_source_commit"],
        "sha256": sha256,
    }


def build_matrix(
    *,
    repo_root: Path,
    windows_evidence: Mapping[str, Any],
    linux_evidence: Mapping[str, Any],
    windows_sha256: str,
    linux_sha256: str,
    generated_from_commit: str,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    validate_windows_evidence(windows_evidence)
    validate_linux_docker_evidence(linux_evidence)

    windows_hosting = _extract_hosting(windows_evidence, "windows")
    windows_interaction = _extract_interaction(
        windows_evidence,
        label="windows",
        expected_kind=WINDOWS_INTERACTION_KIND,
        expected_adapter="lkw.windows_powershell",
        expected_source="windows_powershell",
        expected_wrapper="windows_powershell",
        expected_powershell="Windows PowerShell",
    )
    linux_hosting = _extract_hosting(linux_evidence, "linux")
    linux_interaction = _extract_interaction(
        linux_evidence,
        label="linux",
        expected_kind=LINUX_INTERACTION_KIND,
        expected_adapter="lkw.linux_shell",
        expected_source="linux_shell",
        expected_wrapper="posix_sh",
        expected_powershell=None,
    )

    windows_certified_at = _require_str(
        windows_evidence.get("certified_at_utc"), "windows.certified_at_utc"
    )
    linux_certified_at = _require_str(
        linux_evidence.get("certified_at_utc"), "linux.certified_at_utc"
    )
    if generated_at_utc is None:
        generated_at_utc = max(windows_certified_at, linux_certified_at)

    source_summary = {
        "windows_sha256": windows_sha256,
        "linux_sha256": linux_sha256,
        "windows_certified_at_utc": windows_certified_at,
        "linux_certified_at_utc": linux_certified_at,
        "generated_from_commit": generated_from_commit,
    }
    summary_bytes = json.dumps(source_summary, sort_keys=True, separators=(",", ":"))
    matrix_id = "lkw-platform-matrix-" + hashlib.sha256(
        summary_bytes.encode("utf-8")
    ).hexdigest()[:12]

    windows_source = WINDOWS_SOURCE_REL.as_posix()
    linux_source = LINUX_SOURCE_REL.as_posix()

    profiles = {
        PROFILE_WINDOWS: _certified_profile(
            profile_id=PROFILE_WINDOWS,
            operating_system="windows",
            execution_environment="native_host",
            native_host_certified=True,
            evidence=windows_evidence,
            evidence_source=windows_source,
            hosting=windows_hosting,
            interaction=windows_interaction,
            limitations=[
                "Application Hosting certification is not the same as complete "
                "multi-phase Core Platform Proof certification.",
                "Full multi-phase Core Platform Proof was not re-executed by this "
                "Windows native profile.",
                "Windows Service installation was not tested.",
                "Windows desktop or tray integration was not tested.",
            ],
        ),
        PROFILE_LINUX_DOCKER: _certified_profile(
            profile_id=PROFILE_LINUX_DOCKER,
            operating_system="linux",
            execution_environment="container",
            native_host_certified=False,
            evidence=linux_evidence,
            evidence_source=linux_source,
            hosting=linux_hosting,
            interaction=linux_interaction,
            limitations=[
                "Application Hosting certification is not the same as complete "
                "multi-phase Core Platform Proof certification.",
                "Linux Docker runtime evidence does not certify native Linux "
                "installation.",
                "Full multi-phase Core Platform Proof was not executed by this "
                "Linux Docker profile.",
                "systemd, native desktop integration and native package "
                "installation remain outside scope.",
            ],
        ),
        PROFILE_LINUX_NATIVE: _uncertified_profile(
            profile_id=PROFILE_LINUX_NATIVE,
            operating_system="linux",
            execution_environment="native_host",
            limitations=[
                "Linux entrypoints are implemented.",
                "No separate native Linux host live certification artifact exists.",
                "Linux Docker runtime evidence does not certify native Linux "
                "installation.",
            ],
        ),
        PROFILE_MACOS: _uncertified_profile(
            profile_id=PROFILE_MACOS,
            operating_system="macos",
            execution_environment="native_host",
            limitations=[
                "macOS entrypoints are implemented.",
                "No macOS live certification artifact exists.",
                "No macOS ProofReceipt has been recorded for this matrix.",
            ],
        ),
    }

    # Fail closed: never allow uncertified profiles to look certified.
    if profiles[PROFILE_LINUX_NATIVE]["application_hosting_status"] != (
        "not_live_certified"
    ):
        raise MatrixGenerationError("linux_native_must_remain_uncertified")
    if profiles[PROFILE_MACOS]["application_hosting_status"] != "not_live_certified":
        raise MatrixGenerationError("macos_must_remain_uncertified")
    if profiles[PROFILE_LINUX_NATIVE]["evidence_available"] is not False:
        raise MatrixGenerationError("linux_native_evidence_must_be_false")
    if profiles[PROFILE_MACOS]["evidence_available"] is not False:
        raise MatrixGenerationError("macos_evidence_must_be_false")

    matrix = {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "generated_at_utc": generated_at_utc,
        "generated_from_commit": generated_from_commit,
        "matrix_status": MATRIX_STATUS_VALID,
        "profiles": profiles,
        "claims": {
            "shared_architecture_live_certified_on_native_windows": True,
            "shared_architecture_live_certified_in_linux_docker_runtime": True,
            "native_linux_host_live_certified": False,
            "macos_live_certified": False,
            "full_cross_platform_certification_complete": False,
            "full_multi_phase_core_certified_on_linux": False,
            "all_linux_deployments_certified": False,
        },
        "limitations": [
            "The current shared LKW proof architecture is receipt-backed and "
            "live-certified on native Windows and in a Linux Docker runtime. "
            "Native Linux host and macOS runtime certification remain pending.",
            "Application Hosting certification is not the same as complete "
            "multi-phase Core Platform Proof certification.",
            "Linux Docker runtime evidence does not certify native Linux "
            "installation.",
            "macOS remains implemented but not live-certified.",
        ],
        "source_artifacts": {
            "windows_native": _source_artifact_entry(
                path=WINDOWS_SOURCE_REL,
                evidence=windows_evidence,
                sha256=windows_sha256,
            ),
            "linux_docker": _source_artifact_entry(
                path=LINUX_SOURCE_REL,
                evidence=linux_evidence,
                sha256=linux_sha256,
            ),
        },
    }
    _ = repo_root  # reserved for future path checks
    return matrix


def render_markdown(matrix: Mapping[str, Any]) -> str:
    profiles = matrix["profiles"]
    windows = profiles[PROFILE_WINDOWS]
    linux_docker = profiles[PROFILE_LINUX_DOCKER]
    linux_native = profiles[PROFILE_LINUX_NATIVE]
    macos = profiles[PROFILE_MACOS]
    windows_src = matrix["source_artifacts"]["windows_native"]
    linux_src = matrix["source_artifacts"]["linux_docker"]
    windows_hosting = windows["proofs"]["application_hosting"]
    windows_ix = windows["proofs"]["os_interaction"]
    linux_hosting = linux_docker["proofs"]["application_hosting"]
    linux_ix = linux_docker["proofs"]["os_interaction"]

    lines = [
        "# LKW Platform Certification Matrix",
        "",
        "Authoritative, machine-validated consolidation of current LKW",
        "cross-platform certification evidence.",
        "",
        f"- Matrix ID: `{matrix['matrix_id']}`",
        f"- Matrix status: `{matrix['matrix_status']}`",
        f"- Generated at (UTC): `{matrix['generated_at_utc']}`",
        f"- Generated from commit: `{matrix['generated_from_commit']}`",
        "",
        "## Current certification status",
        "",
        "The current shared LKW proof architecture is receipt-backed and "
        "live-certified",
        "on native Windows and in a Linux Docker runtime. Native Linux host and "
        "macOS",
        "runtime certification remain pending.",
        "",
        "Application Hosting certification is not the same as complete multi-phase",
        "Core Platform Proof certification.",
        "",
        "| Profile | Environment | Implementation | Application Hosting | "
        "OS Interaction | Full Multi-Phase Core | Native Host Certified | Evidence |",
        "|---------|-------------|----------------|---------------------|"
        "----------------|-----------------------|------------------------|----------|",
        "| Windows native | native_host / windows | implemented | live-certified | "
        "live-certified | not certified by this profile | yes | yes |",
        "| Linux Docker runtime | container / linux | implemented | live-certified | "
        "live-certified | not certified by this profile | no | yes |",
        "| Linux native host | native_host / linux | implemented | not live-certified | "
        "not live-certified | not certified by this profile | no | no |",
        "| macOS native | native_host / macos | implemented | not live-certified | "
        "not live-certified | not certified by this profile | no | no |",
        "",
        "## Certified profiles",
        "",
        "### windows_native_runtime",
        "",
        f"- Certification profile: `{windows['profile_id']}`",
        f"- Certification result: `{windows['certification_result']}`",
        f"- Certification date: `{windows['certified_at_utc']}`",
        f"- Source commit: `{windows['certification_source_commit']}`",
        f"- Application-hosting proof ID: `{windows_hosting['proof_id']}`",
        f"- Interaction proof ID: `{windows_ix['proof_id']}`",
        f"- Source artifact: `{windows_src['path']}`",
        f"- Source artifact SHA-256: `{windows_src['sha256']}`",
        "",
        "### linux_docker_runtime",
        "",
        f"- Certification profile: `{linux_docker['profile_id']}`",
        f"- Certification result: `{linux_docker['certification_result']}`",
        f"- Certification date: `{linux_docker['certified_at_utc']}`",
        f"- Source commit: `{linux_docker['certification_source_commit']}`",
        f"- Application-hosting proof ID: `{linux_hosting['proof_id']}`",
        f"- Interaction proof ID: `{linux_ix['proof_id']}`",
        f"- Source artifact: `{linux_src['path']}`",
        f"- Source artifact SHA-256: `{linux_src['sha256']}`",
        "",
        "## Implemented but not live-certified profiles",
        "",
        "### linux_native_runtime",
        "",
        f"- Status: `{linux_native['certification_result']}`",
        "- Limitations:",
    ]
    for item in linux_native["limitations"]:
        lines.append(f"  - {item}")
    lines.extend(
        [
            "",
            "### macos_native_runtime",
            "",
            f"- Status: `{macos['certification_result']}`",
            "- Limitations:",
        ]
    )
    for item in macos["limitations"]:
        lines.append(f"  - {item}")
    lines.extend(
        [
            "",
            "## Evidence sources",
            "",
            "```text",
            "docs/project/maintainers/public-adoption/evidence/LKW_WINDOWS_NATIVE_CERTIFICATION.json",
            "docs/project/maintainers/public-adoption/evidence/LKW_LINUX_DOCKER_CERTIFICATION.json",
            "docs/project/maintainers/public-adoption/evidence/LKW_PLATFORM_CERTIFICATION_MATRIX.json",
            "```",
            "",
            "## Scope limitations",
            "",
        ]
    )
    for item in matrix["limitations"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Reproduction and validation",
            "",
            "Generate or refresh the matrix:",
            "",
            "```bash",
            "uv run python applications/local_workspace_application/scripts/"
            "generate-lkw-platform-certification-matrix.py",
            "```",
            "",
            "Check committed artifacts for staleness:",
            "",
            "```bash",
            "uv run python applications/local_workspace_application/scripts/"
            "generate-lkw-platform-certification-matrix.py --check",
            "```",
            "",
            "This matrix does not execute live proofs and does not create "
            "ProofReceipts.",
            "It only aggregates and validates existing certification evidence.",
            "",
        ]
    )
    text = "\n".join(lines)
    _reject_secrets(text, "markdown")
    return text


def _reject_secrets(text: str, label: str) -> None:
    lowered = text.lower()
    for needle in SECRET_NEEDLES:
        if needle in lowered:
            raise MatrixGenerationError(f"secret_needle_detected:{label}:{needle}")


def serialize_matrix_json(matrix: Mapping[str, Any]) -> str:
    text = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    _reject_secrets(text, "json")
    return text


def generate_artifacts(
    *,
    repo_root: Path,
    generated_at_utc: str | None = None,
    generated_from_commit: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    windows_path = repo_root / WINDOWS_SOURCE_REL
    linux_path = repo_root / LINUX_SOURCE_REL
    windows_evidence = load_json(windows_path)
    linux_evidence = load_json(linux_path)
    windows_sha = sha256_file(windows_path)
    linux_sha = sha256_file(linux_path)
    commit = generated_from_commit or git_rev_parse_head(repo_root)
    matrix = build_matrix(
        repo_root=repo_root,
        windows_evidence=windows_evidence,
        linux_evidence=linux_evidence,
        windows_sha256=windows_sha,
        linux_sha256=linux_sha,
        generated_from_commit=commit,
        generated_at_utc=generated_at_utc,
    )
    json_text = serialize_matrix_json(matrix)
    md_text = render_markdown(matrix)
    return matrix, json_text, md_text


def write_artifacts(repo_root: Path, json_text: str, md_text: str) -> None:
    json_path = repo_root / MATRIX_JSON_REL
    md_path = repo_root / MATRIX_MD_REL
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8", newline="\n")
    md_path.write_text(md_text, encoding="utf-8", newline="\n")


def check_artifacts(repo_root: Path, json_text: str, md_text: str) -> None:
    json_path = repo_root / MATRIX_JSON_REL
    md_path = repo_root / MATRIX_MD_REL
    errors: list[str] = []
    if not json_path.is_file():
        errors.append(f"missing_matrix_json:{json_path.as_posix()}")
    else:
        current = json_path.read_text(encoding="utf-8")
        if current != json_text:
            errors.append("stale_matrix_json")
    if not md_path.is_file():
        errors.append(f"missing_matrix_markdown:{md_path.as_posix()}")
    else:
        current_md = md_path.read_text(encoding="utf-8")
        if current_md != md_text:
            errors.append("stale_matrix_markdown")
    if errors:
        raise MatrixGenerationError(";".join(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or check the LKW platform certification matrix."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated content with committed files without writing.",
    )
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Override generated_at_utc (default: max of source certification timestamps).",
    )
    parser.add_argument(
        "--generated-from-commit",
        default=None,
        help="Override generated_from_commit (default: git rev-parse HEAD).",
    )
    return parser.parse_args(argv)


def _committed_generation_metadata(
    repo_root: Path,
) -> tuple[str | None, str | None]:
    """Read generation stamps from the committed matrix when present.

    After the documentation commit exists, ``git rev-parse HEAD`` changes while
    the matrix must retain the pre-commit ``generated_from_commit``. Check mode
    therefore reuses the committed stamps so unchanged sources stay byte-stable.
    """
    path = repo_root / MATRIX_JSON_REL
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    commit = data.get("generated_from_commit")
    generated_at = data.get("generated_at_utc")
    commit_s = commit if isinstance(commit, str) and commit.strip() else None
    at_s = generated_at if isinstance(generated_at, str) and generated_at.strip() else None
    return commit_s, at_s


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo_root = find_repo_root()
        generated_from_commit = args.generated_from_commit
        generated_at_utc = args.generated_at_utc
        if args.check and generated_from_commit is None:
            committed_commit, committed_at = _committed_generation_metadata(repo_root)
            if committed_commit is not None:
                generated_from_commit = committed_commit
            if generated_at_utc is None and committed_at is not None:
                generated_at_utc = committed_at
        _matrix, json_text, md_text = generate_artifacts(
            repo_root=repo_root,
            generated_at_utc=generated_at_utc,
            generated_from_commit=generated_from_commit,
        )
        if args.check:
            check_artifacts(repo_root, json_text, md_text)
            print("matrix_check=PASS")
            return 0
        write_artifacts(repo_root, json_text, md_text)
        print("matrix_write=PASS")
        print(f"matrix_json={MATRIX_JSON_REL.as_posix()}")
        print(f"matrix_markdown={MATRIX_MD_REL.as_posix()}")
        return 0
    except MatrixGenerationError as exc:
        print(f"matrix_error={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
