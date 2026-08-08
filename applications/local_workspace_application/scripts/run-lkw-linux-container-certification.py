#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Host-side orchestrator for LKW Linux Docker runtime certification.

Runs on the orchestrator host (typically Windows + Docker Desktop Linux engine),
builds the certification image, starts the dedicated Compose project, executes
the in-container proofs, writes receipt-backed evidence, and cleans up.
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
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_APP_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_DIR.parent.parent
_DOCKER_DIR = _APP_DIR / "docker"
_COMPOSE_FILE = _DOCKER_DIR / "linux-certification.compose.yml"
_DOCKERFILE = _DOCKER_DIR / "Dockerfile.linux-certification"
_INSIDE_SCRIPT = (
    "applications/local_workspace_application/scripts/"
    "run-lkw-linux-container-certification-inside.py"
)
_EVIDENCE_PATH = (
    _REPO_ROOT / "docs/project/maintainers/public-adoption/evidence/LKW_LINUX_DOCKER_CERTIFICATION.json"
)
_EXPECTED_PARENT = "40a73fbb455def6d5106180d74a7e65388457465"
_CERT_PROFILE = "linux_docker_runtime"
_IMAGE_TAG = "intergrax-lkw-linux-certification:local"
_COMPOSE_PROJECT = "lkw-linux-certification"
_BASE_IMAGE_REF = (
    "python:3.12-slim-bookworm@"
    "sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b"
)
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
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(args),
        cwd=str(cwd),
        env=None if env is None else dict(env),
        shell=False,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise CertificationOrchestratorError(
            f"command_failed:{args[0]}:{completed.returncode}"
        )
    return completed


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
    engine_os = (os_type.stdout or "").strip().lower()
    if engine_os != "linux":
        raise CertificationOrchestratorError(f"windows_container_mode:{engine_os}")

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
    return (completed.stdout or "").strip()


def git_diff_sha256() -> tuple[bool, str]:
    completed = _run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=_REPO_ROOT,
        timeout=120,
    )
    if completed.returncode not in (0, 1):
        raise CertificationOrchestratorError("git_diff_failed")
    # Also include untracked files that matter for certification via status porcelain.
    status = _run(
        ["git", "status", "--porcelain"],
        cwd=_REPO_ROOT,
        timeout=60,
    )
    if status.returncode != 0:
        raise CertificationOrchestratorError("git_status_failed")
    payload = (completed.stdout or "").encode("utf-8", errors="surrogateescape")
    payload += b"\n--STATUS--\n"
    payload += (status.stdout or "").encode("utf-8", errors="surrogateescape")
    digest = hashlib.sha256(payload).hexdigest()
    dirty = bool((completed.stdout or "").strip() or (status.stdout or "").strip())
    return dirty, digest


def resolve_image_metadata(image_ref: str) -> dict[str, Any]:
    """Resolve local image id and optional real repository digest.

    ``certification_image_id`` always comes from ``.Id``.
    ``certification_image_repo_digest`` is set only from a real
    ``repository-name@sha256:<digest>`` RepoDigests entry. A local image id is
    never copied into the repository digest field.
    """
    inspect_id = _run(
        ["docker", "image", "inspect", image_ref, "--format", "{{.Id}}"],
        cwd=_REPO_ROOT,
        timeout=60,
    )
    if inspect_id.returncode != 0:
        raise CertificationOrchestratorError("image_id_unresolved")
    image_id = (inspect_id.stdout or "").strip()
    if not image_id:
        raise CertificationOrchestratorError("image_id_unresolved")

    digests = _run(
        [
            "docker",
            "image",
            "inspect",
            image_ref,
            "--format",
            "{{json .RepoDigests}}",
        ],
        cwd=_REPO_ROOT,
        timeout=60,
    )
    repo_digest = "unavailable"
    raw_repo_digests: list[str] = []
    if digests.returncode == 0 and (digests.stdout or "").strip():
        try:
            parsed = json.loads(digests.stdout)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            for item in parsed:
                text = str(item).strip()
                if text:
                    raw_repo_digests.append(text)
                # Real registry digest form: repository-name@sha256:<digest>
                if "@sha256:" not in text:
                    continue
                repo_name, digest_part = text.split("@", 1)
                if not repo_name.strip():
                    continue
                if digest_part.startswith("sha256:") and len(digest_part) > len(
                    "sha256:"
                ):
                    repo_digest = digest_part
                    break
    return {
        "certification_image_reference": image_ref,
        "certification_image_id": image_id,
        "certification_image_repo_digest": repo_digest,
        "raw_repo_digests": raw_repo_digests,
    }


def resolve_base_image_digest() -> dict[str, str]:
    inspect = _run(
        [
            "docker",
            "image",
            "inspect",
            _BASE_IMAGE_REF,
            "--format",
            "{{json .RepoDigests}}",
        ],
        cwd=_REPO_ROOT,
        timeout=60,
    )
    digest = "unavailable"
    if inspect.returncode == 0 and (inspect.stdout or "").strip():
        try:
            parsed = json.loads(inspect.stdout)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            for item in parsed:
                text = str(item).strip()
                if "@sha256:" in text:
                    digest = text.split("@", 1)[1]
                    break
    # Dockerfile already pins the digest; fall back to the pin itself.
    if digest == "unavailable" and "@sha256:" in _BASE_IMAGE_REF:
        digest = _BASE_IMAGE_REF.split("@", 1)[1]
    return {
        "container_base_image": "python:3.12-slim-bookworm",
        "container_base_image_digest": digest,
    }


def compose_args(*extra: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        _COMPOSE_PROJECT,
        "-f",
        str(_COMPOSE_FILE),
        *extra,
    ]


def wait_for_mongodb_healthy(*, timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        completed = _run(
            [
                *compose_args(
                    "ps",
                    "--format",
                    "json",
                    "lkw-linux-certification-mongodb",
                )
            ],
            cwd=_REPO_ROOT,
            timeout=60,
        )
        if completed.returncode == 0 and (completed.stdout or "").strip():
            for line in completed.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows = row if isinstance(row, list) else [row]
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    health = str(item.get("Health") or "").strip().lower()
                    if health == "healthy":
                        return
        time.sleep(2)
    raise CertificationOrchestratorError("mongodb_health_timeout")


def extract_json_summary(output: str) -> dict[str, Any]:
    # Prefer the last JSON object line in stdout/stderr.
    candidates: list[dict[str, Any]] = []
    for line in output.splitlines():
        text = line.strip()
        if not text.startswith("{") or not text.endswith("}"):
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "certification_result" in parsed:
            candidates.append(parsed)
    if not candidates:
        # Fallback: whole output is JSON
        stripped = output.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CertificationOrchestratorError(
                    "malformed_in_container_output"
                ) from exc
            if isinstance(parsed, dict) and "certification_result" in parsed:
                return parsed
        raise CertificationOrchestratorError("malformed_in_container_output")
    return candidates[-1]


def validate_inside_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("certification_result") != "PASS":
        raise CertificationOrchestratorError("inside_certification_not_pass")
    if summary.get("certification_profile") != _CERT_PROFILE:
        raise CertificationOrchestratorError("unexpected_certification_profile")
    if summary.get("execution_environment") != "container":
        raise CertificationOrchestratorError("unexpected_execution_environment")
    if summary.get("execution_os_family") != "linux":
        raise CertificationOrchestratorError("non_linux_execution_os")
    if summary.get("containerized") is not True:
        raise CertificationOrchestratorError("not_containerized")
    if summary.get("container_runtime") != "docker":
        raise CertificationOrchestratorError("unexpected_container_runtime")
    if summary.get("client_runtime") != "python":
        raise CertificationOrchestratorError("unexpected_client_runtime")
    if summary.get("wrapper_runtime") != "posix_sh":
        raise CertificationOrchestratorError("unexpected_wrapper_runtime")

    if "core_proof" in summary and "application_hosting_proof" not in summary:
        raise CertificationOrchestratorError("missing_application_hosting_proof")
    if summary.get("full_core_platform_proof_certified") is True:
        raise CertificationOrchestratorError(
            "full_core_platform_proof_must_not_be_certified"
        )
    if summary.get("full_core_platform_proof_certified") is not False:
        raise CertificationOrchestratorError(
            "missing_full_core_platform_proof_certified_false"
        )

    hosting = summary.get("application_hosting_proof")
    interaction = summary.get("interaction_proof")
    if not isinstance(hosting, dict):
        raise CertificationOrchestratorError("missing_application_hosting_proof")
    if not isinstance(interaction, dict):
        raise CertificationOrchestratorError("missing_interaction_proof")
    if hosting.get("proof_kind") != "platform_application_hosting":
        raise CertificationOrchestratorError(
            "unexpected_application_hosting_proof_kind"
        )
    if hosting.get("certified_scope") != "application_hosting_phase":
        raise CertificationOrchestratorError(
            "unexpected_application_hosting_certified_scope"
        )
    if hosting.get("full_core_platform_proof") is not False:
        raise CertificationOrchestratorError(
            "application_hosting_must_not_claim_full_core"
        )
    if interaction.get("proof_kind") != "platform_linux_interaction":
        raise CertificationOrchestratorError("unexpected_interaction_proof_kind")
    for block_name, block in (
        ("application_hosting", hosting),
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
    if interaction.get("adapter_id") != "lkw.linux_shell":
        raise CertificationOrchestratorError("unexpected_adapter_id")
    if interaction.get("source") != "linux_shell":
        raise CertificationOrchestratorError("unexpected_source")
    if interaction.get("client_runtime") != "python":
        raise CertificationOrchestratorError("unexpected_interaction_client_runtime")
    if interaction.get("wrapper_runtime") != "posix_sh":
        raise CertificationOrchestratorError("unexpected_interaction_wrapper_runtime")


_SECRET_PATTERN = re.compile(
    r"(?i)(mongodb://[^\s\"']+|password\s*[:=]\s*\S+|token\s*[:=]\s*\S+)"
)


def scrub_secrets(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if str(key).strip().lower() in _SECRET_KEYS:
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
    engine: Mapping[str, str],
    base_image: Mapping[str, str],
    image_meta: Mapping[str, str],
    inside: Mapping[str, Any],
    source_commit: str,
    source_tree_dirty: bool,
    source_tree_diff_sha256: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evidence = {
        "schema_version": "lkw.linux_docker_certification.v1",
        "certification_id": f"lkw-linux-docker-{uuid.uuid4().hex[:12]}",
        "certification_profile": _CERT_PROFILE,
        "certification_result": "PASS",
        "certified_at_utc": now,
        "source_commit": source_commit,
        "certification_source_commit": source_commit,
        "certification_commit_parent": _EXPECTED_PARENT,
        "final_documentation_commit": "pending_pre_commit",
        "source_tree_dirty": source_tree_dirty,
        "source_tree_diff_sha256": source_tree_diff_sha256,
        "orchestrator_host_os": platform.system().lower(),
        "docker_engine_os": engine["docker_engine_os"],
        "docker_engine_architecture": engine["docker_engine_architecture"],
        "docker_engine_version": engine["docker_engine_version"],
        "execution_environment": "container",
        "execution_os_family": "linux",
        "execution_os_version": inside.get("os_version", "unavailable"),
        "execution_kernel_release": inside.get("kernel_release", "unavailable"),
        "execution_architecture": inside.get("architecture", "unavailable"),
        "container_runtime": "docker",
        "container_base_image": base_image["container_base_image"],
        "container_base_image_digest": base_image["container_base_image_digest"],
        "certification_image_reference": image_meta.get(
            "certification_image_reference", _IMAGE_TAG
        ),
        "certification_image_id": image_meta["certification_image_id"],
        "certification_image_repo_digest": image_meta[
            "certification_image_repo_digest"
        ],
        "application_hosting_proof": inside["application_hosting_proof"],
        "interaction_proof": inside["interaction_proof"],
        "full_core_platform_proof_certified": False,
        "native_linux_host_certified": False,
        "limitations": (
            "Linux Application Hosting Proof (application-hosting phase, "
            "proof_kind=platform_application_hosting) was live-certified in a "
            "Linux Docker runtime. Linux Optional OS Interaction Proof "
            "(proof_kind=platform_linux_interaction) was live-certified in the "
            "same Linux Docker runtime. The full multi-phase Core Platform Proof "
            "was not executed in this container certification. Elasticsearch, "
            "Kafka, Sentry and file-watcher Compose phases were not re-executed. "
            "The Docker engine was hosted through Docker Desktop/WSL2. Native "
            "Linux host installation was not separately tested. systemd, native "
            "desktop integration and native package installation remain outside "
            "scope."
        ),
        "reproduction_command": (
            "applications\\local_workspace_application\\scripts\\"
            "run-lkw-linux-container-certification.bat"
        ),
    }
    return scrub_secrets(evidence)


def write_evidence(evidence: Mapping[str, Any]) -> Path:
    _EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    _EVIDENCE_PATH.write_text(text, encoding="utf-8")
    return _EVIDENCE_PATH


def compose_down() -> None:
    _run(
        [*compose_args("down", "--remove-orphans", "-v")],
        cwd=_REPO_ROOT,
        timeout=300,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Host orchestrator for LKW Linux Docker runtime certification.",
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
        help="Required HEAD commit unless --pre-commit-certification is set.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse an already-built certification image (advanced).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cleanup_ok = True
    started = False
    try:
        require_docker()
        engine = inspect_docker_engine()
        head = git_rev_parse_head()
        dirty, diff_sha = git_diff_sha256()
        if args.pre_commit_certification:
            if head != args.expected_source_commit:
                raise CertificationOrchestratorError(
                    f"unexpected_parent_for_pre_commit:{head}"
                )
            if not dirty:
                # Still allow clean tree in pre-commit mode, but fingerprint remains.
                pass
        else:
            if head != args.expected_source_commit:
                raise CertificationOrchestratorError(
                    f"source_commit_mismatch:{head}"
                )
            if dirty:
                raise CertificationOrchestratorError("source_tree_dirty_without_flag")

        if not _COMPOSE_FILE.is_file() or not _DOCKERFILE.is_file():
            raise CertificationOrchestratorError("certification_docker_assets_missing")

        env = os.environ.copy()
        env["LKW_CERTIFICATION_SOURCE_COMMIT"] = head

        if not args.skip_build:
            ignore_src = _DOCKER_DIR / ".dockerignore"
            ignore_dst = _REPO_ROOT / ".dockerignore"
            wrote_temp_ignore = False
            previous_ignore: str | None = None
            if ignore_dst.is_file():
                previous_ignore = ignore_dst.read_text(encoding="utf-8")
            ignore_dst.write_text(ignore_src.read_text(encoding="utf-8"), encoding="utf-8")
            wrote_temp_ignore = previous_ignore is None
            try:
                build = _run(
                    [
                        "docker",
                        "build",
                        "-f",
                        str(_DOCKERFILE),
                        "-t",
                        _IMAGE_TAG,
                        str(_REPO_ROOT),
                    ],
                    cwd=_REPO_ROOT,
                    env=env,
                    timeout=3600,
                )
            finally:
                if wrote_temp_ignore and ignore_dst.is_file():
                    ignore_dst.unlink()
                elif previous_ignore is not None:
                    ignore_dst.write_text(previous_ignore, encoding="utf-8")
            if build.returncode != 0:
                raise CertificationOrchestratorError(
                    f"image_build_failed:{(build.stderr or build.stdout or '')[-800:]}"
                )

        base_image = resolve_base_image_digest()
        image_meta = resolve_image_metadata(_IMAGE_TAG)

        up = _run(
            [
                *compose_args(
                    "up",
                    "-d",
                    "--no-build",
                    "lkw-linux-certification-mongodb",
                )
            ],
            cwd=_REPO_ROOT,
            env=env,
            timeout=300,
        )
        if up.returncode != 0:
            raise CertificationOrchestratorError("compose_up_mongodb_failed")
        started = True
        wait_for_mongodb_healthy(timeout_seconds=180)

        run = _run(
            [
                *compose_args(
                    "run",
                    "--rm",
                    "--no-deps",
                    "lkw-linux-certification",
                    "python",
                    _INSIDE_SCRIPT,
                )
            ],
            cwd=_REPO_ROOT,
            env=env,
            timeout=3600,
        )
        combined = (run.stdout or "") + ("\n" + run.stderr if run.stderr else "")
        if run.returncode != 0:
            raise CertificationOrchestratorError(
                f"inside_runner_failed:{run.returncode}"
            )
        summary = extract_json_summary(combined)
        validate_inside_summary(summary)

        evidence = build_evidence(
            engine=engine,
            base_image=base_image,
            image_meta=image_meta,
            inside=summary,
            source_commit=head,
            source_tree_dirty=dirty if args.pre_commit_certification else False,
            source_tree_diff_sha256=diff_sha,
        )
        if evidence.get("certification_result") != "PASS":
            raise CertificationOrchestratorError("evidence_not_pass")
        path = write_evidence(evidence)
        print("certification_result=PASS", flush=True)
        print(f"evidence_file={path}", flush=True)
        print(
            "certification_image_reference="
            f"{image_meta.get('certification_image_reference', _IMAGE_TAG)}",
            flush=True,
        )
        print(f"certification_image_id={image_meta['certification_image_id']}", flush=True)
        print(
            "raw_repo_digests="
            f"{json.dumps(image_meta.get('raw_repo_digests', []))}",
            flush=True,
        )
        print(
            "certification_image_repo_digest="
            f"{image_meta['certification_image_repo_digest']}",
            flush=True,
        )
        print("full_core_platform_proof_certified=false", flush=True)
        return 0
    except CertificationOrchestratorError as exc:
        print("certification_result=FAIL", flush=True)
        print(f"failure_reason={exc.reason}", flush=True)
        return 1
    except subprocess.TimeoutExpired:
        print("certification_result=FAIL", flush=True)
        print("failure_reason=command_timeout", flush=True)
        return 1
    finally:
        if started or _COMPOSE_FILE.is_file():
            down = _run(
                [*compose_args("down", "--remove-orphans", "-v")],
                cwd=_REPO_ROOT,
                timeout=300,
            )
            if down.returncode != 0:
                cleanup_ok = False
                print("compose_cleanup_result=FAIL", flush=True)
            else:
                print("compose_cleanup_result=PASS", flush=True)
        if not cleanup_ok:
            # Material cleanup failure must not leave certification looking green.
            return 1


if __name__ == "__main__":
    sys.exit(main())
