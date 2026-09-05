# © Artur Czarnecki. All rights reserved.

"""Docker crash/resume orchestration for DS-E2E-06."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from testing_support.decision_e2e.environment import docker_cli_available, docker_daemon_available
from testing_support.decision_e2e.qualification_evidence import DockerCrashEvidence


@dataclass(frozen=True, slots=True)
class DockerQualificationRun:
    run_id: str
    image: str
    durable_root: Path
    checkpoint_evidence: DockerCrashEvidence | None
    authority_evidence: DockerCrashEvidence | None
    block_reason: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _docker_bin() -> str:
    return "docker"


def _qualification_image() -> str:
    return "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"


def _worker_command(phase: str, db_dir: Path, *, signal: Path, result: Path) -> list[str]:
    repo = _repo_root()
    worker = "testing_support.decision_e2e.docker_worker"
    return [
        "uv",
        "run",
        "python",
        "-m",
        worker,
        phase,
        "--db-dir",
        str(db_dir),
        "--signal",
        str(signal),
        "--result",
        str(result),
    ]


def _run_container(
    *,
    name: str,
    repo_mount: Path,
    durable_mount: Path,
    command: list[str],
    detach: bool,
) -> subprocess.CompletedProcess[str]:
    repo = repo_mount.resolve()
    durable = durable_mount.resolve()
    args = [
        _docker_bin(),
        "run",
        "--name",
        name,
        "--rm" if not detach else "--detach",
        "-v",
        f"{repo}:/workspace",
        "-v",
        f"{durable}:/durable",
        "-w",
        "/workspace",
        _qualification_image(),
        *command,
    ]
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _wait_for_ready(signal_path: Path, *, timeout_sec: float = 120.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if signal_path.is_file():
            return True
        time.sleep(0.5)
    return False


def _container_exit_code(container_id: str) -> int | None:
    completed = subprocess.run(
        [_docker_bin(), "inspect", container_id, "--format", "{{.State.ExitCode}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    raw = completed.stdout.strip()
    if not raw or raw == "<no value>":
        return None
    return int(raw)


def _docker_kill(container_name: str) -> tuple[str, int | None]:
    inspect = subprocess.run(
        [_docker_bin(), "inspect", container_name, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    container_id = inspect.stdout.strip() if inspect.returncode == 0 else container_name
    completed = subprocess.run(
        [_docker_bin(), "kill", "-s", "KILL", container_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker kill failed for {container_name}: {completed.stderr}")
    exit_code = _container_exit_code(container_id)
    return container_id, exit_code


def _cleanup_container(name: str) -> None:
    subprocess.run(
        [_docker_bin(), "rm", "-f", name],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_crash_window(
    *,
    run_id: str,
    window: str,
    persist_phase: str,
    resume_phase: str,
    durable_root: Path,
) -> tuple[DockerCrashEvidence | None, str | None]:
    window_dir = durable_root / window
    window_dir.mkdir(parents=True, exist_ok=True)
    signal_path = window_dir / "ready.json"
    result_path = window_dir / "result.json"
    writer_name = f"decision-e2e-{run_id}-{window}-writer"
    reader_name = f"decision-e2e-{run_id}-{window}-reader"
    db_dir = Path(f"/durable/{window}")

    writer = _run_container(
        name=writer_name,
        repo_mount=_repo_root(),
        durable_mount=durable_root,
        command=_worker_command(
            persist_phase,
            db_dir,
            signal=Path(f"/durable/{window}/ready.json"),
            result=Path(f"/durable/{window}/result.json"),
        ),
        detach=True,
    )
    if writer.returncode != 0:
        return None, f"{window} writer start failed: {writer.stderr}"

    host_signal = signal_path
    if not _wait_for_ready(host_signal):
        _cleanup_container(writer_name)
        return None, f"{window} writer readiness signal missing"

    try:
        killed_id, killed_exit = _docker_kill(writer_name)
    except RuntimeError as exc:
        _cleanup_container(writer_name)
        return None, str(exc)

    resume = _run_container(
        name=reader_name,
        repo_mount=_repo_root(),
        durable_mount=durable_root,
        command=_worker_command(
            resume_phase,
            db_dir,
            signal=Path(f"/durable/{window}/ready.json"),
            result=Path(f"/durable/{window}/result.json"),
        ),
        detach=False,
    )
    if resume.returncode != 0:
        return None, f"{window} resume failed: {resume.stderr}"
    if not result_path.is_file():
        return None, f"{window} resume result missing"

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return (
        DockerCrashEvidence(
            kill_method="docker_kill",
            killed_container_id=killed_id,
            killed_exit_code=killed_exit,
            resume_container_id=reader_name,
            durable_store_path=str(window_dir),
            window=window,
            final_disposition=str(payload.get("stage", "unknown")),
        ),
        None,
    )


def run_docker_crash_qualification(
    *,
    output_root: Path | None = None,
) -> DockerQualificationRun:
    if not docker_cli_available() or not docker_daemon_available():
        return DockerQualificationRun(
            run_id="blocked",
            image=_qualification_image(),
            durable_root=Path("."),
            checkpoint_evidence=None,
            authority_evidence=None,
            block_reason="Docker CLI or daemon unavailable",
        )

    run_id = uuid.uuid4().hex[:12]
    durable_root = output_root or Path(".tmp/decision_e2e_qualification") / f"docker-{run_id}"
    durable_root.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint_evidence, checkpoint_reason = _run_crash_window(
            run_id=run_id,
            window="checkpoint",
            persist_phase="checkpoint-persist",
            resume_phase="checkpoint-resume",
            durable_root=durable_root,
        )
        if checkpoint_evidence is None:
            return DockerQualificationRun(
                run_id=run_id,
                image=_qualification_image(),
                durable_root=durable_root,
                checkpoint_evidence=None,
                authority_evidence=None,
                block_reason=checkpoint_reason,
            )

        authority_evidence, authority_reason = _run_crash_window(
            run_id=run_id,
            window="authority",
            persist_phase="authority-commit",
            resume_phase="authority-resume",
            durable_root=durable_root,
        )
        if authority_evidence is None:
            return DockerQualificationRun(
                run_id=run_id,
                image=_qualification_image(),
                durable_root=durable_root,
                checkpoint_evidence=checkpoint_evidence,
                authority_evidence=None,
                block_reason=authority_reason,
            )

        return DockerQualificationRun(
            run_id=run_id,
            image=_qualification_image(),
            durable_root=durable_root,
            checkpoint_evidence=checkpoint_evidence,
            authority_evidence=authority_evidence,
        )
    finally:
        for window in ("checkpoint", "authority"):
            for role in ("writer", "reader"):
                _cleanup_container(f"decision-e2e-{run_id}-{window}-{role}")
