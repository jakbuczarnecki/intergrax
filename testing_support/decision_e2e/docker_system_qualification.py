# © Artur Czarnecki. All rights reserved.

"""Docker orchestration for DS-E2E-15J system qualification."""

from __future__ import annotations

import json
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from testing_support.decision_e2e.environment import (
    docker_cli_available,
    docker_daemon_available,
)


@dataclass(frozen=True, slots=True)
class DockerSystemScenarioRun:
    scenario_id: str
    run_id: str
    image: str
    durable_root: Path
    result: dict[str, Any] | None
    block_reason: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _docker_bin() -> str:
    return "docker"


def _qualification_image() -> str:
    return "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"


_CONTAINER_PROJECT_ENV = "/opt/intergrax-decision-e2e-system-qual-venv"


def _container_result_path(scenario_id: str) -> str:
    """POSIX path for in-container writes (never use host Path for /durable)."""
    return f"/durable/{scenario_id}/result.json"


def _worker_command(scenario: str, result_path: str) -> list[str]:
    worker = [
        "uv",
        "run",
        "python",
        "-m",
        "testing_support.decision_e2e.docker_system_worker",
        scenario,
        "--result",
        result_path,
    ]
    env = _CONTAINER_PROJECT_ENV
    run_line = " ".join(shlex.quote(part) for part in worker)
    script = (
        f"uv venv --clear {shlex.quote(env)} && "
        f"UV_PROJECT_ENVIRONMENT={shlex.quote(env)} UV_LINK_MODE=copy "
        f"uv sync --frozen --no-dev && "
        f"UV_PROJECT_ENVIRONMENT={shlex.quote(env)} UV_LINK_MODE=copy {run_line}"
    )
    return ["bash", "-lc", script]


def _cleanup_container(name: str) -> None:
    subprocess.run(
        [_docker_bin(), "rm", "-f", name],
        capture_output=True,
        text=True,
        check=False,
    )


def run_docker_system_scenario(
    scenario_id: str,
    *,
    output_root: Path | None = None,
) -> DockerSystemScenarioRun:
    image = _qualification_image()
    if not docker_cli_available() or not docker_daemon_available():
        return DockerSystemScenarioRun(
            scenario_id=scenario_id,
            run_id="blocked",
            image=image,
            durable_root=Path("."),
            result=None,
            block_reason="Docker CLI or daemon unavailable",
        )

    run_id = uuid.uuid4().hex[:12]
    durable_root = (
        output_root
        or Path(".tmp/decision_e2e_qualification") / f"docker-system-{run_id}"
    )
    scenario_dir = durable_root / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    host_result = scenario_dir / "result.json"
    container_result = _container_result_path(scenario_id)
    container_name = f"decision-e2e-sys-{run_id}-{scenario_id}"

    try:
        completed = subprocess.run(
            [
                _docker_bin(),
                "run",
                "--name",
                container_name,
                "--rm",
                "-v",
                f"{_repo_root().resolve()}:/workspace",
                "-v",
                f"{durable_root.resolve()}:/durable",
                "-w",
                "/workspace",
                image,
                *_worker_command(scenario_id, container_result),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if not host_result.is_file():
            detail = completed.stderr.strip() or completed.stdout.strip()
            return DockerSystemScenarioRun(
                scenario_id=scenario_id,
                run_id=run_id,
                image=image,
                durable_root=durable_root,
                result=None,
                block_reason=f"container exit {completed.returncode}: {detail}",
            )
        payload = json.loads(host_result.read_text(encoding="utf-8"))
        if completed.returncode != 0 and payload.get("passed"):
            payload["passed"] = False
            payload["detail"] = (
                f"container exit {completed.returncode} contradicts passed flag"
            )
        return DockerSystemScenarioRun(
            scenario_id=scenario_id,
            run_id=run_id,
            image=image,
            durable_root=durable_root,
            result=payload,
        )
    finally:
        _cleanup_container(container_name)
