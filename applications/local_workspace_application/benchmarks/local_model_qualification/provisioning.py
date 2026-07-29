# © Artur Czarnecki. All rights reserved.

"""Docker Ollama runtime provisioning for local model qualification."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from local_workspace_application.benchmarks.local_model_qualification.config import (
    LocalModelQualificationConfig,
    enabled_model_names,
)
from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    PERSISTENT_MODEL_VOLUME,
    ModelProvisioningStatus,
    ProvisionedModel,
    ProvisioningResult,
)
from local_workspace_application.benchmarks.local_model_qualification.environment import (
    check_ollama_readiness,
    fetch_model_inventory,
    list_installed_models,
    pull_model,
)


class ProvisioningError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CommandRunner(Protocol):
    def __call__(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        shell: bool = False,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


class Clock(Protocol):
    def monotonic(self) -> float: ...


class Sleep(Protocol):
    def __call__(self, seconds: float) -> None: ...


@dataclass(frozen=True, slots=True)
class _CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _default_command_runner(
    args: list[str],
    *,
    cwd: str | None = None,
    shell: bool = False,
    capture_output: bool = True,
    text: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        shell=shell,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        check=False,
    )


def _compose_up_args(config: LocalModelQualificationConfig) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(config.compose_file_path),
        "up",
        "-d",
        config.ollama.compose_service,
    ]


def _container_inspect_args(container_name: str) -> list[str]:
    return [
        "docker",
        "inspect",
        "--format",
        "{{.State.Running}}",
        container_name,
    ]


def _start_docker_compose(
    config: LocalModelQualificationConfig,
    *,
    command_runner: CommandRunner,
) -> None:
    result = command_runner(
        _compose_up_args(config),
        cwd=str(config.repository_root),
        shell=False,
    )
    if result.returncode != 0:
        raise ProvisioningError("DOCKER_OLLAMA_START_FAILED")


def _verify_container_running(
    config: LocalModelQualificationConfig,
    *,
    command_runner: CommandRunner,
) -> None:
    result = command_runner(
        _container_inspect_args(config.ollama.container_name),
        cwd=str(config.repository_root),
        shell=False,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise ProvisioningError("DOCKER_OLLAMA_NOT_RUNNING")


def _wait_for_readiness(
    config: LocalModelQualificationConfig,
    *,
    client_factory: Callable[[str], Any] | None,
    clock: Clock,
    sleep: Sleep,
    http_get: Callable[[str], Any] | None = None,
) -> None:
    deadline = clock.monotonic() + config.ollama.startup_timeout_seconds
    while clock.monotonic() < deadline:
        if check_ollama_readiness(
            config.ollama,
            client_factory=client_factory,
            http_get=http_get,
        ):
            return
        sleep(config.ollama.readiness_poll_seconds)
    raise ProvisioningError("OLLAMA_READINESS_TIMEOUT")


def _pull_missing_models(
    config: LocalModelQualificationConfig,
    required_models: tuple[str, ...],
    installed: set[str],
    *,
    client_factory: Callable[[str], Any] | None,
    progress: Callable[[str], None],
) -> list[ProvisionedModel]:
    provisioned: list[ProvisionedModel] = []
    for model_name in required_models:
        if model_name in installed:
            progress(f"model={model_name} provisioning=ALREADY_AVAILABLE")
            provisioned.append(
                ProvisionedModel(model=model_name, status=ModelProvisioningStatus.ALREADY_AVAILABLE)
            )
            continue
        progress(f"model={model_name} provisioning=PULLING")
        try:
            pull_model(config.ollama, model_name, client_factory=client_factory)
        except Exception:
            progress(f"model={model_name} provisioning=FAILED")
            raise ProvisioningError("MODEL_PROVISIONING_INCOMPLETE") from None
        progress(f"model={model_name} provisioning=PULLED")
        provisioned.append(
            ProvisionedModel(model=model_name, status=ModelProvisioningStatus.PULLED)
        )
    return provisioned


def _verify_inventory_complete(required_models: tuple[str, ...], installed: set[str]) -> None:
    missing = sorted(set(required_models) - installed)
    if missing:
        for model_name in missing:
            print(f"model={model_name} provisioning=MISSING", flush=True)
        raise ProvisioningError("MODEL_PROVISIONING_INCOMPLETE")


def _verify_inventory_metadata(
    config: LocalModelQualificationConfig,
    required_models: tuple[str, ...],
    *,
    client_factory: Callable[[str], Any] | None,
) -> None:
    inventory = fetch_model_inventory(config.ollama, client_factory=client_factory)
    for model_name in required_models:
        record = inventory.get(model_name)
        if record is None or not record.digest.strip() or record.artifact_size_bytes <= 0:
            print(f"model={model_name} inventory_metadata=INCOMPLETE", flush=True)
            raise ProvisioningError("MODEL_INVENTORY_METADATA_INCOMPLETE")


def provision_ollama_runtime(
    config: LocalModelQualificationConfig,
    *,
    command_runner: CommandRunner | None = None,
    client_factory: Callable[[str], Any] | None = None,
    clock: Clock | None = None,
    sleep: Sleep | None = None,
    progress: Callable[[str], None] | None = None,
    http_get: Callable[[str], Any] | None = None,
) -> ProvisioningResult:
    """Start Docker Ollama and ensure every enabled model tag exists."""
    runner = command_runner or _default_command_runner
    monotonic = clock.monotonic if clock is not None else time.monotonic
    sleeper = sleep or time.sleep
    emit = progress or (lambda _message: None)

    class _Clock:
        @staticmethod
        def monotonic() -> float:
            return monotonic()

    required_models = enabled_model_names(config)

    _start_docker_compose(config, command_runner=runner)
    _verify_container_running(config, command_runner=runner)
    _wait_for_readiness(
        config,
        client_factory=client_factory,
        clock=_Clock(),
        sleep=sleeper,
        http_get=http_get,
    )

    installed = list_installed_models(config.ollama, client_factory=client_factory)
    provisioned_models = _pull_missing_models(
        config,
        required_models,
        installed,
        client_factory=client_factory,
        progress=emit,
    )

    installed_after = list_installed_models(config.ollama, client_factory=client_factory)
    _verify_inventory_complete(required_models, installed_after)
    _verify_inventory_metadata(config, required_models, client_factory=client_factory)

    return ProvisioningResult(
        runtime="docker",
        compose_file=config.ollama.compose_file,
        compose_service=config.ollama.compose_service,
        container_name=config.ollama.container_name,
        persistent_model_volume=PERSISTENT_MODEL_VOLUME,
        readiness_result="READY",
        required_models=required_models,
        models=tuple(provisioned_models),
    )
