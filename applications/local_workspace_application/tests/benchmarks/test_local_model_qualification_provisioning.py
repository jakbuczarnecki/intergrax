# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_workspace_application.benchmarks.local_model_qualification.config import load_config
from local_workspace_application.benchmarks.local_model_qualification.provisioning import (
    ProvisioningError,
    provision_ollama_runtime,
)

_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "local-model-qualification.toml"
)


@dataclass
class FakeClock:
    value: float = 0.0

    def monotonic(self) -> float:
        return self.value


@dataclass
class FakeSleep:
    clock: FakeClock
    interval: float

    def __call__(self, seconds: float) -> None:
        self.clock.value += seconds


@dataclass
class FakeOllamaClient:
    installed: set[str]
    pull_calls: list[str] = field(default_factory=list)
    pull_should_fail: set[str] = field(default_factory=set)

    def version(self) -> dict[str, str]:
        return {"version": "0.5.0"}

    def list(self) -> dict[str, list[dict[str, object]]]:
        return {
            "models": [
                {"name": name, "digest": f"sha256:{name}", "size": 1000}
                for name in sorted(self.installed)
            ]
        }

    def pull(self, *, model: str, stream: bool = False) -> None:
        if model in self.pull_should_fail:
            raise RuntimeError("pull failed")
        self.pull_calls.append(model)
        self.installed.add(model)


@dataclass
class RecordedCommand:
    args: list[str]
    cwd: str | None
    shell: bool


class FakeCommandRunner:
    def __init__(self, *, running: bool = True) -> None:
        self.calls: list[RecordedCommand] = []
        self.running = running

    def __call__(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        shell: bool = False,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
    ):
        self.calls.append(RecordedCommand(args=args, cwd=cwd, shell=shell))
        if args[:3] == ["docker", "compose", "-f"]:
            from types import SimpleNamespace

            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["docker", "inspect"]:
            from types import SimpleNamespace

            stdout = "true" if self.running else "false"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        from types import SimpleNamespace

        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _run_provision(
    *,
    installed: set[str] | None = None,
    pull_should_fail: set[str] | None = None,
    running: bool = True,
    ready: bool = True,
) -> tuple[Any, FakeCommandRunner, FakeOllamaClient]:
    config = load_config(_CONFIG)
    client = FakeOllamaClient(
        installed=set(installed or ()),
        pull_should_fail=set(pull_should_fail or ()),
    )
    runner = FakeCommandRunner(running=running)
    clock = FakeClock()
    if not ready:
        clock.value = 10_000.0
    result = provision_ollama_runtime(
        config,
        command_runner=runner,
        client_factory=lambda _host: client,
        clock=clock,
        sleep=FakeSleep(clock=clock, interval=config.ollama.readiness_poll_seconds),
        progress=lambda _message: None,
    )
    return result, runner, client


def test_compose_command_uses_argument_list() -> None:
    config = load_config(_CONFIG)
    runner = FakeCommandRunner()
    client = FakeOllamaClient(installed=set(config.models[i].name for i in range(5)))
    provision_ollama_runtime(
        config,
        command_runner=runner,
        client_factory=lambda _host: client,
        progress=lambda _message: None,
    )
    compose_call = runner.calls[0]
    assert compose_call.shell is False
    assert compose_call.args == [
        "docker",
        "compose",
        "-f",
        str(config.compose_file_path),
        "up",
        "-d",
        "ollama",
    ]
    assert compose_call.cwd == str(config.repository_root)


def test_container_name_inspected() -> None:
    _, runner, _ = _run_provision(installed={"qwen2.5:14b", "qwen3:14b", "llama3.1:8b", "gpt-oss:20b", "mistral-small3.2:24b"})
    inspect_call = runner.calls[1]
    assert inspect_call.args[-1] == "intergrax-ollama"


def test_readiness_timeout_classified() -> None:
    config = load_config(_CONFIG)
    runner = FakeCommandRunner()
    client = FakeOllamaClient(installed=set())

    class NeverReadyClient(FakeOllamaClient):
        def version(self) -> dict[str, str]:
            raise RuntimeError("not ready")

    never_ready = NeverReadyClient(installed=set())
    clock = FakeClock()
    with pytest.raises(ProvisioningError, match="OLLAMA_READINESS_TIMEOUT"):
        provision_ollama_runtime(
            config,
            command_runner=runner,
            client_factory=lambda _host: never_ready,
            clock=clock,
            sleep=FakeSleep(clock=clock, interval=config.ollama.readiness_poll_seconds),
            progress=lambda _message: None,
            http_get=lambda _url: (_ for _ in ()).throw(RuntimeError("not ready")),
        )


def test_all_available_models_cause_zero_pulls() -> None:
    required = {"qwen2.5:14b", "qwen3:14b", "llama3.1:8b", "gpt-oss:20b", "mistral-small3.2:24b"}
    _, _, client = _run_provision(installed=required)
    assert client.pull_calls == []


def test_missing_models_pulled_in_toml_order() -> None:
    required = ["qwen2.5:14b", "qwen3:14b", "llama3.1:8b", "gpt-oss:20b", "mistral-small3.2:24b"]
    _, _, client = _run_provision(installed={"qwen2.5:14b", "llama3.1:8b"})
    assert client.pull_calls == ["qwen3:14b", "gpt-oss:20b", "mistral-small3.2:24b"]


def test_only_missing_models_pulled() -> None:
    _, _, client = _run_provision(installed={"qwen2.5:14b", "qwen3:14b", "llama3.1:8b", "gpt-oss:20b", "mistral-small3.2:24b"})
    assert client.pull_calls == []


def test_incomplete_second_inventory_fails() -> None:
    config = load_config(_CONFIG)

    class IncompleteAfterPullClient(FakeOllamaClient):
        def __init__(self) -> None:
            super().__init__(installed={"qwen2.5:14b"})
            self._pulls = 0

        def pull(self, *, model: str, stream: bool = False) -> None:
            self._pulls += 1

        def list(self) -> dict[str, list[dict[str, object]]]:
            if self._pulls:
                return {"models": [{"name": "qwen2.5:14b", "digest": "sha256:a", "size": 1}]}
            return super().list()

    with pytest.raises(ProvisioningError, match="MODEL_PROVISIONING_INCOMPLETE"):
        provision_ollama_runtime(
            config,
            command_runner=FakeCommandRunner(),
            client_factory=lambda _host: IncompleteAfterPullClient(),
            progress=lambda _message: None,
        )


def test_pull_failure_fails_provisioning() -> None:
    with pytest.raises(ProvisioningError, match="MODEL_PROVISIONING_INCOMPLETE"):
        _run_provision(installed=set(), pull_should_fail={"qwen2.5:14b"})


def test_disabled_models_not_pulled() -> None:
    config = load_config(_CONFIG)
    disabled_config = config.model_copy(
        update={
            "models": tuple(
                model.model_copy(update={"enabled": model.name == "qwen2.5:14b"})
                for model in config.models
            )
        }
    )
    client = FakeOllamaClient(installed=set())
    provision_ollama_runtime(
        disabled_config,
        command_runner=FakeCommandRunner(),
        client_factory=lambda _host: client,
        progress=lambda _message: None,
    )
    assert client.pull_calls == ["qwen2.5:14b"]


def test_no_compose_down_invoked() -> None:
    _, runner, _ = _run_provision(
        installed={"qwen2.5:14b", "qwen3:14b", "llama3.1:8b", "gpt-oss:20b", "mistral-small3.2:24b"}
    )
    assert not any("down" in call.args for call in runner.calls)
    assert not any("volume" in " ".join(call.args) for call in runner.calls)


def test_docker_start_failure() -> None:
    config = load_config(_CONFIG)

    class FailingRunner(FakeCommandRunner):
        def __call__(self, args, **kwargs):
            if args[:3] == ["docker", "compose", "-f"]:
                from types import SimpleNamespace

                return SimpleNamespace(returncode=1, stdout="", stderr="fail")
            return super().__call__(args, **kwargs)

    with pytest.raises(ProvisioningError, match="DOCKER_OLLAMA_START_FAILED"):
        provision_ollama_runtime(
            config,
            command_runner=FailingRunner(),
            client_factory=lambda _host: FakeOllamaClient(installed=set()),
            progress=lambda _message: None,
        )


def test_container_not_running_failure() -> None:
    with pytest.raises(ProvisioningError, match="DOCKER_OLLAMA_NOT_RUNNING"):
        _run_provision(installed=set(), running=False)


def test_inventory_metadata_incomplete_fails() -> None:
    config = load_config(_CONFIG)
    client = FakeOllamaClient(installed={"qwen2.5:14b", "qwen3:14b", "llama3.1:8b", "gpt-oss:20b", "mistral-small3.2:24b"})

    def list_without_digest() -> dict[str, list[dict[str, object]]]:
        return {
            "models": [
                {"name": name, "size": 1000}
                for name in sorted(client.installed)
            ]
        }

    client.list = list_without_digest  # type: ignore[method-assign]
    with pytest.raises(ProvisioningError, match="MODEL_INVENTORY_METADATA_INCOMPLETE"):
        provision_ollama_runtime(
            config,
            command_runner=FakeCommandRunner(),
            client_factory=lambda _host: client,
            progress=lambda _message: None,
        )
