# © Artur Czarnecki. All rights reserved.

"""E2B SDK exec result mapping and hosted-session regression tests (P0-3A-01)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from intergrax.integrations.contracts.sandbox_host import SandboxExecResult
from intergrax.integrations.providers.sandbox_host.e2b.backend import E2bSandboxHostBackend
from intergrax.integrations.providers.sandbox_host.e2b.client import (
    LegacyInjectedE2bSandboxApiClient,
    SdkE2bSandboxApiClient,
)
from intergrax.integrations.providers.sandbox_host.e2b.config import E2bSandboxHostConfig
from intergrax.integrations.providers.sandbox_host.e2b.errors import E2bSandboxHostError
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession

pytestmark = pytest.mark.unit


@dataclass
class _FakeCommandResult:
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""


@dataclass
class _FakeCommands:
    last_command: str = ""
    results: list[_FakeCommandResult] = field(default_factory=list)
    _index: int = 0

    def run(self, command: str) -> _FakeCommandResult:
        self.last_command = command
        if not self.results:
            return _FakeCommandResult(exit_code=0)
        result = self.results[min(self._index, len(self.results) - 1)]
        self._index += 1
        return result


@dataclass
class _FakeSdkSandbox:
    sandbox_id: str
    commands: _FakeCommands


def _sdk_client_with_sandbox(
    sandbox: _FakeSdkSandbox,
) -> tuple[SdkE2bSandboxApiClient, str]:
    config = E2bSandboxHostConfig(api_key="test-key", template_id="base")
    client = SdkE2bSandboxApiClient(config)
    client._handles[sandbox.sandbox_id] = sandbox  # noqa: SLF001 — inject fake SDK handle
    return client, sandbox.sandbox_id


@pytest.mark.parametrize(
    ("provider_exit_code", "expected"),
    [
        (0, 0),
        (1, 1),
        (17, 17),
        (-1, -1),
    ],
)
def test_sdk_exec_preserves_provider_exit_code(provider_exit_code: int, expected: int) -> None:
    sandbox = _FakeSdkSandbox(
        sandbox_id="sbx-exit",
        commands=_FakeCommands(results=[_FakeCommandResult(exit_code=provider_exit_code)]),
    )
    client, sandbox_id = _sdk_client_with_sandbox(sandbox)
    result = client.exec_command(sandbox_id, "echo ok")
    assert result.exit_code == expected


def test_sdk_exec_missing_exit_code_uses_failure_fallback() -> None:
    sandbox = _FakeSdkSandbox(
        sandbox_id="sbx-missing",
        commands=_FakeCommands(results=[_FakeCommandResult(exit_code=None)]),
    )
    client, sandbox_id = _sdk_client_with_sandbox(sandbox)
    result = client.exec_command(sandbox_id, "echo ok")
    assert result.exit_code == 1


def test_sdk_exec_preserves_stdout_stderr() -> None:
    sandbox = _FakeSdkSandbox(
        sandbox_id="sbx-io",
        commands=_FakeCommands(
            results=[_FakeCommandResult(exit_code=0, stdout="out-data", stderr="err-data")],
        ),
    )
    client, sandbox_id = _sdk_client_with_sandbox(sandbox)
    result = client.exec_command(sandbox_id, "echo ok")
    assert result.stdout == "out-data"
    assert result.stderr == "err-data"


def test_sdk_exec_invalid_exit_code_raises_provider_error() -> None:
    sandbox = _FakeSdkSandbox(
        sandbox_id="sbx-bad",
        commands=_FakeCommands(results=[_FakeCommandResult(exit_code="abc")]),  # type: ignore[arg-type]
    )
    client, sandbox_id = _sdk_client_with_sandbox(sandbox)
    with pytest.raises(E2bSandboxHostError, match="invalid exit code"):
        client.exec_command(sandbox_id, "echo ok")


def test_sdk_exec_command_failure_raises_canonical_error() -> None:
    class _BrokenCommands:
        def run(self, command: str) -> _FakeCommandResult:
            raise RuntimeError("provider transport failed")

    sandbox = _FakeSdkSandbox(sandbox_id="sbx-fail", commands=_BrokenCommands())  # type: ignore[arg-type]
    client, sandbox_id = _sdk_client_with_sandbox(sandbox)
    with pytest.raises(E2bSandboxHostError, match="command execution failed"):
        client.exec_command(sandbox_id, "echo ok")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"exit_code": 0}, 0),
        ({"exitCode": 0}, 0),
        ({}, 0),
        ({"exit_code": 17}, 17),
    ],
)
def test_legacy_exec_preserves_numeric_zero_and_explicit_codes(
    payload: dict[str, Any],
    expected: int,
) -> None:
    class _LegacyClient:
        def exec(self, sandbox_id: str, command: str) -> dict[str, Any]:
            return {
                **payload,
                "stdout": "legacy-out",
                "stderr": "legacy-err",
            }

    client = LegacyInjectedE2bSandboxApiClient(_LegacyClient())
    result = client.exec_command("legacy-sbx", "echo ok")
    assert result.exit_code == expected
    assert result.stdout == "legacy-out"
    assert result.stderr == "legacy-err"


def _hosted_stack_with_exit_code(exit_code: int) -> HostedSandboxSession:
    sandbox = _FakeSdkSandbox(
        sandbox_id="sbx-hosted",
        commands=_FakeCommands(results=[_FakeCommandResult(exit_code=exit_code, stdout="done")]),
    )
    sdk_client, _ = _sdk_client_with_sandbox(sandbox)
    config = E2bSandboxHostConfig(api_key="test-key", template_id="base")
    backend = E2bSandboxHostBackend(client=sdk_client, config=config)
    return HostedSandboxSession(
        session_id=sandbox.sandbox_id,
        backend=backend,
        tenant_id="tenant",
        task_id="task",
        allowed_operations=frozenset({"run_python"}),
    )


def test_hosted_session_success_on_sdk_exit_zero() -> None:
    session = _hosted_stack_with_exit_code(0)
    result = session.execute("run_python", {"code": "print('ok')"})
    assert result.success is True
    assert result.output is not None
    assert result.output["exit_code"] == 0
    assert result.output["stdout"] == "done"


def test_hosted_session_failure_on_sdk_exit_nonzero() -> None:
    session = _hosted_stack_with_exit_code(17)
    result = session.execute("run_python", {"code": "raise SystemExit(17)"})
    assert result.success is False
    assert result.output is not None
    assert result.output["exit_code"] == 17


def test_backend_destroy_session_invokes_provider_cleanup() -> None:
    destroyed: list[str] = []

    class _TrackingSdkClient:
        def destroy_sandbox(self, sandbox_id: str) -> None:
            destroyed.append(sandbox_id)

        def exec_command(self, sandbox_id: str, command: str) -> SandboxExecResult:
            return SandboxExecResult(exit_code=0, stdout="", stderr="")

        def create_sandbox(self, **kwargs: object) -> Any:
            raise NotImplementedError

        def get_sandbox_network_state(self, sandbox_id: str) -> Any:
            raise NotImplementedError

        def upload_file(self, sandbox_id: str, *, local_path: str, remote_path: str) -> str:
            raise NotImplementedError

        def health(self) -> bool:
            return True

    config = E2bSandboxHostConfig(api_key="test-key", template_id="base")
    backend = E2bSandboxHostBackend(client=_TrackingSdkClient(), config=config)
    backend.destroy_session("sbx-cleanup")
    assert destroyed == ["sbx-cleanup"]
