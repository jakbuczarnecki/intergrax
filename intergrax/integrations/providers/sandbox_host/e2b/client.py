# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""E2B transport client separated from Intergrax sandbox host adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.sandbox_host import SandboxExecResult
from intergrax.integrations.providers.sandbox_host.e2b.config import E2bSandboxHostConfig
from intergrax.integrations.providers.sandbox_host.e2b.errors import E2bSandboxHostError
from intergrax.integrations.providers.sandbox_host.e2b.network_policy import (
    E2bNetworkCreatePayload,
    E2bProviderNetworkState,
    parse_provider_network_state,
)


@dataclass(frozen=True, slots=True)
class E2bCreatedSandbox:
    """Provider sandbox identity returned at creation time."""

    sandbox_id: str


@runtime_checkable
class E2bSandboxApiClient(Protocol):
    """Provider transport surface used by ``E2bSandboxHostBackend``."""

    def create_sandbox(
        self,
        *,
        template_id: str,
        timeout_seconds: int,
        network: E2bNetworkCreatePayload | None = None,
    ) -> E2bCreatedSandbox:
        """Create a sandbox, optionally with creation-time network policy."""

    def get_sandbox_network_state(self, sandbox_id: str) -> E2bProviderNetworkState:
        """Read provider-applied network policy for attestation."""

    def exec_command(self, sandbox_id: str, command: str) -> SandboxExecResult:
        """Execute a shell command inside an existing sandbox."""

    def upload_file(self, sandbox_id: str, *, local_path: str, remote_path: str) -> str:
        """Upload a local file into the sandbox; return remote artifact id."""

    def destroy_sandbox(self, sandbox_id: str) -> None:
        """Terminate a provider sandbox."""

    def health(self) -> bool:
        """Control-plane health probe that does not create sandboxes."""


def _require_e2b_sdk() -> Any:
    try:
        from e2b import Sandbox
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "E2B sandbox host requires the optional e2b SDK. "
            "Install with: uv sync --extra integrations-e2b",
        ) from exc
    return Sandbox


def _network_payload_to_sdk(network: E2bNetworkCreatePayload) -> dict[str, object]:
    return {
        "allow_out": list(network.allow_out),
        "deny_out": lambda ctx: [ctx.all_traffic],
    }


def _network_info_to_state(network: Any) -> E2bProviderNetworkState:
    if network is None:
        raise E2bSandboxHostError("provider sandbox info missing network state")
    if hasattr(network, "allow_out"):
        return parse_provider_network_state(
            {
                "allow_out": list(network.allow_out or []),
                "deny_out": list(network.deny_out or []),
            },
        )
    if isinstance(network, Mapping):
        return parse_provider_network_state(network)
    raise E2bSandboxHostError("provider sandbox network state is malformed")


class SdkE2bSandboxApiClient:
    """Official E2B SDK client for create/exec/attestation."""

    def __init__(self, config: E2bSandboxHostConfig) -> None:
        self._config = config
        self._sandbox_cls = _require_e2b_sdk()
        self._handles: dict[str, Any] = {}

    def create_sandbox(
        self,
        *,
        template_id: str,
        timeout_seconds: int,
        network: E2bNetworkCreatePayload | None = None,
    ) -> E2bCreatedSandbox:
        opts: dict[str, object] = {
            "api_key": self._config.resolved_api_key(),
        }
        create_kwargs: dict[str, object] = {
            "template": template_id,
            "timeout": timeout_seconds,
            **opts,
        }
        if network is not None:
            create_kwargs["network"] = _network_payload_to_sdk(network)
        try:
            sandbox = self._sandbox_cls.create(**create_kwargs)
        except Exception as exc:  # noqa: BLE001 — provider boundary
            raise E2bSandboxHostError("E2B sandbox creation failed") from exc
        sandbox_id = str(getattr(sandbox, "sandbox_id", "") or "")
        if not sandbox_id:
            raise E2bSandboxHostError("E2B sandbox creation returned no sandbox id")
        self._handles[sandbox_id] = sandbox
        return E2bCreatedSandbox(sandbox_id=sandbox_id)

    def get_sandbox_network_state(self, sandbox_id: str) -> E2bProviderNetworkState:
        sandbox = self._require_handle(sandbox_id)
        try:
            info = sandbox.get_info()
        except Exception as exc:  # noqa: BLE001 — provider boundary
            raise E2bSandboxHostError("E2B sandbox info read failed") from exc
        return _network_info_to_state(getattr(info, "network", None))

    def exec_command(self, sandbox_id: str, command: str) -> SandboxExecResult:
        sandbox = self._require_handle(sandbox_id)
        try:
            result = sandbox.commands.run(command)
        except Exception as exc:  # noqa: BLE001 — provider boundary
            raise E2bSandboxHostError("E2B sandbox command execution failed") from exc
        exit_code = int(getattr(result, "exit_code", 1) or 1)
        stdout = str(getattr(result, "stdout", "") or "")
        stderr = str(getattr(result, "stderr", "") or "")
        return SandboxExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    def upload_file(self, sandbox_id: str, *, local_path: str, remote_path: str) -> str:
        sandbox = self._require_handle(sandbox_id)
        content = Path(local_path).read_bytes()
        try:
            sandbox.files.write(remote_path, content)
        except Exception as exc:  # noqa: BLE001 — provider boundary
            raise E2bSandboxHostError("E2B sandbox artifact upload failed") from exc
        return remote_path

    def destroy_sandbox(self, sandbox_id: str) -> None:
        sandbox = self._handles.pop(sandbox_id, None)
        if sandbox is None:
            return
        try:
            sandbox.kill()
        except Exception:
            return

    def health(self) -> bool:
        return bool(self._config.resolved_api_key())

    def _require_handle(self, sandbox_id: str) -> Any:
        sandbox = self._handles.get(sandbox_id)
        if sandbox is None:
            raise E2bSandboxHostError(f"unknown E2B sandbox session: {sandbox_id}")
        return sandbox


class LegacyInjectedE2bSandboxApiClient:
    """Adapter for legacy injected clients used by catalog compatibility tests."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def create_sandbox(
        self,
        *,
        template_id: str,
        timeout_seconds: int,
        network: E2bNetworkCreatePayload | None = None,
    ) -> E2bCreatedSandbox:
        if network is not None and hasattr(self._client, "create_sandbox_with_network"):
            payload = self._client.create_sandbox_with_network(
                template_id=template_id,
                timeout_seconds=timeout_seconds,
                network=network,
            )
        else:
            payload = self._client.create_session()
        if isinstance(payload, E2bCreatedSandbox):
            return payload
        data = dict(payload or {})
        sandbox_id = str(data.get("sandbox_id") or data.get("session_id") or data.get("id") or "")
        if not sandbox_id:
            raise E2bSandboxHostError("legacy E2B client returned no sandbox id")
        return E2bCreatedSandbox(sandbox_id=sandbox_id)

    def get_sandbox_network_state(self, sandbox_id: str) -> E2bProviderNetworkState:
        if hasattr(self._client, "get_sandbox_network_state"):
            state = self._client.get_sandbox_network_state(sandbox_id)
            if isinstance(state, E2bProviderNetworkState):
                return state
            return parse_provider_network_state(dict(state or {}))
        raise E2bSandboxHostError("legacy E2B client does not support network attestation")

    def exec_command(self, sandbox_id: str, command: str) -> SandboxExecResult:
        payload = self._client.exec(sandbox_id, command)
        if isinstance(payload, SandboxExecResult):
            return payload
        data = dict(payload or {})
        return SandboxExecResult(
            exit_code=int(data.get("exit_code") or data.get("exitCode") or 0),
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
        )

    def upload_file(self, sandbox_id: str, *, local_path: str, remote_path: str) -> str:
        payload = self._client.upload_artifact(
            sandbox_id,
            local_path=local_path,
            remote_name=remote_path,
        )
        if isinstance(payload, str):
            return payload
        data = dict(payload or {})
        return str(data.get("artifact_id") or data.get("id") or remote_path)

    def destroy_sandbox(self, sandbox_id: str) -> None:
        if hasattr(self._client, "destroy_sandbox"):
            self._client.destroy_sandbox(sandbox_id)

    def health(self) -> bool:
        if hasattr(self._client, "health"):
            return bool(self._client.health())
        return True


def build_e2b_sandbox_api_client(
    config: E2bSandboxHostConfig,
    *,
    client: Any | None = None,
) -> E2bSandboxApiClient:
    if client is not None:
        return LegacyInjectedE2bSandboxApiClient(client)
    return SdkE2bSandboxApiClient(config)
