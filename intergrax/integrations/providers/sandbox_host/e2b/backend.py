# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Security-qualified E2B sandbox host backend."""

from __future__ import annotations

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.sandbox_host import (
    SandboxArtifact,
    SandboxExecResult,
    SandboxHostBackend,
    SandboxSession,
)
from intergrax.integrations.providers.sandbox_host.e2b.client import E2bSandboxApiClient
from intergrax.integrations.providers.sandbox_host.e2b.config import E2bSandboxHostConfig
from intergrax.integrations.providers.sandbox_host.e2b.errors import E2bSandboxSecurityError
from intergrax.integrations.providers.sandbox_host.e2b.integration import E2B_SANDBOX_HOST_PROVIDER_ID
from intergrax.integrations.providers.sandbox_host.e2b.network_policy import (
    intergrax_hosts_to_e2b_allow_out,
    provider_state_to_enforced_allowlist,
)
from intergrax.runtime.sandbox.contracts import (
    SandboxSecurityCapabilities,
    SandboxSecurityConfigurable,
    SandboxSecurityRequirements,
    SandboxSessionSecurityEvidenceProvider,
)


class E2bSandboxHostBackend:
    """Real E2B provider adapter with session-scoped security attestation."""

    def __init__(
        self,
        *,
        client: E2bSandboxApiClient,
        config: E2bSandboxHostConfig,
    ) -> None:
        self._client = client
        self._config = config
        self._session_security: dict[str, SandboxSecurityCapabilities] = {}

    def create_session(self) -> SandboxSession:
        created = self._client.create_sandbox(
            template_id=self._config.resolved_template_id(),
            timeout_seconds=self._config.sandbox_timeout_seconds,
            network=None,
        )
        return SandboxSession(session_id=created.sandbox_id, status="running")

    def create_session_with_security(
        self,
        requirements: SandboxSecurityRequirements,
    ) -> SandboxSession:
        if requirements.isolation_tier != "cloud":
            raise E2bSandboxSecurityError(
                f"E2B adapter requires cloud isolation tier; got {requirements.isolation_tier}",
            )
        if requirements.network_egress != "allowlist":
            raise E2bSandboxSecurityError(
                f"E2B adapter V1 supports allowlist egress only; got {requirements.network_egress}",
            )
        allowlist = requirements.network_egress_allowlist
        network_payload = intergrax_hosts_to_e2b_allow_out(allowlist)
        created = self._client.create_sandbox(
            template_id=self._config.resolved_template_id(),
            timeout_seconds=self._config.sandbox_timeout_seconds,
            network=network_payload,
        )
        try:
            provider_state = self._client.get_sandbox_network_state(created.sandbox_id)
            enforced = provider_state_to_enforced_allowlist(
                provider_state,
                requested=allowlist,  # type: ignore[arg-type]
            )
        except E2bSandboxSecurityError:
            self._client.destroy_sandbox(created.sandbox_id)
            raise
        except Exception as exc:  # noqa: BLE001 — provider boundary
            self._client.destroy_sandbox(created.sandbox_id)
            raise E2bSandboxSecurityError("E2B sandbox attestation failed") from exc

        self._session_security[created.sandbox_id] = SandboxSecurityCapabilities(
            isolation_tier="cloud",
            provider_id=E2B_SANDBOX_HOST_PROVIDER_ID,
            network_egress_deny_enforced=False,
            network_egress_allowlist_enforced=True,
            enforced_network_hosts=enforced,
        )
        return SandboxSession(session_id=created.sandbox_id, status="running")

    def session_security_capabilities(self, session_id: str) -> SandboxSecurityCapabilities:
        capabilities = self._session_security.get(session_id)
        if capabilities is None:
            raise E2bSandboxSecurityError(
                f"no security evidence for E2B sandbox session: {session_id}",
            )
        return capabilities

    def exec(self, session_id: str, command: str) -> SandboxExecResult:
        return self._client.exec_command(session_id, command)

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        artifact_id = self._client.upload_file(
            session_id,
            local_path=local_path,
            remote_path=remote_name,
        )
        return SandboxArtifact(artifact_id=artifact_id, uri=remote_name)

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=E2B_SANDBOX_HOST_PROVIDER_ID)


SandboxHostBackend.register(E2bSandboxHostBackend)
SandboxSecurityConfigurable.register(E2bSandboxHostBackend)
SandboxSessionSecurityEvidenceProvider.register(E2bSandboxHostBackend)
