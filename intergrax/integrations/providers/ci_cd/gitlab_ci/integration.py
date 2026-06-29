# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gitlab Ci ci cd integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GITLAB_CI_CI_CD_PROVIDER_ID = "gitlab_ci"


class GitlabCiCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gitlab Ci ci cd integration."""

    pass


@runtime_checkable
class GitlabCiCiCdClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GitlabCiCiCdIntegration(CiCdIntegrationContract):
    """
    Single public Gitlab Ci ci cd entrypoint.

    Legacy catalog factory (create_gitlab_ci_ci_cd) delegates to this class.
    """

    config: GitlabCiCiCdIntegrationConfig = GitlabCiCiCdIntegrationConfig()
    _client: GitlabCiCiCdClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> GitlabCiCiCdIntegration:
        integration = cls.for_provider(
            provider_id=GITLAB_CI_CI_CD_PROVIDER_ID,
            display_name="Gitlab Ci",
            config=GitlabCiCiCdIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Gitlab Ci integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: GitlabCiCiCdClient,
        *,
        enabled: bool = False,
    ) -> GitlabCiCiCdIntegration:
        integration = cls.for_provider(
            provider_id=GITLAB_CI_CI_CD_PROVIDER_ID,
            display_name="Gitlab Ci",
            config=GitlabCiCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GitlabCiCiCdClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CiCdBackend.register(GitlabCiCiCdIntegration)
