# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Devops issue tracker integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.contracts.issue_tracker import IssueRecord, IssueSearchResult, IssueTracker
from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID = "azure_devops"


class AzureDevopsIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Devops issue tracker integration."""

    pass


@runtime_checkable
class AzureDevopsIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AzureDevopsIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Single public Azure Devops issue tracker entrypoint.

    Legacy catalog factory (create_azure_devops_issue_tracker) delegates to this class.
    """

    config: AzureDevopsIssueTrackerIntegrationConfig = AzureDevopsIssueTrackerIntegrationConfig()
    _client: AzureDevopsIssueTrackerClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> AzureDevopsIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Azure Devops",
            config=AzureDevopsIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration


    def search_issues(self, query: str, *, limit: int = 20) -> IssueSearchResult:
        return self._require_runtime().search_issues(query, limit=limit)

    def get_issue(self, issue_id: str) -> IssueRecord | None:
        return self._require_runtime().get_issue(issue_id)

    def create_issue(self, *, title: str, body: str = "", labels: Sequence[str] = ()) -> IssueRecord:
        return self._require_runtime().create_issue(title=title, body=body, labels=labels)


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


    @classmethod
    def from_client(
        cls,
        client: AzureDevopsIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> AzureDevopsIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Azure Devops",
            config=AzureDevopsIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AzureDevopsIssueTrackerClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

IssueTracker.register(AzureDevopsIssueTrackerIntegration)
