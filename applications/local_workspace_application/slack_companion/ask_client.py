# © Artur Czarnecki. All rights reserved.

"""HTTP client for existing LKW Ask Workspace endpoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

from local_workspace_application.slack_companion.models import (
    SlackAskClientError,
    SlackAskHttpResponse,
    SlackWorkspaceListItem,
    SlackWorkspaceListResponse,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_LIMIT = 10


@dataclass(frozen=True, slots=True)
class SlackAskClientConfig:
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    limit: int = _DEFAULT_LIMIT


class WorkspaceAskHttpClient:
    """LKW-owned HTTP client; does not import WorkspaceAskService."""

    def __init__(
        self,
        config: SlackAskClientConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = config.base_url.rstrip("/") + "/"
        self._api_key = (config.api_key or "").strip() or None
        self._timeout = float(config.timeout_seconds)
        self._limit = int(config.limit)
        self._transport = transport

    def build_url(self, workspace_id: str) -> str:
        path = f"v1/local_workspace/workspaces/{workspace_id}/ask"
        return urljoin(self._base_url, path)

    def build_list_url(self) -> str:
        return urljoin(self._base_url, "v1/local_workspace/workspaces")

    async def list_workspaces(self, *, tenant_id: str) -> list[SlackWorkspaceListItem]:
        """Return tenant-scoped workspaces allowed for Ask (status=active)."""
        url = self.build_list_url()
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Tenant-Id": tenant_id.strip(),
        }
        if self._api_key is not None:
            headers["X-API-Key"] = self._api_key

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning(
                "slack_companion list_workspaces timeout kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="timeout") from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "slack_companion list_workspaces transport_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="transport_error") from exc

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "slack_companion list_workspaces http_error status=%s",
                response.status_code,
            )
            raise SlackAskClientError(kind=f"http_{response.status_code}")

        try:
            payload = response.json()
            parsed = SlackWorkspaceListResponse.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 — map parse failures to product error
            logger.warning(
                "slack_companion list_workspaces parse_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="parse_error") from exc

        return [
            item
            for item in parsed.workspaces
            if (item.status or "").strip().casefold() == "active"
            and (item.workspace_id or "").strip()
            and (item.name or "").strip()
        ]

    async def ask(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        question: str,
    ) -> SlackAskHttpResponse:
        url = self.build_url(workspace_id)
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Tenant-Id": tenant_id.strip(),
        }
        if self._api_key is not None:
            headers["X-API-Key"] = self._api_key

        body: dict[str, Any] = {
            "question": question,
            "limit": self._limit,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning("slack_companion ask timeout kind=%s", type(exc).__name__)
            raise SlackAskClientError(kind="timeout") from exc
        except httpx.HTTPError as exc:
            logger.warning("slack_companion ask transport_error kind=%s", type(exc).__name__)
            raise SlackAskClientError(kind="transport_error") from exc

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "slack_companion ask http_error status=%s",
                response.status_code,
            )
            raise SlackAskClientError(kind=f"http_{response.status_code}")

        try:
            payload = response.json()
            return SlackAskHttpResponse.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 — map parse failures to product error
            logger.warning("slack_companion ask parse_error kind=%s", type(exc).__name__)
            raise SlackAskClientError(kind="parse_error") from exc
