# © Artur Czarnecki. All rights reserved.

"""HTTP client for existing LKW Ask Workspace endpoint."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

from intergrax.integrations.contracts.conversation_channel import (
    ConversationAttachmentContent,
)
from local_workspace_application.slack_companion.models import (
    SlackAskClientError,
    SlackAskHttpResponse,
    SlackManagedFileBatchResponse,
    SlackSourceCandidateAcceptResponse,
    SlackSourceCandidateListItem,
    SlackSourceCandidateListResponse,
    SlackSourceListItem,
    SlackSourceListResponse,
    SlackWorkspaceCreateResponse,
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

    def build_sources_url(self, workspace_id: str) -> str:
        path = f"v1/local_workspace/workspaces/{workspace_id}/sources"
        return urljoin(self._base_url, path)

    def build_source_candidates_url(self, workspace_id: str) -> str:
        path = f"v1/local_workspace/workspaces/{workspace_id}/source-candidates"
        return urljoin(self._base_url, path)

    def build_accept_source_candidate_url(
        self, workspace_id: str, candidate_id: str
    ) -> str:
        path = (
            f"v1/local_workspace/workspaces/{workspace_id}"
            f"/knowledge/source-candidates/{candidate_id}"
        )
        return urljoin(self._base_url, path)

    def build_managed_files_url(self, workspace_id: str) -> str:
        path = f"v1/local_workspace/workspaces/{workspace_id}/knowledge/files"
        return urljoin(self._base_url, path)

    async def upload_managed_files(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        attachments: Sequence[ConversationAttachmentContent],
    ) -> SlackManagedFileBatchResponse:
        tenant = tenant_id.strip()
        workspace = workspace_id.strip()
        idem = idempotency_key.strip()
        if not tenant or not workspace or not idem:
            raise SlackAskClientError(kind="parse_error")
        if not attachments:
            raise SlackAskClientError(kind="parse_error")

        url = self.build_managed_files_url(workspace)
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Tenant-Id": tenant,
            "Idempotency-Key": idem,
        }
        if self._api_key is not None:
            headers["X-API-Key"] = self._api_key

        files = [
            (
                "files",
                (
                    attachment.file_name,
                    attachment.body,
                    attachment.content_type,
                ),
            )
            for attachment in attachments
        ]

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(url, files=files, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning(
                "slack_companion upload_managed_files timeout kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="timeout") from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "slack_companion upload_managed_files transport_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="transport_error") from exc

        if response.status_code < 200 or response.status_code >= 300:
            status = response.status_code
            logger.warning(
                "slack_companion upload_managed_files http_error status=%s",
                status,
            )
            raise SlackAskClientError(kind=f"http_{status}")

        try:
            payload = response.json()
            parsed = SlackManagedFileBatchResponse.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion upload_managed_files parse_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="parse_error") from exc

        if (parsed.workspace_id or "").strip() != workspace:
            raise SlackAskClientError(kind="parse_error")
        return parsed

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

    async def list_sources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[SlackSourceListItem]:
        """Return tenant-scoped safe source summaries for one workspace."""
        url = self.build_sources_url(workspace_id.strip())
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
                "slack_companion list_sources timeout kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="timeout") from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "slack_companion list_sources transport_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="transport_error") from exc

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "slack_companion list_sources http_error status=%s",
                response.status_code,
            )
            raise SlackAskClientError(kind=f"http_{response.status_code}")

        try:
            payload = response.json()
            parsed = SlackSourceListResponse.model_validate(payload)
        except Exception as exc:
            logger.warning(
                "slack_companion list_sources parse_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="parse_error") from exc

        return list(parsed.sources)

    async def list_source_candidates(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[SlackSourceCandidateListItem]:
        tenant = tenant_id.strip()
        workspace = workspace_id.strip()
        if not tenant or not workspace:
            raise SlackAskClientError(kind="parse_error")

        url = self.build_source_candidates_url(workspace)
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Tenant-Id": tenant,
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
                "slack_companion list_source_candidates timeout kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="timeout") from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "slack_companion list_source_candidates transport_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="transport_error") from exc

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "slack_companion list_source_candidates http_error status=%s",
                response.status_code,
            )
            raise SlackAskClientError(kind=f"http_{response.status_code}")

        try:
            payload = response.json()
            parsed = SlackSourceCandidateListResponse.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion list_source_candidates parse_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="parse_error") from exc

        if (parsed.workspace_id or "").strip() != workspace:
            raise SlackAskClientError(kind="parse_error")
        return list(parsed.candidates)

    async def accept_source_candidate(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        candidate_id: str,
        idempotency_key: str,
    ) -> SlackSourceCandidateAcceptResponse:
        tenant = tenant_id.strip()
        workspace = workspace_id.strip()
        candidate = candidate_id.strip()
        idem = idempotency_key.strip()
        if not tenant or not workspace or not candidate or not idem:
            raise SlackAskClientError(kind="parse_error")

        url = self.build_accept_source_candidate_url(workspace, candidate)
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Tenant-Id": tenant,
            "Idempotency-Key": idem,
        }
        if self._api_key is not None:
            headers["X-API-Key"] = self._api_key

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(url, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning(
                "slack_companion accept_source_candidate timeout kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="timeout") from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "slack_companion accept_source_candidate transport_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="transport_error") from exc

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "slack_companion accept_source_candidate http_error status=%s",
                response.status_code,
            )
            raise SlackAskClientError(kind=f"http_{response.status_code}")

        try:
            payload = response.json()
            parsed = SlackSourceCandidateAcceptResponse.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion accept_source_candidate parse_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="parse_error") from exc

        if (parsed.workspace_id or "").strip() != workspace:
            raise SlackAskClientError(kind="parse_error")
        if (parsed.candidate_id or "").strip() != candidate:
            raise SlackAskClientError(kind="parse_error")
        return parsed

    async def create_workspace(
        self,
        *,
        tenant_id: str,
        name: str,
    ) -> SlackWorkspaceCreateResponse:
        url = self.build_list_url()
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Tenant-Id": tenant_id.strip(),
        }
        if self._api_key is not None:
            headers["X-API-Key"] = self._api_key

        body: dict[str, Any] = {"name": name}

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning(
                "slack_companion create_workspace timeout kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="timeout") from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "slack_companion create_workspace transport_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="transport_error") from exc

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "slack_companion create_workspace http_error status=%s",
                response.status_code,
            )
            raise SlackAskClientError(kind=f"http_{response.status_code}")

        try:
            payload = response.json()
            parsed = SlackWorkspaceCreateResponse.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion create_workspace parse_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="parse_error") from exc

        if not (parsed.workspace_id or "").strip() or not (parsed.name or "").strip():
            raise SlackAskClientError(kind="parse_error")
        return parsed

    async def delete_workspace(self, *, tenant_id: str, workspace_id: str) -> None:
        path = f"v1/local_workspace/workspaces/{workspace_id.strip()}"
        url = urljoin(self._base_url, path)
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
                response = await client.delete(url, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning(
                "slack_companion delete_workspace timeout kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="timeout") from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "slack_companion delete_workspace transport_error kind=%s",
                type(exc).__name__,
            )
            raise SlackAskClientError(kind="transport_error") from exc

        if response.status_code == 204:
            return
        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "slack_companion delete_workspace http_error status=%s",
                response.status_code,
            )
            raise SlackAskClientError(kind=f"http_{response.status_code}")
        # Non-204 success with body is unexpected for this contract.
        raise SlackAskClientError(kind="parse_error")

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
