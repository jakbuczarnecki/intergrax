# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from typing import Final

import httpx
from pydantic import ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.project_status.config import ProjectStatusIntegrationConfig
from intergrax.integrations.providers.project_status.knowledge_read import (
    ProjectStatusNotFoundError,
    ProjectStatusReadClient,
    ProjectStatusReadError,
    ProjectStatusSnapshotV1,
)

_MAX_RESPONSE_BYTES: Final[int] = 65_536


def _response_status_code(response: httpx.Response) -> int:
    return int(response.status_code)


def _decode_json(response: httpx.Response) -> dict[str, object]:
    if len(response.content) > _MAX_RESPONSE_BYTES:
        raise ProjectStatusReadError("response_too_large")
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise ProjectStatusReadError("malformed_response") from exc
    if not isinstance(payload, dict):
        raise ProjectStatusReadError("malformed_response")
    return payload


def _parse_snapshot(payload: dict[str, object]) -> ProjectStatusSnapshotV1:
    try:
        return ProjectStatusSnapshotV1.model_validate(payload)
    except ValidationError as exc:
        raise ProjectStatusReadError("malformed_response") from exc


class HttpxProjectStatusReadClient(ProjectStatusReadClient):
    """Real HTTP client for the controlled Project Status authority."""

    def __init__(
        self,
        *,
        config: ProjectStatusIntegrationConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=False,
        )
        self._owns_client = client is None

    async def read_project_status(self, *, project_id: str) -> ProjectStatusSnapshotV1:
        cleaned = project_id.strip()
        if not cleaned or cleaned != project_id:
            raise IntegrationConfigurationError("project_id_invalid")
        try:
            response = await self._client.get(f"/projects/{cleaned}/status")
        except httpx.TimeoutException as exc:
            raise IntegrationDependencyError("project_status_timeout") from exc
        except httpx.HTTPError as exc:
            raise IntegrationDependencyError("project_status_dependency_failure") from exc

        status_code = _response_status_code(response)
        if status_code == 404:
            raise ProjectStatusNotFoundError("project_not_found")
        if status_code >= 500 or status_code == 429:
            raise IntegrationDependencyError("project_status_dependency_failure")
        if status_code >= 400:
            raise IntegrationConfigurationError("project_status_configuration_failure")
        return _parse_snapshot(_decode_json(response))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
