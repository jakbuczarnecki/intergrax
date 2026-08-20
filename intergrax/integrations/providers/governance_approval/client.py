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
from intergrax.integrations.providers.governance_approval.config import (
    GovernanceApprovalIntegrationConfig,
)
from intergrax.integrations.providers.governance_approval.knowledge_read import (
    GovernanceApprovalNotFoundError,
    GovernanceApprovalReadClient,
    GovernanceApprovalReadError,
    GovernanceApprovalSnapshotV1,
)

_MAX_RESPONSE_BYTES: Final[int] = 65_536


def _response_status_code(response: httpx.Response) -> int:
    return int(response.status_code)


def _decode_json(response: httpx.Response) -> dict[str, object]:
    if len(response.content) > _MAX_RESPONSE_BYTES:
        raise GovernanceApprovalReadError("response_too_large")
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise GovernanceApprovalReadError("malformed_response") from exc
    if not isinstance(payload, dict):
        raise GovernanceApprovalReadError("malformed_response")
    return payload


def _parse_snapshot(payload: dict[str, object]) -> GovernanceApprovalSnapshotV1:
    try:
        return GovernanceApprovalSnapshotV1.model_validate(payload)
    except ValidationError as exc:
        raise GovernanceApprovalReadError("malformed_response") from exc


class HttpxGovernanceApprovalReadClient(GovernanceApprovalReadClient):
    """Real HTTP client for governance approval reads."""

    def __init__(
        self,
        *,
        config: GovernanceApprovalIntegrationConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=False,
        )
        self._owns_client = client is None

    async def read_governance_approval(
        self,
        *,
        subject_id: str,
    ) -> GovernanceApprovalSnapshotV1:
        cleaned = subject_id.strip()
        if not cleaned or cleaned != subject_id:
            raise IntegrationConfigurationError("subject_id_invalid")
        try:
            response = await self._client.get(f"/approvals/{cleaned}/status")
        except httpx.TimeoutException as exc:
            raise IntegrationDependencyError("governance_approval_timeout") from exc
        except httpx.HTTPError as exc:
            raise IntegrationDependencyError("governance_approval_dependency_failure") from exc

        status_code = _response_status_code(response)
        if status_code == 404:
            raise GovernanceApprovalNotFoundError("subject_not_found")
        if status_code >= 500 or status_code == 429:
            raise IntegrationDependencyError("governance_approval_dependency_failure")
        if status_code >= 400:
            raise IntegrationConfigurationError("governance_approval_configuration_failure")
        return _parse_snapshot(_decode_json(response))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
