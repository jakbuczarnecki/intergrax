# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""HTTP webhook delivery backend (opt-in, injectable client for tests)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from intergrax.runtime.notifications.delivery_contract import NotificationDelivery

logger = logging.getLogger(__name__)


class HttpWebhookDelivery(NotificationDelivery):
    """POST JSON payloads to webhook URLs via httpx."""

    def __init__(
        self,
        *,
        client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._client = client
        self._timeout_seconds = timeout_seconds

    async def deliver(
        self,
        *,
        destination: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if not destination:
            return
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            response = await client.post(
                destination,
                json=payload,
                headers=headers or {"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "webhook delivery failed destination=%s error=%s",
                destination,
                exc,
            )
            raise
        finally:
            if owns_client:
                await client.aclose()
