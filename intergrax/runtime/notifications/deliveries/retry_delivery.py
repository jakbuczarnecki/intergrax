# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Retry wrapper for notification delivery (Appendix B.13)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from intergrax.runtime.notifications.deliveries.delivery_ledger_protocol import DeliveryLedger
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery

logger = logging.getLogger(__name__)


class RetryingNotificationDelivery(NotificationDelivery):
    """
    Wraps a transport with bounded retries, backoff, and optional receipt ledger.

    After ``max_attempts`` failures the error is re-raised and a dead-letter
    receipt is recorded when a ledger is configured.
    """

    def __init__(
        self,
        inner: NotificationDelivery,
        *,
        max_attempts: int = 3,
        base_delay_seconds: float = 0.05,
        ledger: Optional[DeliveryLedger] = None,
        channel: str = "webhook",
        task_id_from_payload: Callable[[Dict[str, Any]], str] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._inner = inner
        self._max_attempts = max_attempts
        self._base_delay_seconds = base_delay_seconds
        self._ledger = ledger
        self._channel = channel
        self._task_id_from_payload = task_id_from_payload or _default_task_id

    async def deliver(
        self,
        *,
        destination: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> DeliveryReceipt | None:
        task_id = self._task_id_from_payload(payload)
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                await self._inner.deliver(
                    destination=destination,
                    payload=payload,
                    headers=headers,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "notification delivery attempt failed attempt=%s/%s task=%s error=%s",
                    attempt,
                    self._max_attempts,
                    task_id,
                    exc,
                )
                if attempt < self._max_attempts:
                    await asyncio.sleep(self._base_delay_seconds * attempt)
                    continue
                if self._ledger is not None:
                    return self._ledger.record_dead_letter(
                        destination=destination,
                        task_id=task_id,
                        channel=self._channel,
                        attempts=attempt,
                        last_error=str(exc),
                        payload_summary=_payload_summary(payload),
                    )
                raise
            else:
                if self._ledger is not None:
                    return self._ledger.record_success(
                        destination=destination,
                        task_id=task_id,
                        channel=self._channel,
                        attempts=attempt,
                        payload_summary=_payload_summary(payload),
                    )
                return None
        assert last_error is not None
        raise last_error


def _default_task_id(payload: Dict[str, Any]) -> str:
    for key in ("task_id", "run_id", "id"):
        value = payload.get(key)
        if value:
            return str(value)
    return "unknown"


def _payload_summary(payload: Dict[str, Any]) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    for key in ("task_id", "run_id", "event_type", "title"):
        if key in payload and payload[key] is not None:
            summary[key] = str(payload[key])
    return summary
