# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PagerDuty Events API v2 client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.pagerduty.config import PagerDutyIntegrationConfig


class PagerDutyEventsClient:
    """PagerDuty Events API v2 — trigger/acknowledge/resolve incidents."""

    def __init__(self, config: PagerDutyIntegrationConfig, *, http_client: Any) -> None:
        if not config.routing_key:
            raise IntegrationConfigurationError(
                "PagerDuty routing_key is required (INTERGRAX_PAGERDUTY_ROUTING_KEY)"
            )
        self._config = config
        self._http = http_client

    @property
    def config(self) -> PagerDutyIntegrationConfig:
        return self._config

    def trigger_incident(
        self,
        *,
        summary: str,
        severity: str = "error",
        source: str = "intergrax",
        custom_details: Optional[Mapping[str, Any]] = None,
        dedup_key: Optional[str] = None,
    ) -> str:
        payload: dict[str, object] = {
            "routing_key": self._config.routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": summary,
                "severity": severity,
                "source": source,
                "custom_details": dict(custom_details or {}),
            },
        }
        if dedup_key:
            payload["dedup_key"] = dedup_key
        response = self._http.post("/v2/enqueue", json=payload)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return str(data.get("dedup_key") or data.get("message") or "")
        return ""

    def send_notification(self, *, subject: str, body: str, task_id: str) -> None:
        self.trigger_incident(
            summary=subject or task_id,
            custom_details={"body": body, "task_id": task_id},
            dedup_key=task_id or None,
        )

    def health(self) -> bool:
        try:
            response = self._http.get("/health")
            return int(response.status_code) < 400  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — health probe surface
            return False
