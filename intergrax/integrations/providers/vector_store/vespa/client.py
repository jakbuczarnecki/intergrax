# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vespa document API client — feed and YQL query."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.vespa.config import VespaIntegrationConfig


class VespaRestClient:
    """Minimal Vespa document/v1 feed and search client."""

    def __init__(self, config: VespaIntegrationConfig, *, http_client: Any) -> None:
        if not config.base_url:
            raise IntegrationConfigurationError("Vespa base_url is required (INTERGRAX_VESPA_URL)")
        self._config = config
        self._http = http_client

    @property
    def config(self) -> VespaIntegrationConfig:
        return self._config

    def _doc_path(self, doc_id: str) -> str:
        return (
            f"/document/v1/{self._config.tenant_id}/{self._config.collection}/docid/{doc_id}"
        )

    def feed_document(self, *, doc_id: str, fields: Mapping[str, Any]) -> str:
        response = self._http.post(
            self._doc_path(doc_id),
            json={"fields": dict(fields)},
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("id") or doc_id)
        return doc_id

    def query_yql(
        self,
        yql: str,
        *,
        hits: int = 10,
        ranking: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, object] = {
            "yql": yql,
            "hits": max(1, int(hits)),
        }
        if ranking:
            body["ranking"] = ranking
        response = self._http.post("/search/", json=body)
        response.raise_for_status()
        payload = response.json()
        root = payload.get("root") if isinstance(payload, dict) else {}
        children = root.get("children") if isinstance(root, dict) else []
        results: list[dict[str, Any]] = []
        for child in children or []:
            if isinstance(child, dict):
                results.append(child)
        return results

    def delete_document(self, doc_id: str) -> None:
        response = self._http.delete(self._doc_path(doc_id))
        if response.status_code not in {200, 404}:
            response.raise_for_status()

    def count_documents(self) -> int:
        rows = self.query_yql(
            f"select * from sources {self._config.collection} where true limit 0",
            hits=0,
        )
        _ = rows
        response = self._http.post(
            "/search/",
            json={"yql": f"select * from sources {self._config.collection} where true", "hits": 0},
        )
        response.raise_for_status()
        payload = response.json()
        root = payload.get("root") if isinstance(payload, dict) else {}
        fields = root.get("fields") if isinstance(root, dict) else {}
        total = fields.get("totalCount") if isinstance(fields, dict) else 0
        return int(total or 0)
