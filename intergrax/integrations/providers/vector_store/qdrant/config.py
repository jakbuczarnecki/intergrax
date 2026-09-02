# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qdrant vector store integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os
from typing import Literal, Optional

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_QDRANT_URL = "INTERGRAX_QDRANT_URL"
ENV_QDRANT_API_KEY = "INTERGRAX_QDRANT_API_KEY"
ENV_QDRANT_HOST = "INTERGRAX_QDRANT_HOST"
ENV_QDRANT_PORT = "INTERGRAX_QDRANT_PORT"
ENV_QDRANT_COLLECTION = "INTERGRAX_QDRANT_COLLECTION"
ENV_QDRANT_TENANT_ID = "INTERGRAX_QDRANT_TENANT_ID"
ENV_QDRANT_METRIC = "INTERGRAX_QDRANT_METRIC"
ENV_QDRANT_BATCH_SIZE = "INTERGRAX_QDRANT_BATCH_SIZE"
ENV_QDRANT_SPARSE_VECTORS = "INTERGRAX_RAG_QDRANT_SPARSE"

Metric = Literal["cosine", "dot", "euclidean"]

DEFAULT_COLLECTION = "intergrax"
DEFAULT_TENANT_ID = "default"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6333
DEFAULT_METRIC: Metric = "cosine"
DEFAULT_BATCH_SIZE = 256


class QdrantIntegrationConfig(BaseIntegrationConfig):
    """Settings for the Qdrant catalog bridge."""

    url: str = ""
    api_key: str = ""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    collection_name: str = DEFAULT_COLLECTION
    tenant_id: str = DEFAULT_TENANT_ID
    metric: Metric = DEFAULT_METRIC
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_sparse_vectors: bool = False

    def resolved_url(self) -> Optional[str]:
        return self.url.strip() or None

    @classmethod
    def from_env(cls, **overrides: object) -> QdrantIntegrationConfig:
        url = os.environ.get(ENV_QDRANT_URL, "").strip()
        api_key = os.environ.get(ENV_QDRANT_API_KEY, "").strip()
        host = os.environ.get(ENV_QDRANT_HOST, DEFAULT_HOST).strip() or DEFAULT_HOST
        port_raw = os.environ.get(ENV_QDRANT_PORT, "").strip()
        collection_name = (
            os.environ.get(ENV_QDRANT_COLLECTION, DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
        )
        tenant_id = (
            os.environ.get(ENV_QDRANT_TENANT_ID, DEFAULT_TENANT_ID).strip() or DEFAULT_TENANT_ID
        )
        metric_raw = os.environ.get(ENV_QDRANT_METRIC, DEFAULT_METRIC).strip() or DEFAULT_METRIC
        batch_raw = os.environ.get(ENV_QDRANT_BATCH_SIZE, "").strip()
        sparse_raw = os.environ.get(ENV_QDRANT_SPARSE_VECTORS, "").strip().lower()
        payload: dict[str, object] = {
            "url": url,
            "api_key": api_key,
            "host": host,
            "collection_name": collection_name,
            "tenant_id": tenant_id,
            "metric": metric_raw,
        }
        if port_raw:
            payload["port"] = int(port_raw)
        else:
            payload["port"] = DEFAULT_PORT
        payload["batch_size"] = int(batch_raw) if batch_raw else DEFAULT_BATCH_SIZE
        payload["enable_sparse_vectors"] = sparse_raw in ("1", "true", "yes", "on")
        payload.update(overrides)
        return cls.model_validate(payload)
