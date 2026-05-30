# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Chroma vector store integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os
from typing import Literal, Optional

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_CHROMA_MODE = "INTERGRAX_CHROMA_MODE"
ENV_CHROMA_HOST = "INTERGRAX_CHROMA_HOST"
ENV_CHROMA_PORT = "INTERGRAX_CHROMA_PORT"
ENV_CHROMA_PERSIST_DIRECTORY = "INTERGRAX_CHROMA_PERSIST_DIRECTORY"
ENV_CHROMA_COLLECTION = "INTERGRAX_CHROMA_COLLECTION"
ENV_CHROMA_TENANT_ID = "INTERGRAX_CHROMA_TENANT_ID"
ENV_CHROMA_METRIC = "INTERGRAX_CHROMA_METRIC"
ENV_CHROMA_BATCH_SIZE = "INTERGRAX_CHROMA_BATCH_SIZE"

Mode = Literal["embedded", "http"]
Metric = Literal["cosine", "l2"]

DEFAULT_COLLECTION = "intergrax"
DEFAULT_TENANT_ID = "default"
DEFAULT_MODE: Mode = "embedded"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_METRIC: Metric = "cosine"
DEFAULT_BATCH_SIZE = 256


class ChromaIntegrationConfig(BaseIntegrationConfig):
    """Settings for the Chroma catalog bridge."""

    mode: Mode = DEFAULT_MODE
    http_host: str = DEFAULT_HOST
    http_port: int = DEFAULT_PORT
    persist_directory: Optional[str] = None
    collection_name: str = DEFAULT_COLLECTION
    tenant_id: str = DEFAULT_TENANT_ID
    metric: Metric = DEFAULT_METRIC
    batch_size: int = DEFAULT_BATCH_SIZE

    @classmethod
    def from_env(cls, **overrides: object) -> ChromaIntegrationConfig:
        mode_raw = os.environ.get(ENV_CHROMA_MODE, DEFAULT_MODE).strip() or DEFAULT_MODE
        http_host = os.environ.get(ENV_CHROMA_HOST, DEFAULT_HOST).strip() or DEFAULT_HOST
        port_raw = os.environ.get(ENV_CHROMA_PORT, "").strip()
        persist_directory = os.environ.get(ENV_CHROMA_PERSIST_DIRECTORY, "").strip() or None
        collection_name = (
            os.environ.get(ENV_CHROMA_COLLECTION, DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
        )
        tenant_id = (
            os.environ.get(ENV_CHROMA_TENANT_ID, DEFAULT_TENANT_ID).strip() or DEFAULT_TENANT_ID
        )
        metric_raw = os.environ.get(ENV_CHROMA_METRIC, DEFAULT_METRIC).strip() or DEFAULT_METRIC
        batch_raw = os.environ.get(ENV_CHROMA_BATCH_SIZE, "").strip()
        payload: dict[str, object] = {
            "mode": mode_raw,
            "http_host": http_host,
            "collection_name": collection_name,
            "tenant_id": tenant_id,
            "metric": metric_raw,
            "persist_directory": persist_directory,
        }
        payload["http_port"] = int(port_raw) if port_raw else DEFAULT_PORT
        payload["batch_size"] = int(batch_raw) if batch_raw else DEFAULT_BATCH_SIZE
        payload.update(overrides)
        return cls.model_validate(payload)
