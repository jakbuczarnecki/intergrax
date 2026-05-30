# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pinecone vector store integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os
from typing import Literal, Optional

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_PINECONE_API_KEY = "INTERGRAX_PINECONE_API_KEY"
ENV_PINECONE_INDEX = "INTERGRAX_PINECONE_INDEX"
ENV_PINECONE_COLLECTION = "INTERGRAX_PINECONE_COLLECTION"
ENV_PINECONE_TENANT_ID = "INTERGRAX_PINECONE_TENANT_ID"
ENV_PINECONE_METRIC = "INTERGRAX_PINECONE_METRIC"
ENV_PINECONE_CLOUD = "INTERGRAX_PINECONE_CLOUD"
ENV_PINECONE_REGION = "INTERGRAX_PINECONE_REGION"
ENV_PINECONE_BATCH_SIZE = "INTERGRAX_PINECONE_BATCH_SIZE"

Metric = Literal["cosine", "dot", "euclidean"]

DEFAULT_COLLECTION = "intergrax"
DEFAULT_TENANT_ID = "default"
DEFAULT_METRIC: Metric = "cosine"
DEFAULT_BATCH_SIZE = 100


class PineconeIntegrationConfig(BaseIntegrationConfig):
    """
    Settings for the Pinecone catalog bridge.

    Delegates to ``intergrax.rag.vectorstore.providers.pinecone_vector_store.PineconeVectorStore``.
    """

    api_key: str = ""
    index_name: str = ""
    collection_name: str = DEFAULT_COLLECTION
    tenant_id: str = DEFAULT_TENANT_ID
    metric: Metric = DEFAULT_METRIC
    batch_size: int = DEFAULT_BATCH_SIZE
    cloud: Optional[str] = None
    region: Optional[str] = None

    def resolved_index_name(self) -> str:
        return (self.index_name or self.collection_name).strip() or DEFAULT_COLLECTION

    @classmethod
    def from_env(cls, **overrides: object) -> PineconeIntegrationConfig:
        api_key = os.environ.get(ENV_PINECONE_API_KEY, "").strip()
        index_name = os.environ.get(ENV_PINECONE_INDEX, "").strip()
        collection_name = (
            os.environ.get(ENV_PINECONE_COLLECTION, DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
        )
        tenant_id = (
            os.environ.get(ENV_PINECONE_TENANT_ID, DEFAULT_TENANT_ID).strip() or DEFAULT_TENANT_ID
        )
        metric_raw = os.environ.get(ENV_PINECONE_METRIC, DEFAULT_METRIC).strip() or DEFAULT_METRIC
        cloud = os.environ.get(ENV_PINECONE_CLOUD, "").strip() or None
        region = os.environ.get(ENV_PINECONE_REGION, "").strip() or None
        batch_raw = os.environ.get(ENV_PINECONE_BATCH_SIZE, "").strip()
        payload: dict[str, object] = {
            "api_key": api_key,
            "index_name": index_name,
            "collection_name": collection_name,
            "tenant_id": tenant_id,
            "metric": metric_raw,
            "cloud": cloud,
            "region": region,
        }
        if batch_raw:
            payload["batch_size"] = int(batch_raw)
        else:
            payload["batch_size"] = DEFAULT_BATCH_SIZE
        payload.update(overrides)
        return cls.model_validate(payload)
