# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pinecone vector store integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import field_validator

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError

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
    tenant_id: str = ""
    metric: Metric = DEFAULT_METRIC
    batch_size: int = DEFAULT_BATCH_SIZE
    cloud: Optional[str] = None
    region: Optional[str] = None

    def require_tenant_id(self) -> str:
        tenant = self.tenant_id.strip()
        if not tenant:
            raise IntegrationConfigurationError(
                "Pinecone integration requires explicit tenant_id configuration",
            )
        return tenant

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: object) -> str:
        if not isinstance(value, str):
            raise IntegrationConfigurationError("Pinecone tenant_id must be a string")
        normalized = value.strip()
        if not normalized:
            raise IntegrationConfigurationError(
                "Pinecone integration requires explicit tenant_id configuration",
            )
        return normalized

    def resolved_index_name(self) -> str:
        return (self.index_name or self.collection_name).strip() or DEFAULT_COLLECTION

    @classmethod
    def from_env(cls, **overrides: object) -> PineconeIntegrationConfig:
        tenant_override = overrides.get("tenant_id")
        tenant_from_env = os.environ.get(ENV_PINECONE_TENANT_ID, "").strip()
        if tenant_override is not None:
            tenant_id = str(tenant_override).strip()
        elif tenant_from_env:
            tenant_id = tenant_from_env
        else:
            raise IntegrationConfigurationError(
                f"Pinecone integration requires {ENV_PINECONE_TENANT_ID}",
            )

        api_key = os.environ.get(ENV_PINECONE_API_KEY, "").strip()
        index_name = os.environ.get(ENV_PINECONE_INDEX, "").strip()
        collection_name = (
            os.environ.get(ENV_PINECONE_COLLECTION, DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
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
