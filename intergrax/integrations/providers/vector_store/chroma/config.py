# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Chroma vector store integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import ValidationInfo, field_validator

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError

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
DEFAULT_MODE: Mode = "http"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_METRIC: Metric = "cosine"
DEFAULT_BATCH_SIZE = 256


class ChromaIntegrationConfig(BaseIntegrationConfig):
    """Settings for the Chroma catalog bridge."""

    mode: Mode = DEFAULT_MODE
    http_host: str = DEFAULT_HOST
    http_port: int = DEFAULT_PORT
    persist_directory: str | None = None
    collection_name: str = DEFAULT_COLLECTION
    tenant_id: str = DEFAULT_TENANT_ID
    metric: Metric = DEFAULT_METRIC
    batch_size: int = DEFAULT_BATCH_SIZE

    @staticmethod
    def _required_text(value: object, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise IntegrationConfigurationError(
                f"Chroma {field_name} must be a string",
            )
        normalized = value.strip()
        if not normalized:
            raise IntegrationConfigurationError(
                f"Chroma {field_name} must be non-empty",
            )
        return normalized

    @field_validator("mode", mode="before")
    @classmethod
    def _validate_mode(cls, value: object) -> Mode:
        if value not in ("embedded", "http"):
            raise IntegrationConfigurationError(
                "Chroma mode must be either 'http' or explicit non-production 'embedded'",
            )
        return value

    @field_validator("http_port")
    @classmethod
    def _validate_port(cls, value: object) -> int:
        if type(value) is not int or not 1 <= value <= 65535:
            raise IntegrationConfigurationError(
                "Chroma http_port must be an integer between 1 and 65535",
            )
        return value

    @staticmethod
    def _positive_int(value: object, *, field_name: str) -> int:
        if type(value) is not int or value < 1:
            raise IntegrationConfigurationError(
                f"Chroma {field_name} must be a positive integer",
            )
        return value

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, value: object) -> int:
        return cls._positive_int(value, field_name="batch_size")

    @field_validator("http_host", "collection_name", "tenant_id")
    @classmethod
    def _validate_text_fields(cls, value: object, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return cls._required_text(value, field_name=field_name)

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, value: object) -> Metric:
        if value not in ("cosine", "l2"):
            raise IntegrationConfigurationError(
                "Chroma metric must be either 'cosine' or 'l2'",
            )
        return value

    @classmethod
    def from_env(cls, **overrides: object) -> ChromaIntegrationConfig:
        mode_raw = os.environ.get(ENV_CHROMA_MODE, DEFAULT_MODE).strip() or DEFAULT_MODE
        http_host = (
            os.environ.get(ENV_CHROMA_HOST, DEFAULT_HOST).strip() or DEFAULT_HOST
        )
        port_raw = os.environ.get(ENV_CHROMA_PORT, "").strip()
        persist_directory = (
            os.environ.get(ENV_CHROMA_PERSIST_DIRECTORY, "").strip() or None
        )
        collection_name = (
            os.environ.get(ENV_CHROMA_COLLECTION, DEFAULT_COLLECTION).strip()
            or DEFAULT_COLLECTION
        )
        tenant_id = (
            os.environ.get(ENV_CHROMA_TENANT_ID, DEFAULT_TENANT_ID).strip()
            or DEFAULT_TENANT_ID
        )
        metric_raw = (
            os.environ.get(ENV_CHROMA_METRIC, DEFAULT_METRIC).strip() or DEFAULT_METRIC
        )
        batch_raw = os.environ.get(ENV_CHROMA_BATCH_SIZE, "").strip()
        payload: dict[str, object] = {
            "mode": mode_raw,
            "http_host": http_host,
            "collection_name": collection_name,
            "tenant_id": tenant_id,
            "metric": metric_raw,
            "persist_directory": persist_directory,
        }
        try:
            payload["http_port"] = int(port_raw) if port_raw else DEFAULT_PORT
        except ValueError as exc:
            raise IntegrationConfigurationError(
                "Chroma http_port must be an integer",
            ) from exc
        try:
            payload["batch_size"] = int(batch_raw) if batch_raw else DEFAULT_BATCH_SIZE
        except ValueError as exc:
            raise IntegrationConfigurationError(
                "Chroma batch_size must be an integer",
            ) from exc
        payload.update(overrides)
        config = cls.model_validate(payload)
        config._validate_mode(config.mode)
        config._validate_port(config.http_port)
        config._required_text(config.http_host, field_name="http_host")
        config._required_text(config.collection_name, field_name="collection_name")
        config._required_text(config.tenant_id, field_name="tenant_id")
        config._positive_int(config.batch_size, field_name="batch_size")
        if config.metric not in ("cosine", "l2"):
            raise IntegrationConfigurationError(
                "Chroma metric must be either 'cosine' or 'l2'",
            )
        return config
