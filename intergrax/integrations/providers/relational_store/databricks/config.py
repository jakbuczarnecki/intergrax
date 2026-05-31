# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Databricks SQL Warehouse integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os
import re
from typing import Optional

from pydantic import field_validator

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_DATABRICKS_HOST = "INTERGRAX_DATABRICKS_HOST"
ENV_DATABRICKS_HTTP_PATH = "INTERGRAX_DATABRICKS_HTTP_PATH"
ENV_DATABRICKS_TOKEN = "INTERGRAX_DATABRICKS_TOKEN"
ENV_DATABRICKS_CATALOG = "INTERGRAX_DATABRICKS_CATALOG"
ENV_DATABRICKS_SCHEMA = "INTERGRAX_DATABRICKS_SCHEMA"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DatabricksIntegrationConfig(BaseIntegrationConfig):
    """
    Connection settings for Databricks SQL Warehouse ``RelationalStore``.

    ``tenant_schema`` maps to Unity Catalog ``USE SCHEMA`` (optionally after ``USE CATALOG``).
    """

    host: str = ""
    http_path: str = ""
    access_token: str = ""
    catalog: Optional[str] = None
    tenant_schema: Optional[str] = None

    @field_validator("host")
    @classmethod
    def _normalize_host(cls, value: str) -> str:
        host = value.strip()
        for prefix in ("https://", "http://"):
            if host.lower().startswith(prefix):
                host = host[len(prefix) :]
        return host.rstrip("/")

    @field_validator("catalog", "tenant_schema")
    @classmethod
    def _validate_identifier(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        if not _IDENTIFIER_PATTERN.match(value):
            raise ValueError(
                "catalog and tenant_schema must be simple SQL identifiers (letters, digits, underscore)"
            )
        return value

    def connect_kwargs(self) -> dict[str, str]:
        if not self.host or not self.http_path or not self.access_token:
            raise ValueError(
                "Databricks integration requires host, http_path, and access_token "
                "(INTERGRAX_DATABRICKS_HOST, INTERGRAX_DATABRICKS_HTTP_PATH, INTERGRAX_DATABRICKS_TOKEN)"
            )
        return {
            "server_hostname": self.host,
            "http_path": self.http_path,
            "access_token": self.access_token,
        }

    @classmethod
    def from_env(cls, **overrides: object) -> DatabricksIntegrationConfig:
        payload: dict[str, object] = {}
        host = os.environ.get(ENV_DATABRICKS_HOST, "").strip()
        if host:
            payload["host"] = host
        http_path = os.environ.get(ENV_DATABRICKS_HTTP_PATH, "").strip()
        if http_path:
            payload["http_path"] = http_path
        if ENV_DATABRICKS_TOKEN in os.environ:
            payload["access_token"] = os.environ.get(ENV_DATABRICKS_TOKEN, "")
        catalog = os.environ.get(ENV_DATABRICKS_CATALOG, "").strip()
        if catalog:
            payload["catalog"] = catalog
        schema = os.environ.get(ENV_DATABRICKS_SCHEMA, "").strip()
        if schema:
            payload["tenant_schema"] = schema
        payload.update(overrides)
        return cls.model_validate(payload)
