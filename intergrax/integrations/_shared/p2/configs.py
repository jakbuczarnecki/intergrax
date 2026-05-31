# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""P2/P3 integration config models (shared across thin provider shells)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class GcsIntegrationConfig(BaseIntegrationConfig):
    bucket: str = ""
    prefix: str = ""
    project_id: str = ""

    def require_bucket(self) -> str:
        if not self.bucket.strip():
            raise IntegrationConfigurationError("GCS requires bucket (INTERGRAX_GCS_BUCKET)")
        return self.bucket.strip()

    def object_key(self, key: str) -> str:
        normalized = key.lstrip("/")
        prefix = self.prefix.strip("/")
        return f"{prefix}/{normalized}" if prefix else normalized

    @classmethod
    def from_env(cls, **overrides: object) -> GcsIntegrationConfig:
        payload = {
            "bucket": _env("INTERGRAX_GCS_BUCKET"),
            "prefix": _env("INTERGRAX_GCS_PREFIX"),
            "project_id": _env("INTERGRAX_GCS_PROJECT_ID"),
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class DynamoDBIntegrationConfig(BaseIntegrationConfig):
    table_name: str = "intergrax"
    region: str = ""
    partition_attr: str = "partition_key"
    sort_attr: str = "row_key"

    @classmethod
    def from_env(cls, **overrides: object) -> DynamoDBIntegrationConfig:
        payload = {
            "table_name": _env("INTERGRAX_DYNAMODB_TABLE", "intergrax"),
            "region": _env("INTERGRAX_DYNAMODB_REGION"),
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class QueueIntegrationConfig(BaseIntegrationConfig):
    queue_name: str = "intergrax"
    topic: str = "intergrax"
    region: str = ""
    connection_string: str = ""
    project_id: str = ""

    @classmethod
    def from_env(cls, prefix: str, **overrides: object) -> QueueIntegrationConfig:
        payload = {
            "queue_name": _env(f"{prefix}_QUEUE", "intergrax"),
            "topic": _env(f"{prefix}_TOPIC", "intergrax"),
            "region": _env(f"{prefix}_REGION"),
            "connection_string": _env(f"{prefix}_CONNECTION_STRING"),
            "project_id": _env(f"{prefix}_PROJECT_ID"),
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class MemcachedIntegrationConfig(BaseIntegrationConfig):
    host: str = "localhost"
    port: int = 11211

    @classmethod
    def from_env(cls, **overrides: object) -> MemcachedIntegrationConfig:
        payload = {"host": _env("INTERGRAX_MEMCACHED_HOST", "localhost") or "localhost"}
        port_raw = _env("INTERGRAX_MEMCACHED_PORT")
        if port_raw:
            payload["port"] = int(port_raw)
        payload.update(overrides)
        return cls.model_validate(payload)


class SqlIntegrationConfig(BaseIntegrationConfig):
    dsn: str = ""
    connection_string: str = ""
    host: str = ""
    user: str = ""
    password: str = ""
    database: str = ""
    tenant_schema: str = ""

    def connection_dsn(self) -> str:
        dsn = self.dsn.strip() or self.connection_string.strip()
        if not dsn:
            raise IntegrationConfigurationError("SQL integration requires DSN or connection_string")
        return dsn

    @classmethod
    def from_env(cls, prefix: str, **overrides: object) -> SqlIntegrationConfig:
        payload = {
            "dsn": _env(f"{prefix}_DSN"),
            "connection_string": _env(f"{prefix}_CONNECTION_STRING"),
            "host": _env(f"{prefix}_HOST"),
            "user": _env(f"{prefix}_USER"),
            "password": _env(f"{prefix}_PASSWORD"),
            "database": _env(f"{prefix}_DATABASE"),
            "tenant_schema": _env(f"{prefix}_SCHEMA"),
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class SmtpIntegrationConfig(BaseIntegrationConfig):
    smtp_host: str = ""
    smtp_port: int = 587
    user: str = ""
    password: str = ""
    from_address: str = ""
    use_tls: bool = True

    @classmethod
    def from_env(cls, **overrides: object) -> SmtpIntegrationConfig:
        payload = {
            "smtp_host": _env("INTERGRAX_EMAIL_SMTP_HOST"),
            "user": _env("INTERGRAX_EMAIL_SMTP_USER"),
            "password": _env("INTERGRAX_EMAIL_SMTP_PASSWORD"),
            "from_address": _env("INTERGRAX_EMAIL_SMTP_FROM"),
        }
        port_raw = _env("INTERGRAX_EMAIL_SMTP_PORT")
        if port_raw:
            payload["smtp_port"] = int(port_raw)
        payload.update(overrides)
        return cls.model_validate(payload)


class HttpIntegrationConfig(BaseIntegrationConfig):
    base_url: str = ""
    api_key: str = ""
    token: str = ""
    user: str = ""
    password: str = ""
    org: str = ""
    repo: str = ""
    site_url: str = ""
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls, prefix: str, **overrides: object) -> HttpIntegrationConfig:
        payload = {
            "base_url": _env(f"{prefix}_URL"),
            "api_key": _env(f"{prefix}_API_KEY"),
            "token": _env(f"{prefix}_TOKEN"),
            "user": _env(f"{prefix}_USER"),
            "password": _env(f"{prefix}_PASSWORD"),
            "org": _env(f"{prefix}_ORG"),
            "repo": _env(f"{prefix}_REPO"),
            "site_url": _env(f"{prefix}_SITE_URL"),
        }
        timeout_raw = _env(f"{prefix}_TIMEOUT")
        if timeout_raw:
            payload["timeout_seconds"] = int(timeout_raw)
        payload.update(overrides)
        return cls.model_validate(payload)


class OtelIntegrationConfig(BaseIntegrationConfig):
    endpoint: str = "http://localhost:4318"
    service_name: str = "intergrax"

    @classmethod
    def from_env(cls, **overrides: object) -> OtelIntegrationConfig:
        payload = {
            "endpoint": _env("INTERGRAX_OTEL_ENDPOINT", "http://localhost:4318") or "http://localhost:4318",
            "service_name": _env("INTERGRAX_OTEL_SERVICE_NAME", "intergrax") or "intergrax",
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class PlaywrightIntegrationConfig(BaseIntegrationConfig):
    headless: bool = True
    timeout_ms: int = 30000

    @classmethod
    def from_env(cls, **overrides: object) -> PlaywrightIntegrationConfig:
        payload: dict[str, object] = {"headless": True, "timeout_ms": 30000}
        payload.update(overrides)
        return cls.model_validate(payload)
