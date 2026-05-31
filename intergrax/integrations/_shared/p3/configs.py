# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.7 integration config models."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class VaultIntegrationConfig(BaseIntegrationConfig):
    addr: str = "http://127.0.0.1:8200"
    token: str = ""
    mount: str = "secret"
    namespace: str = ""

    @classmethod
    def from_env(cls, **overrides: object) -> VaultIntegrationConfig:
        payload = {
            "addr": _env("INTERGRAX_VAULT_ADDR", "http://127.0.0.1:8200") or "http://127.0.0.1:8200",
            "token": _env("INTERGRAX_VAULT_TOKEN"),
            "mount": _env("INTERGRAX_VAULT_MOUNT", "secret") or "secret",
            "namespace": _env("INTERGRAX_VAULT_NAMESPACE"),
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class VectorIntegrationConfig(BaseIntegrationConfig):
    url: str = ""
    api_key: str = ""
    collection: str = "intergrax"
    tenant_id: str = "default"

    def require_url(self) -> str:
        if not self.url.strip():
            raise IntegrationConfigurationError("Vector integration requires URL")
        return self.url.strip()

    @classmethod
    def from_env(cls, prefix: str, **overrides: object) -> VectorIntegrationConfig:
        payload = {
            "url": _env(f"{prefix}_URL"),
            "api_key": _env(f"{prefix}_API_KEY"),
            "collection": _env(f"{prefix}_COLLECTION", "intergrax") or "intergrax",
            "tenant_id": _env(f"{prefix}_TENANT_ID", "default") or "default",
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class MinioIntegrationConfig(BaseIntegrationConfig):
    endpoint: str = ""
    access_key: str = ""
    secret_key: str = ""
    bucket: str = "intergrax"
    prefix: str = ""
    secure: bool = True

    def object_key(self, key: str) -> str:
        normalized = key.lstrip("/")
        prefix = self.prefix.strip("/")
        return f"{prefix}/{normalized}" if prefix else normalized

    def require_bucket(self) -> str:
        if not self.bucket.strip():
            raise IntegrationConfigurationError("MinIO requires bucket (INTERGRAX_MINIO_BUCKET)")
        return self.bucket.strip()

    @classmethod
    def from_env(cls, **overrides: object) -> MinioIntegrationConfig:
        payload = {
            "endpoint": _env("INTERGRAX_MINIO_ENDPOINT"),
            "access_key": _env("INTERGRAX_MINIO_ACCESS_KEY"),
            "secret_key": _env("INTERGRAX_MINIO_SECRET_KEY"),
            "bucket": _env("INTERGRAX_MINIO_BUCKET", "intergrax") or "intergrax",
            "prefix": _env("INTERGRAX_MINIO_PREFIX"),
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class FilesystemIntegrationConfig(BaseIntegrationConfig):
    root_dir: str = "build/artifacts"
    prefix: str = ""

    def object_key(self, key: str) -> str:
        normalized = key.lstrip("/")
        prefix = self.prefix.strip("/")
        return f"{prefix}/{normalized}" if prefix else normalized

    def require_root(self) -> str:
        if not self.root_dir.strip():
            raise IntegrationConfigurationError("Filesystem storage requires INTERGRAX_FILESYSTEM_ROOT_DIR")
        return self.root_dir.strip()

    @classmethod
    def from_env(cls, **overrides: object) -> FilesystemIntegrationConfig:
        payload = {
            "root_dir": _env("INTERGRAX_FILESYSTEM_ROOT_DIR", "build/artifacts") or "build/artifacts",
            "prefix": _env("INTERGRAX_FILESYSTEM_PREFIX"),
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class FirecrawlIntegrationConfig(BaseIntegrationConfig):
    api_key: str = ""
    base_url: str = "https://api.firecrawl.dev"
    timeout_seconds: int = 60

    @classmethod
    def from_env(cls, **overrides: object) -> FirecrawlIntegrationConfig:
        payload = {
            "api_key": _env("INTERGRAX_FIRECRAWL_API_KEY"),
            "base_url": _env("INTERGRAX_FIRECRAWL_URL", "https://api.firecrawl.dev") or "https://api.firecrawl.dev",
        }
        payload.update(overrides)
        return cls.model_validate(payload)


class SeleniumIntegrationConfig(BaseIntegrationConfig):
    driver_url: str = ""
    browser: str = "chrome"
    headless: bool = True
    timeout_ms: int = 30000

    @classmethod
    def from_env(cls, **overrides: object) -> SeleniumIntegrationConfig:
        payload = {
            "driver_url": _env("INTERGRAX_SELENIUM_DRIVER_URL"),
            "browser": _env("INTERGRAX_SELENIUM_BROWSER", "chrome") or "chrome",
        }
        payload.update(overrides)
        return cls.model_validate(payload)
