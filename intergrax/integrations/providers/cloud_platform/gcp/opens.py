# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level GCP credential openers — internal to the gcp integration package.

Only this module may construct ``google.auth`` credentials for the GCP cloud facade.
All composition roots use ``bundle.create_gcp_*`` or ``profile.resolve(CLOUD_PLATFORM)``.
"""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.providers.cloud_platform.gcp.adapter import GcpCloudPlatform
from intergrax.integrations.providers.cloud_platform.gcp.config import GcpIntegrationConfig

_GCP_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _import_google_auth() -> tuple[Any, Any, Any]:
    try:
        import google.auth
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "GCP integration requires google-auth. "
            "Install with: uv sync --extra dev  (includes google-auth)"
        ) from exc
    return google.auth, Request, service_account


def _ensure_valid_credentials(credentials: Any, request_cls: Any) -> None:
    if not attribute_access.optional(credentials, "valid", False):
        credentials.refresh(request_cls())


def open_gcp_credentials(
    config: GcpIntegrationConfig,
    *,
    credential_factory: Optional[Callable[[], tuple[Any, str]]] = None,
) -> tuple[Any, str]:
    if credential_factory is not None:
        credentials, project_id = credential_factory()
        return credentials, project_id
    google_auth, Request, service_account = _import_google_auth()
    scopes = [_GCP_CLOUD_PLATFORM_SCOPE]
    if config.credentials_file:
        credentials = service_account.Credentials.from_service_account_file(
            config.credentials_file,
            scopes=scopes,
        )
        project_id = config.project_id or str(attribute_access.optional(credentials, "project_id", "") or "")
    else:
        credentials, default_project = google_auth.default(scopes=scopes)
        project_id = config.project_id or str(default_project or "")
    _ensure_valid_credentials(credentials, Request)
    return credentials, project_id


def open_gcp_cloud_platform(
    config: GcpIntegrationConfig,
    *,
    implementation: Optional[CloudPlatform] = None,
    credentials: Optional[Any] = None,
    resolved_project_id: str = "",
    credential_factory: Optional[Callable[[], tuple[Any, str]]] = None,
) -> CloudPlatform:
    if implementation is not None:
        return implementation
    if credentials is None:
        credentials, resolved_project_id = open_gcp_credentials(
            config,
            credential_factory=credential_factory,
        )
    return GcpCloudPlatform(
        config,
        credentials,
        resolved_project_id=resolved_project_id,
    )
