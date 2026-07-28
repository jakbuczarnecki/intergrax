# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Drive knowledge-read: bounded binary content download for one file."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from typing import Any, Mapping, Protocol, runtime_checkable
from urllib.parse import quote, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeTransport,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.drive import (
    MsGraphDriveItem,
    MsGraphDriveItemKind,
    parse_msgraph_drive_item,
    validate_msgraph_drive_id,
    validate_msgraph_drive_item_id,
)

DEFAULT_DRIVE_CONTENT_MAX_BYTES = 25 * 1024 * 1024
ABSOLUTE_DRIVE_CONTENT_MAX_BYTES = 100 * 1024 * 1024

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_DRIVE_RESPONSE = "unexpected Microsoft Graph drive response"
_INVALID_REDIRECT = "Microsoft Graph Drive content redirect response is invalid"
_INVALID_DOWNLOAD_URL = "Microsoft Graph Drive download location is invalid"
_INVALID_DOWNLOAD_RESPONSE = "Microsoft Graph Drive download response is invalid"
_DOWNLOAD_CLIENT_NOT_CONFIGURED = "Microsoft Graph Drive download client is not configured"
_INVALID_MAX_BYTES = "invalid Microsoft Graph Drive content max_bytes"
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_CONTENT_ITEM_SELECT = (
    "id,name,parentReference,webUrl,eTag,cTag,size,file,createdDateTime,lastModifiedDateTime"
)


class MsGraphDriveContentChanged(IntegrationDependencyError):
    """Drive file cTag or size changed during content download."""

    def __init__(self) -> None:
        super().__init__("Microsoft Graph Drive file changed during content download")


class MsGraphDriveContentTooLarge(IntegrationConfigurationError):
    """Drive file exceeds the configured content byte limit."""

    def __init__(self) -> None:
        super().__init__("Microsoft Graph Drive file exceeds the configured content limit")


def validate_msgraph_drive_download_url(value: object) -> str:
    if not isinstance(value, str):
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None
    cleaned = value.strip()
    if not cleaned:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None
    if _ASCII_CONTROL.search(cleaned):
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None

    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None
    if parsed.scheme != "https":
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None
    if parsed.username or parsed.password:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None
    if parsed.fragment:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None
    if not parsed.hostname:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None
    if not parsed.path or parsed.path == "/":
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None

    hostname = parsed.hostname.lower()
    if hostname == "localhost":
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None
    if hostname.endswith(".local") or hostname.endswith(".internal"):
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None

    if parsed.port is not None and parsed.port != 443:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None

    host_for_ip = hostname
    if host_for_ip.startswith("[") and host_for_ip.endswith("]"):
        host_for_ip = host_for_ip[1:-1]

    try:
        ipaddress.ip_address(host_for_ip)
    except ValueError:
        pass
    else:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_URL) from None

    return cleaned


class MsGraphDriveFileContent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    drive_id: str
    remote_id: str
    content_revision: str
    data: bytes = Field(repr=False)
    size_bytes: int
    mime_type: str | None
    content_hash: str

    @field_validator("drive_id", "remote_id", mode="before")
    @classmethod
    def _validate_ids(cls, value: object) -> str:
        return validate_msgraph_drive_id(value)

    @field_validator("content_revision", mode="before")
    @classmethod
    def _validate_content_revision(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        trimmed = value.strip()
        if not trimmed:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        if _ASCII_CONTROL.search(trimmed):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return trimmed

    @field_validator("data", mode="before")
    @classmethod
    def _validate_data(cls, value: object) -> bytes:
        if type(value) is not bytes:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return value

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _validate_size_bytes(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        if value < 0:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return value

    @field_validator("mime_type", mode="before")
    @classmethod
    def _validate_mime_type(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        trimmed = value.strip()
        if not trimmed:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return trimmed

    @field_validator("content_hash", mode="before")
    @classmethod
    def _validate_content_hash_format(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        if not _CONTENT_HASH_PATTERN.match(value):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_content_shape(self) -> MsGraphDriveFileContent:
        if self.size_bytes != len(self.data):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        expected_hash = hashlib.sha256(self.data).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return self


@runtime_checkable
class MsGraphDriveContentReadClient(Protocol):
    def read_drive_file_content(
        self,
        *,
        item: MsGraphDriveItem,
        max_bytes: int,
    ) -> MsGraphDriveFileContent:
        ...


def _validate_max_bytes(max_bytes: object) -> int:
    if type(max_bytes) is not int:
        raise IntegrationConfigurationError(_INVALID_MAX_BYTES) from None
    if max_bytes < 1 or max_bytes > ABSOLUTE_DRIVE_CONTENT_MAX_BYTES:
        raise IntegrationConfigurationError(_INVALID_MAX_BYTES) from None
    return max_bytes


def _validate_content_item(item: object) -> MsGraphDriveItem:
    if not isinstance(item, MsGraphDriveItem):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None
    if item.kind != MsGraphDriveItemKind.FILE:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None
    if item.c_tag is None or not item.c_tag.strip():
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None
    return item


def _response_status_code(response: object) -> int:
    try:
        status_code = response.status_code
    except AttributeError:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
    if type(status_code) is not int:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
    if status_code < 100 or status_code > 599:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
    return status_code


def _response_headers(response: object) -> Mapping[str, str]:
    try:
        headers = response.headers
    except AttributeError:
        raise IntegrationDependencyError(_INVALID_REDIRECT) from None
    if not isinstance(headers, Mapping):
        raise IntegrationDependencyError(_INVALID_REDIRECT) from None
    return headers


def _extract_location_header(response: object) -> str:
    headers = _response_headers(response)
    location_values: list[str] = []
    for key, value in headers.items():
        if key.lower() == "location":
            if isinstance(value, str):
                location_values.append(value)
            else:
                raise IntegrationDependencyError(_INVALID_REDIRECT) from None
    if len(location_values) != 1:
        raise IntegrationDependencyError(_INVALID_REDIRECT) from None
    trimmed = location_values[0].strip()
    if not trimmed:
        raise IntegrationDependencyError(_INVALID_REDIRECT) from None
    return trimmed


def _raise_for_content_redirect_response(response: object) -> None:
    status_code = _response_status_code(response)
    if status_code == 302:
        return
    if 200 <= status_code <= 299:
        raise IntegrationDependencyError(_INVALID_REDIRECT) from None
    if 300 <= status_code <= 399:
        raise IntegrationDependencyError(_INVALID_REDIRECT) from None
    if status_code in {400, 401, 403}:
        raise IntegrationConfigurationError("Microsoft Graph knowledge configuration failure") from None
    if status_code == 404:
        raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None
    if status_code in {408, 410, 429} or status_code >= 500:
        raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None
    raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None


def _parse_content_length(headers: Mapping[str, str]) -> int | None:
    raw_value: str | None = None
    for key, value in headers.items():
        if key.lower() == "content-length":
            if raw_value is not None:
                raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
            if not isinstance(value, str):
                raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
            raw_value = value
    if raw_value is None:
        return None
    trimmed = raw_value.strip()
    if not trimmed or not trimmed.isdigit():
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
    parsed = int(trimmed)
    if parsed < 0:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
    return parsed


def _raise_for_download_response(response: object) -> None:
    status_code = _response_status_code(response)
    if status_code == 200:
        return
    if status_code == 206:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
    if 300 <= status_code <= 399:
        raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
    if 400 <= status_code <= 499:
        raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None
    if status_code >= 500:
        raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure") from None
    raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None


def _execute_transport(transport_fn: Any) -> object:
    try:
        return transport_fn()
    except (IntegrationConfigurationError, IntegrationDependencyError, MsGraphDriveContentChanged):
        raise
    except Exception:
        raise IntegrationDependencyError(
            "Microsoft Graph knowledge dependency is unavailable"
        ) from None


def _verify_metadata_matches_item(
    current: MsGraphDriveItem,
    *,
    expected: MsGraphDriveItem,
) -> None:
    if current.kind != MsGraphDriveItemKind.FILE:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None
    if current.remote_id != expected.remote_id:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None
    if current.drive_id != expected.drive_id:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None
    if current.c_tag != expected.c_tag:
        raise MsGraphDriveContentChanged() from None


def _verify_downloaded_size(
    *,
    downloaded_len: int,
    expected_item: MsGraphDriveItem,
    current: MsGraphDriveItem,
) -> None:
    if current.size_bytes is not None and current.size_bytes != downloaded_len:
        raise MsGraphDriveContentChanged() from None
    if expected_item.size_bytes is not None and expected_item.size_bytes != downloaded_len:
        raise MsGraphDriveContentChanged() from None


class MsGraphDriveContentReader:
    """Drive file content reader: metadata cTag gate, 302 redirect, bounded download."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        graph_transport: MsGraphKnowledgeTransport,
        graph_http_client: Any,
        download_http_client: Any,
    ) -> None:
        self._config = config
        self._graph_transport = graph_transport
        self._graph_http_client = graph_http_client
        self._download_http_client = download_http_client

    def read_file_content(
        self,
        *,
        item: MsGraphDriveItem,
        max_bytes: int,
    ) -> MsGraphDriveFileContent:
        if self._download_http_client is None:
            raise IntegrationConfigurationError(_DOWNLOAD_CLIENT_NOT_CONFIGURED) from None

        validated_item = _validate_content_item(item)
        validated_max_bytes = _validate_max_bytes(max_bytes)

        if (
            validated_item.size_bytes is not None
            and validated_item.size_bytes > validated_max_bytes
        ):
            raise MsGraphDriveContentTooLarge() from None

        self._fetch_and_verify_metadata(validated_item, max_bytes=validated_max_bytes)

        download_url = self._request_content_redirect(validated_item)
        data = self._download_content(download_url, max_bytes=validated_max_bytes)

        current_after = self._fetch_current_metadata(validated_item)
        _verify_metadata_matches_item(current_after, expected=validated_item)
        _verify_downloaded_size(
            downloaded_len=len(data),
            expected_item=validated_item,
            current=current_after,
        )

        content_hash = hashlib.sha256(data).hexdigest()
        return MsGraphDriveFileContent(
            drive_id=validated_item.drive_id,
            remote_id=validated_item.remote_id,
            content_revision=validated_item.c_tag or "",
            data=data,
            size_bytes=len(data),
            mime_type=validated_item.mime_type,
            content_hash=content_hash,
        )

    def _fetch_and_verify_metadata(
        self,
        item: MsGraphDriveItem,
        *,
        max_bytes: int,
    ) -> MsGraphDriveItem:
        current = self._fetch_current_metadata(item)
        _verify_metadata_matches_item(current, expected=item)
        if current.size_bytes is not None and current.size_bytes > max_bytes:
            raise MsGraphDriveContentTooLarge() from None
        return current

    def _fetch_current_metadata(self, item: MsGraphDriveItem) -> MsGraphDriveItem:
        validated_drive_id = validate_msgraph_drive_id(item.drive_id)
        validated_item_id = validate_msgraph_drive_item_id(item.remote_id)
        quoted_drive_id = quote(validated_drive_id, safe="")
        quoted_item_id = quote(validated_item_id, safe="")
        path = f"/drives/{quoted_drive_id}/items/{quoted_item_id}"
        payload = self._graph_transport.get_initial_json(
            path=path,
            params={"$select": _CONTENT_ITEM_SELECT},
            not_found_is_dependency=True,
        )
        return parse_msgraph_drive_item(payload, expected_drive_id=validated_drive_id)

    def _request_content_redirect(self, item: MsGraphDriveItem) -> str:
        validated_drive_id = validate_msgraph_drive_id(item.drive_id)
        validated_item_id = validate_msgraph_drive_item_id(item.remote_id)
        quoted_drive_id = quote(validated_drive_id, safe="")
        quoted_item_id = quote(validated_item_id, safe="")
        path = f"/drives/{quoted_drive_id}/items/{quoted_item_id}/content"

        def _do_request() -> object:
            return self._graph_http_client.get(
                path,
                headers={"Accept": "application/octet-stream"},
                follow_redirects=False,
            )

        response = _execute_transport(_do_request)
        _raise_for_content_redirect_response(response)
        location = _extract_location_header(response)
        return validate_msgraph_drive_download_url(location)

    def _download_content(self, download_url: str, *, max_bytes: int) -> bytes:
        validated_url = validate_msgraph_drive_download_url(download_url)

        def _do_stream() -> object:
            return self._download_http_client.stream(
                "GET",
                validated_url,
                headers={"Accept": "application/octet-stream"},
                follow_redirects=False,
            )

        try:
            stream_context = _do_stream()
        except Exception:
            raise IntegrationDependencyError(
                "Microsoft Graph knowledge dependency is unavailable"
            ) from None

        try:
            with stream_context as response:
                _raise_for_download_response(response)
                headers = _response_headers(response)
                content_length = _parse_content_length(headers)
                if content_length is not None and content_length > max_bytes:
                    raise MsGraphDriveContentTooLarge() from None

                try:
                    iter_bytes = response.iter_bytes
                except AttributeError:
                    raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
                if not callable(iter_bytes):
                    raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None

                buffer = bytearray()
                for chunk in iter_bytes():
                    if type(chunk) is not bytes:
                        raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
                    buffer.extend(chunk)
                    if len(buffer) > max_bytes:
                        raise MsGraphDriveContentTooLarge() from None

                data = bytes(buffer)
                if content_length is not None and len(data) != content_length:
                    raise IntegrationDependencyError(_INVALID_DOWNLOAD_RESPONSE) from None
                return data
        except (IntegrationConfigurationError, IntegrationDependencyError, MsGraphDriveContentTooLarge):
            raise
        except Exception:
            raise IntegrationDependencyError(
                "Microsoft Graph knowledge dependency is unavailable"
            ) from None
