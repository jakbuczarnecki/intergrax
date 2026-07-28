# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class WebContentCaptureErrorCode(str, Enum):
    WEB_URL_INVALID = "web_url_invalid"
    WEB_URL_SCHEME_NOT_ALLOWED = "web_url_scheme_not_allowed"
    WEB_URL_CREDENTIALS_NOT_ALLOWED = "web_url_credentials_not_allowed"
    WEB_URL_PORT_NOT_ALLOWED = "web_url_port_not_allowed"
    WEB_URL_HOST_NOT_ALLOWED = "web_url_host_not_allowed"
    WEB_URL_RESOLUTION_FAILED = "web_url_resolution_failed"
    WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED = "web_url_non_global_address_blocked"
    WEB_URL_REDIRECT_LOCATION_MISSING = "web_url_redirect_location_missing"
    WEB_URL_REDIRECT_TARGET_BLOCKED = "web_url_redirect_target_blocked"
    WEB_URL_REDIRECT_LIMIT_EXCEEDED = "web_url_redirect_limit_exceeded"
    WEB_URL_TIMEOUT = "web_url_timeout"
    WEB_URL_TLS_FAILED = "web_url_tls_failed"
    WEB_URL_FETCH_FAILED = "web_url_fetch_failed"
    WEB_URL_HTTP_ERROR = "web_url_http_error"
    WEB_URL_CONTENT_TYPE_UNSUPPORTED = "web_url_content_type_unsupported"
    WEB_URL_CONTENT_TOO_LARGE = "web_url_content_too_large"
    WEB_URL_CONTENT_ENCODING_UNSUPPORTED = "web_url_content_encoding_unsupported"
    WEB_URL_RESPONSE_FRAMING_INVALID = "web_url_response_framing_invalid"
    WEB_URL_DECODE_FAILED = "web_url_decode_failed"
    WEB_URL_EMPTY_CONTENT = "web_url_empty_content"
    WEB_URL_EXTRACTION_FAILED = "web_url_extraction_failed"


class WebContentCaptureError(RuntimeError):
  def __init__(
      self,
      code: WebContentCaptureErrorCode,
      *,
      status_code: int | None = None,
      retryable: bool = False,
  ) -> None:
      self.code = code
      self.status_code = status_code
      self.retryable = retryable
      super().__init__(code.value)

  def __str__(self) -> str:
      return self.code.value

  def __repr__(self) -> str:
      return f"WebContentCaptureError(code={self.code.value!r})"


class WebContentCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(repr=False)
    timeout_seconds: int = Field(default=20, ge=5, le=60)
    max_redirects: int = Field(default=5, ge=0, le=10)
    max_response_bytes: int = Field(default=5 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    max_extracted_chars: int = Field(default=2 * 1024 * 1024, ge=1024, le=5 * 1024 * 1024)
    extraction_mode: Literal["basic", "advanced"] = "advanced"


class CapturedWebContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safe_display_url: str
    requested_url_fingerprint: str
    final_url_fingerprint: str
    final_host_changed: bool

    title: str
    text: str
    content_type: str
    content_hash: str

    status_code: int
    redirect_count: int
    content_bytes: int
    text_chars: int

    capture_mode: Literal["http"]
    extraction_method: str
    fetched_at: datetime


@runtime_checkable
class WebContentCapture(Protocol):
    async def capture(
        self,
        request: WebContentCaptureRequest,
    ) -> CapturedWebContent: ...
