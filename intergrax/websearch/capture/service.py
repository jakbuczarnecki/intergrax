# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from urllib.parse import urljoin

from intergrax.websearch.capture.contracts import (
    CapturedWebContent,
    WebContentCaptureError,
    WebContentCaptureErrorCode,
    WebContentCaptureRequest,
)
from intergrax.websearch.capture.http_transport import (
    ApprovedHttpsRequest,
    HttpsTransport,
    PinnedHttpsTransport,
    RawHttpsResponse,
)
from intergrax.websearch.capture.url_policy import (
    DnsResolver,
    WebUrlAccessPolicy,
)
from intergrax.websearch.fetcher.extractor import extract_advanced, extract_basic
from intergrax.websearch.schemas.page_content import PageContent

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_MIME_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "text/plain"},
)
_WHITESPACE_RE = re.compile(r"\s+")


def _content_hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _normalize_mime(content_type: str | None) -> str | None:
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower()


def _extract_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _decode_body(body: bytes, content_type: str | None) -> str:
    charset = _extract_charset(content_type) or "utf-8"
    try:
        return body.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


def _normalize_plain_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n")]
    collapsed: list[str] = []
    for line in lines:
        if not line:
            if collapsed and collapsed[-1]:
                collapsed.append("")
            continue
        collapsed.append(_WHITESPACE_RE.sub(" ", line))
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return "\n".join(collapsed)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _safe_title(title: str | None) -> str:
    if not title:
        return ""
    cleaned = _WHITESPACE_RE.sub(" ", title.strip())
    return cleaned[:500]


def _single_location(headers: Mapping[str, str]) -> str | None:
    location = headers.get("location")
    if not location or not location.strip():
        return None
    return location.strip()


def _extraction_method_from_page(page: PageContent, mode: str) -> str:
    if mode == "basic":
        return "beautifulsoup"
    extra = page.extra or {}
    extraction_info = extra.get("advanced_extraction", {})
    method = extraction_info.get("extraction_method")
    if isinstance(method, str) and method:
        return method
    return "beautifulsoup"


class SecureHttpWebContentCapture:
    def __init__(
        self,
        *,
        policy: WebUrlAccessPolicy | None = None,
        transport: HttpsTransport | None = None,
        dns_resolver: DnsResolver | None = None,
        clock: Callable[[], datetime] | None = None,
        transport_fetch_count: list[int] | None = None,
    ) -> None:
        if policy is not None:
            self._policy = policy
        elif dns_resolver is not None:
            self._policy = WebUrlAccessPolicy(dns_resolver=dns_resolver)
        else:
            self._policy = WebUrlAccessPolicy()
        self._redirect_policy = self._policy.redirect_policy()
        self._transport = transport or PinnedHttpsTransport(fetch_count=transport_fetch_count)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._transport_fetch_count = transport_fetch_count

    async def capture(self, request: WebContentCaptureRequest) -> CapturedWebContent:
        deadline = time.monotonic() + request.timeout_seconds
        requested = self._policy.canonicalize(request.url)
        current = requested
        redirect_count = 0
        initial_host = requested.hostname
        raw: RawHttpsResponse | None = None

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WebContentCaptureError(
                    WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
                    retryable=True,
                )

            active_policy = self._policy if redirect_count == 0 else self._redirect_policy
            approved = await active_policy.approve_target(current)
            current = approved.canonical

            raw = await self._transport.fetch(
                ApprovedHttpsRequest(
                    hostname=approved.hostname,
                    port=approved.port,
                    request_target=approved.request_target,
                    approved_ips=approved.approved_ips,
                    timeout_seconds=remaining,
                    max_response_bytes=request.max_response_bytes,
                )
            )

            if raw.status_code in _REDIRECT_STATUSES:
                if redirect_count >= request.max_redirects:
                    raise WebContentCaptureError(
                        WebContentCaptureErrorCode.WEB_URL_REDIRECT_LIMIT_EXCEEDED,
                    )
                location = _single_location(raw.headers)
                if location is None:
                    raise WebContentCaptureError(
                        WebContentCaptureErrorCode.WEB_URL_REDIRECT_LOCATION_MISSING,
                    )
                redirect_count += 1
                joined = urljoin(current.canonical_private_url, location)
                try:
                    current = self._redirect_policy.canonicalize(joined)
                except WebContentCaptureError:
                    raise WebContentCaptureError(
                        WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED,
                    ) from None
                continue

            break

        if raw is None:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)

        if raw.status_code < 200 or raw.status_code >= 300:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_HTTP_ERROR,
                status_code=raw.status_code,
            )

        mime = _normalize_mime(raw.headers.get("content-type"))
        if mime is None:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_CONTENT_TYPE_UNSUPPORTED,
            )
        if mime not in _ALLOWED_MIME_TYPES:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_CONTENT_TYPE_UNSUPPORTED,
            )

        if not raw.body:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_EMPTY_CONTENT)

        content_type_header = raw.headers.get("content-type", mime)
        try:
            decoded = _decode_body(raw.body, content_type_header)
        except Exception:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_DECODE_FAILED)

        title = ""
        text = ""
        extraction_method = "plain"

        if mime in {"text/html", "application/xhtml+xml"}:
            page = PageContent(
                final_url="",
                status_code=raw.status_code,
                html=decoded,
                text=None,
                title=None,
                description=None,
                lang=None,
            )
            try:
                page = extract_basic(page)
                if request.extraction_mode == "advanced":
                    page = extract_advanced(page)
            except Exception:
                raise WebContentCaptureError(
                    WebContentCaptureErrorCode.WEB_URL_EXTRACTION_FAILED,
                )
            title = _safe_title(page.title)
            text = page.text or ""
            extraction_method = _extraction_method_from_page(page, request.extraction_mode)
        else:
            text = _normalize_plain_text(decoded)
            extraction_method = "plain"

        text = _normalize_text(text)
        text = _truncate_text(text, request.max_extracted_chars)
        if not text.strip():
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_EMPTY_CONTENT)

        final = current
        fetched_at = self._clock()

        return CapturedWebContent(
            safe_display_url=final.safe_display_url,
            requested_url_fingerprint=requested.fingerprint,
            final_url_fingerprint=final.fingerprint,
            final_host_changed=final.hostname != initial_host,
            title=title,
            text=text,
            content_type=mime,
            content_hash=_content_hash(text),
            status_code=raw.status_code,
            redirect_count=redirect_count,
            content_bytes=raw.content_bytes,
            text_chars=len(text),
            capture_mode="http",
            extraction_method=extraction_method,
            fetched_at=fetched_at,
        )
