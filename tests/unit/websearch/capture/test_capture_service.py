# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any

import pytest

from intergrax.websearch.capture.contracts import (
    CapturedWebContent,
    WebContentCaptureError,
    WebContentCaptureErrorCode,
    WebContentCaptureRequest,
)
from intergrax.websearch.capture.http_transport import (
    ApprovedHttpsRequest,
    RawHttpsResponse,
)
from intergrax.websearch.capture.service import SecureHttpWebContentCapture
from intergrax.websearch.capture.url_policy import WebUrlAccessPolicy

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def _public_resolver(_hostname: str) -> tuple[str, ...]:
    return ("93.184.216.34",)


async def _private_resolver(_hostname: str) -> tuple[str, ...]:
    return ("192.168.1.1",)


class RecordingTransport:
    def __init__(self, responses: list[RawHttpsResponse]) -> None:
        self._responses = list(responses)
        self.fetch_count = 0
        self.last_request: ApprovedHttpsRequest | None = None

    async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse:
        self.fetch_count += 1
        self.last_request = request
        if not self._responses:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)
        return self._responses.pop(0)


def _html(title: str, body: str) -> bytes:
    return f"<html><head><title>{title}</title></head><body><p>{body}</p></body></html>".encode()


def _service(
    transport: RecordingTransport,
    *,
    resolver: Any = _public_resolver,
) -> SecureHttpWebContentCapture:
    policy = WebUrlAccessPolicy(dns_resolver=resolver)
    return SecureHttpWebContentCapture(policy=policy, transport=transport)


async def test_html_success() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/html; charset=utf-8"},
                body=_html("Title", "Hello world"),
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/page")
    )
    assert result.title == "Title"
    assert "Hello world" in result.text
    assert result.content_type == "text/html"
    assert result.capture_mode == "http"


async def test_xhtml_success() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "application/xhtml+xml"},
                body=_html("X", "XHTML body"),
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/page")
    )
    assert "XHTML body" in result.text
    assert result.content_type == "application/xhtml+xml"


async def test_plain_text_success() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"plain line one\nplain line two",
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/page")
    )
    assert "plain line one" in result.text
    assert result.extraction_method == "plain"


async def test_charset_from_content_type() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain; charset=latin-1"},
                body="café".encode("latin-1"),
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/page")
    )
    assert "café" in result.text


async def test_unsupported_mime() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                body=b"{}",
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_TYPE_UNSUPPORTED


async def test_missing_content_type() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={},
                body=b"data",
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_TYPE_UNSUPPORTED


async def test_http_4xx() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=404,
                headers={"content-type": "text/html"},
                body=_html("Missing", "nope"),
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_HTTP_ERROR
    assert exc.value.status_code == 404


async def test_http_5xx() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=500,
                headers={"content-type": "text/plain"},
                body=b"error",
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_HTTP_ERROR
    assert exc.value.status_code == 500


async def test_empty_body() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"",
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_EMPTY_CONTENT


async def test_empty_extracted_text() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/html"},
                body=b"<html><body></body></html>",
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_EMPTY_CONTENT


async def test_relative_redirect() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=302,
                headers={"location": "/next"},
                body=b"",
                content_bytes=0,
            ),
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"after redirect",
                content_bytes=0,
            ),
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/start")
    )
    assert result.redirect_count == 1
    assert "after redirect" in result.text


async def test_absolute_redirect() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=301,
                headers={"location": "https://example.com/final"},
                body=b"",
                content_bytes=0,
            ),
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"final page",
                content_bytes=0,
            ),
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/start")
    )
    assert result.redirect_count == 1
    assert "final page" in result.text


async def test_multiple_valid_redirects() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(status_code=302, headers={"location": "/two"}, body=b"", content_bytes=0),
            RawHttpsResponse(status_code=303, headers={"location": "/three"}, body=b"", content_bytes=0),
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"done",
                content_bytes=0,
            ),
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/one")
    )
    assert result.redirect_count == 2


async def test_redirect_host_change() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=302,
                headers={"location": "https://other.example.com/page"},
                body=b"",
                content_bytes=0,
            ),
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"other host",
                content_bytes=0,
            ),
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/start")
    )
    assert result.final_host_changed is True


async def test_redirect_limit() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(status_code=302, headers={"location": "/a"}, body=b"", content_bytes=0),
            RawHttpsResponse(status_code=302, headers={"location": "/b"}, body=b"", content_bytes=0),
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(
            WebContentCaptureRequest(
                url="https://example.com/start",
                max_redirects=1,
            )
        )
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_REDIRECT_LIMIT_EXCEEDED


async def test_redirect_missing_location() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(status_code=302, headers={}, body=b"", content_bytes=0),
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/start"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_REDIRECT_LOCATION_MISSING


async def test_redirect_to_http() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=302,
                headers={"location": "http://example.com/insecure"},
                body=b"",
                content_bytes=0,
            ),
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/start"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED
    assert transport.fetch_count == 1


async def test_redirect_to_private_address() -> None:
    async def _redirect_private(hostname: str) -> tuple[str, ...]:
        if hostname == "other.example.com":
            return ("192.168.1.1",)
        return ("93.184.216.34",)

    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=302,
                headers={"location": "https://other.example.com/page"},
                body=b"",
                content_bytes=0,
            ),
        ]
    )
    service = SecureHttpWebContentCapture(
        policy=WebUrlAccessPolicy(dns_resolver=_redirect_private),
        transport=transport,
    )
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/start"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED
    assert transport.fetch_count == 1


async def test_redirect_to_localhost() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=302,
                headers={"location": "https://localhost/page"},
                body=b"",
                content_bytes=0,
            ),
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/start"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED
    assert transport.fetch_count == 1


async def test_redirect_to_forbidden_port() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=302,
                headers={"location": "https://example.com:8443/page"},
                body=b"",
                content_bytes=0,
            ),
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/start"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED
    assert transport.fetch_count == 1


async def test_transport_not_called_for_blocked_target() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_private_resolver)
    transport = RecordingTransport([])
    service = SecureHttpWebContentCapture(policy=policy, transport=transport)
    with pytest.raises(WebContentCaptureError):
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert transport.fetch_count == 0


async def test_basic_extraction_mode() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/html"},
                body=_html("Basic", "Basic body"),
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(
            url="https://example.com/page",
            extraction_mode="basic",
        )
    )
    assert result.extraction_method == "beautifulsoup"


async def test_max_extracted_chars() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"x" * 2000,
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(
            url="https://example.com/page",
            max_extracted_chars=1024,
        )
    )
    assert result.text_chars == 1024
    assert len(result.text) == 1024


async def test_deterministic_content_hash() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"hash me",
                content_bytes=0,
            ),
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"hash me",
                content_bytes=0,
            ),
        ]
    )
    service = _service(transport)
    first = await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    second = await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert first.content_hash == second.content_hash
    assert first.content_hash.startswith("sha256:")


async def test_no_raw_url_in_result() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"safe",
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/secret?q=token")
    )
    dumped = result.model_dump()
    serialized = str(dumped)
    assert "token" not in serialized
    assert "?" not in result.safe_display_url


async def test_no_secret_data_in_errors() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=302,
                headers={"location": "https://localhost/private"},
                body=b"",
                content_bytes=0,
            ),
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(
            WebContentCaptureRequest(url="https://example.com/secret?q=token")
        )
    assert str(exc.value) == WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED.value
    assert repr(exc.value) == "WebContentCaptureError(code='web_url_redirect_target_blocked')"
    assert "token" not in str(exc.value)
    assert "localhost" not in str(exc.value)


async def test_timeout() -> None:
    class SlowTransport:
        async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_TIMEOUT,
                retryable=True,
            )

    service = SecureHttpWebContentCapture(
        policy=WebUrlAccessPolicy(dns_resolver=_public_resolver),
        transport=SlowTransport(),
    )
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_TIMEOUT


async def test_tls_failure() -> None:
    class FailingTransport:
        async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_TLS_FAILED)

    service = SecureHttpWebContentCapture(
        policy=WebUrlAccessPolicy(dns_resolver=_public_resolver),
        transport=FailingTransport(),
    )
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_TLS_FAILED


async def test_generic_transport_failure() -> None:
    class FailingTransport:
        async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED)

    service = SecureHttpWebContentCapture(
        policy=WebUrlAccessPolicy(dns_resolver=_public_resolver),
        transport=FailingTransport(),
    )
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED


async def test_oversized_content_length() -> None:
    class TooLargeTransport:
        async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE,
            )

    service = SecureHttpWebContentCapture(
        policy=WebUrlAccessPolicy(dns_resolver=_public_resolver),
        transport=TooLargeTransport(),
    )
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_TOO_LARGE


async def test_captured_result_has_no_html_headers_ip() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={
                    "content-type": "text/html",
                    "x-secret": "header-value",
                    "set-cookie": "session=abc",
                },
                body=_html("T", "Body"),
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/page")
    )
    dumped = result.model_dump()
    assert "<html" not in str(dumped).lower()
    assert "header-value" not in str(dumped)
    assert "93.184" not in str(dumped)
    assert isinstance(result, CapturedWebContent)


async def test_invalid_port_never_leaks_value_error() -> None:
    service = _service(RecordingTransport([]))
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(
            WebContentCaptureRequest(url="https://example.com:abc/page"),
        )
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_INVALID
    assert "ValueError" not in str(exc.value)


async def test_unicode_path_never_leaks_unicode_encode_error() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"unicode path ok",
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/café"),
    )
    assert "unicode path ok" in result.text
    assert "UnicodeEncodeError" not in str(result.model_dump())


async def test_unexpected_transport_exception_maps_to_safe_error() -> None:
    class ExplodingTransport:
        async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse:
            raise RuntimeError("secret transport detail")

    service = SecureHttpWebContentCapture(
        policy=WebUrlAccessPolicy(dns_resolver=_public_resolver),
        transport=ExplodingTransport(),
    )
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_FETCH_FAILED
    assert "secret" not in str(exc.value)


async def test_title_controls_removed() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/html"},
                body=_html("Ti\x07tle\nwith\ttab", "Body"),
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/page"),
    )
    assert "\x07" not in result.title
    assert "\n" not in result.title
    assert "\t" not in result.title
    assert "Title with tab" in result.title


async def test_text_controls_removed() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"line\x07one\nline\ttwo",
                content_bytes=0,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/page"),
    )
    assert "\x07" not in result.text
    assert "\t" not in result.text
    dumped = str(result.model_dump())
    assert "\x07" not in dumped


async def test_content_bytes_matches_body_length() -> None:
    body = b"1234567890"
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=body,
                content_bytes=999,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/page"),
    )
    assert result.content_bytes == len(body)


async def test_content_encoding_gzip_rejected_at_service() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={
                    "content-type": "text/plain",
                    "content-encoding": "gzip",
                },
                body=b"ok",
                content_bytes=2,
            )
        ]
    )
    service = _service(transport)
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CONTENT_ENCODING_UNSUPPORTED


async def test_content_encoding_absent_accepted() -> None:
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"plain",
                content_bytes=5,
            )
        ]
    )
    service = _service(transport)
    result = await service.capture(WebContentCaptureRequest(url="https://example.com/page"))
    assert "plain" in result.text


class FakeMonotonic:
    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


async def test_dns_exceeds_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolver(_hostname: str) -> tuple[str, ...]:
        return ("93.184.216.34",)

    async def instant_timeout(coro: object, timeout: float) -> object:
        if hasattr(coro, "close"):
            coro.close()  # type: ignore[union-attr]
        raise asyncio.TimeoutError

    import asyncio

    monkeypatch.setattr(asyncio, "wait_for", instant_timeout)
    transport = RecordingTransport([])
    service = SecureHttpWebContentCapture(
        policy=WebUrlAccessPolicy(dns_resolver=resolver),
        transport=transport,
    )
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(
            WebContentCaptureRequest(url="https://example.com/page", timeout_seconds=5),
        )
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_TIMEOUT
    assert transport.fetch_count == 0


async def test_redirect_chain_shares_one_deadline() -> None:
    clock = FakeMonotonic(0.0)
    deadlines: list[float] = []

    class DeadlineTransport:
        async def fetch(self, request: ApprovedHttpsRequest) -> RawHttpsResponse:
            deadlines.append(request.deadline)
            if len(deadlines) == 1:
                return RawHttpsResponse(
                    status_code=302,
                    headers={"location": "/two"},
                    body=b"",
                    content_bytes=0,
                )
            return RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/plain"},
                body=b"done",
                content_bytes=4,
            )

    service = SecureHttpWebContentCapture(
        policy=WebUrlAccessPolicy(dns_resolver=_public_resolver),
        transport=DeadlineTransport(),
        monotonic=clock,
    )
    result = await service.capture(
        WebContentCaptureRequest(url="https://example.com/one", timeout_seconds=20),
    )
    assert result.redirect_count == 1
    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1] == 20.0


async def test_extraction_exceeds_remaining_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    call_count = 0

    async def timeout_on_extraction(coro: object, timeout: float) -> object:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            if hasattr(coro, "close"):
                coro.close()  # type: ignore[union-attr]
            raise asyncio.TimeoutError
        return await asyncio.wait_for(coro, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", timeout_on_extraction)
    transport = RecordingTransport(
        [
            RawHttpsResponse(
                status_code=200,
                headers={"content-type": "text/html"},
                body=_html("T", "Body"),
                content_bytes=0,
            )
        ]
    )
    service = SecureHttpWebContentCapture(
        policy=WebUrlAccessPolicy(dns_resolver=_public_resolver),
        transport=transport,
    )
    with pytest.raises(WebContentCaptureError) as exc:
        await service.capture(
            WebContentCaptureRequest(url="https://example.com/page", timeout_seconds=5),
        )
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_TIMEOUT
