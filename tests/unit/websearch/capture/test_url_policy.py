# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.websearch.capture.contracts import (
    WebContentCaptureError,
    WebContentCaptureErrorCode,
)
from intergrax.websearch.capture.http_transport import ApprovedHttpsRequest
from intergrax.websearch.capture.url_policy import WebUrlAccessPolicy

pytestmark = pytest.mark.unit


async def _public_resolver(_hostname: str) -> tuple[str, ...]:
    return ("93.184.216.34",)


async def _private_resolver(_hostname: str) -> tuple[str, ...]:
    return ("192.168.1.1",)


async def _mixed_resolver(_hostname: str) -> tuple[str, ...]:
    return ("93.184.216.34", "10.0.0.1")


async def _empty_resolver(_hostname: str) -> tuple[str, ...]:
    return ()


async def _failing_resolver(_hostname: str) -> tuple[str, ...]:
    raise OSError("resolution failed")


async def _duplicate_resolver(_hostname: str) -> tuple[str, ...]:
    return (
        "93.184.216.34",
        "2606:2800:220:1:248:1893:25c8:1946",
        "93.184.216.34",
    )


def test_valid_https_url() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com/page")
    assert canonical.hostname == "example.com"
    assert canonical.request_target == "/page"
    assert canonical.safe_display_url == "https://example.com/page"


def test_hostname_lower_case_normalization() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://EXAMPLE.COM/Page")
    assert canonical.hostname == "example.com"
    assert canonical.request_target == "/Page"


def test_idna_hostname_normalization() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://bücher.example.com/path")
    assert canonical.hostname == "xn--bcher-kva.example.com"


def test_empty_path_normalized_to_slash() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com")
    assert canonical.request_target == "/"
    assert canonical.safe_display_url == "https://example.com"


def test_default_port_removed() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com:443/docs")
    assert ":443" not in canonical.canonical_private_url
    assert canonical.port == 443


def test_non_default_allowed_port_preserved() -> None:
    policy = WebUrlAccessPolicy(
        dns_resolver=_public_resolver,
        allowed_ports=frozenset({8443}),
    )
    canonical = policy.canonicalize("https://example.com:8443/docs")
    assert canonical.port == 8443
    assert ":8443" in canonical.canonical_private_url
    assert canonical.safe_display_url == "https://example.com:8443/docs"


def test_fragment_removed() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com/docs#section")
    assert "#" not in canonical.canonical_private_url
    assert canonical.request_target == "/docs"


def test_query_preserved_in_fingerprint() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with_query = policy.canonicalize("https://example.com/docs?q=1")
    without_query = policy.canonicalize("https://example.com/docs")
    assert with_query.fingerprint != without_query.fingerprint
    assert with_query.request_target == "/docs?q=1"


def test_query_absent_in_safe_display() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com/docs?q=secret")
    assert "?" not in canonical.safe_display_url
    assert "secret" not in canonical.safe_display_url


def test_userinfo_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://user@example.com/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CREDENTIALS_NOT_ALLOWED


def test_empty_userinfo_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://@example.com/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CREDENTIALS_NOT_ALLOWED


def test_userinfo_with_password_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://user:pass@example.com/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_CREDENTIALS_NOT_ALLOWED


def test_http_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("http://example.com/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_SCHEME_NOT_ALLOWED


def test_ftp_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("ftp://example.com/file")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_SCHEME_NOT_ALLOWED


def test_custom_port_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://example.com:8443/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_PORT_NOT_ALLOWED


def test_invalid_textual_port_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://example.com:abc/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_INVALID


def test_out_of_range_port_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://example.com:99999/")
    assert exc.value.code in {
        WebContentCaptureErrorCode.WEB_URL_INVALID,
        WebContentCaptureErrorCode.WEB_URL_PORT_NOT_ALLOWED,
    }


def test_missing_hostname_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https:///path")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_INVALID


def test_control_characters_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://example.com/\x07bad")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_INVALID


@pytest.mark.parametrize(
    "raw_url",
    [
        "https://example.com/\tbad",
        "https://example.com/\nbad",
        "https://example.com/\rbad",
        "\nhttps://example.com/",
        "https://example.com/\t",
    ],
)
def test_whitespace_control_chars_rejected(raw_url: str) -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize(raw_url)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_INVALID


def test_overlong_url_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver, max_url_length=50)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://example.com/" + ("a" * 100))
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_INVALID


def test_ip_literal_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://93.184.216.34/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_HOST_NOT_ALLOWED


def test_localhost_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://localhost/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_HOST_NOT_ALLOWED


def test_local_tld_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://host.local/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_HOST_NOT_ALLOWED


def test_internal_tld_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://svc.internal/")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_HOST_NOT_ALLOWED


def test_unicode_path_percent_encoded() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com/café")
    assert canonical.request_target == "/caf%C3%A9"
    canonical.request_target.encode("ascii")


def test_unicode_query_percent_encoded() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com/?q=café")
    assert canonical.request_target == "/?q=caf%C3%A9"
    canonical.request_target.encode("ascii")


def test_space_in_path_encoded() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com/a b")
    assert canonical.request_target == "/a%20b"


def test_space_in_query_encoded() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com/?a=b c")
    assert canonical.request_target == "/?a=b%20c"


def test_invalid_percent_escape_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://example.com/%ZZ")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_INVALID


def test_request_target_always_ascii() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com/über?q=naïve")
    canonical.request_target.encode("ascii")
    assert "%" in canonical.request_target


def test_host_allowlist_normalized() -> None:
    policy = WebUrlAccessPolicy(
        dns_resolver=_public_resolver,
        host_allowlist=frozenset({" EXAMPLE.COM. "}),
    )
    canonical = policy.canonicalize("https://example.com/page")
    assert canonical.hostname == "example.com"


def test_host_allowlist_rejects_non_member() -> None:
    policy = WebUrlAccessPolicy(
        dns_resolver=_public_resolver,
        host_allowlist=frozenset({"allowed.example.com"}),
    )
    with pytest.raises(WebContentCaptureError) as exc:
        policy.canonicalize("https://example.com/page")
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_HOST_NOT_ALLOWED


@pytest.mark.asyncio
async def test_single_public_ipv4_accepted() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize("https://example.com/")
    approved = await policy.approve_target(canonical)
    assert approved.approved_ips == ("93.184.216.34",)


@pytest.mark.asyncio
async def test_single_public_ipv6_accepted() -> None:
    async def _ipv6(_hostname: str) -> tuple[str, ...]:
        return ("2606:2800:220:1:248:1893:25c8:1946",)

    policy = WebUrlAccessPolicy(dns_resolver=_ipv6)
    canonical = policy.canonicalize("https://example.com/")
    approved = await policy.approve_target(canonical)
    assert approved.approved_ips == ("2606:2800:220:1:248:1893:25c8:1946",)


@pytest.mark.asyncio
async def test_resolver_output_deduplicated_and_sorted() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_duplicate_resolver)
    canonical = policy.canonicalize("https://example.com/")
    approved = await policy.approve_target(canonical)
    assert approved.approved_ips == (
        "2606:2800:220:1:248:1893:25c8:1946",
        "93.184.216.34",
    )


@pytest.mark.asyncio
async def test_private_ipv4_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_private_resolver)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED


@pytest.mark.asyncio
async def test_loopback_rejected() -> None:
    async def _loopback(_hostname: str) -> tuple[str, ...]:
        return ("127.0.0.1",)

    policy = WebUrlAccessPolicy(dns_resolver=_loopback)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED


@pytest.mark.asyncio
async def test_link_local_rejected() -> None:
    async def _link_local(_hostname: str) -> tuple[str, ...]:
        return ("169.254.1.1",)

    policy = WebUrlAccessPolicy(dns_resolver=_link_local)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED


@pytest.mark.asyncio
async def test_reserved_rejected() -> None:
    async def _reserved(_hostname: str) -> tuple[str, ...]:
        return ("240.0.0.1",)

    policy = WebUrlAccessPolicy(dns_resolver=_reserved)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED


@pytest.mark.asyncio
async def test_multicast_rejected() -> None:
    async def _multicast(_hostname: str) -> tuple[str, ...]:
        return ("224.0.0.1",)

    policy = WebUrlAccessPolicy(dns_resolver=_multicast)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED


@pytest.mark.asyncio
async def test_unspecified_rejected() -> None:
    async def _unspecified(_hostname: str) -> tuple[str, ...]:
        return ("0.0.0.0",)

    policy = WebUrlAccessPolicy(dns_resolver=_unspecified)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED


@pytest.mark.asyncio
async def test_ipv4_mapped_private_ipv6_rejected() -> None:
    async def _mapped(_hostname: str) -> tuple[str, ...]:
        return ("::ffff:192.168.1.1",)

    policy = WebUrlAccessPolicy(dns_resolver=_mapped)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED


@pytest.mark.asyncio
async def test_mixed_public_private_result_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_mixed_resolver)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED


@pytest.mark.asyncio
async def test_empty_resolver_result_rejected() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_empty_resolver)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESOLUTION_FAILED


@pytest.mark.asyncio
async def test_resolution_exception_normalized() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_failing_resolver)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError) as exc:
        await policy.approve_target(canonical)
    assert exc.value.code == WebContentCaptureErrorCode.WEB_URL_RESOLUTION_FAILED


@pytest.mark.asyncio
async def test_transport_not_called_after_rejection() -> None:
    fetch_count: list[int] = []
    policy = WebUrlAccessPolicy(dns_resolver=_private_resolver)
    canonical = policy.canonicalize("https://example.com/")
    with pytest.raises(WebContentCaptureError):
        await policy.approve_target(canonical)
    assert fetch_count == []


_PRIVATE_URL = "https://example.com/private?q=secret-token"
_PRIVATE_IP = "93.184.216.34"


def test_canonical_url_repr_hides_private_url() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize(_PRIVATE_URL)
    rendered = repr(canonical)
    assert "secret-token" not in rendered
    assert "?q=" not in rendered
    assert "private" not in rendered
    assert canonical.hostname in rendered
    assert canonical.fingerprint in rendered


@pytest.mark.asyncio
async def test_approved_target_repr_hides_query_and_ip() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize(_PRIVATE_URL)
    approved = await policy.approve_target(canonical)
    rendered = repr(approved)
    assert "secret-token" not in rendered
    assert "?q=" not in rendered
    assert _PRIVATE_IP not in rendered
    assert approved.hostname in rendered
    assert canonical.fingerprint in rendered


def test_approved_https_request_repr_hides_query_and_ip() -> None:
    policy = WebUrlAccessPolicy(dns_resolver=_public_resolver)
    canonical = policy.canonicalize(_PRIVATE_URL)
    request = ApprovedHttpsRequest(
        hostname=canonical.hostname,
        port=canonical.port,
        request_target=canonical.request_target,
        approved_ips=(_PRIVATE_IP,),
        deadline=1.0,
        max_response_bytes=1024,
    )
    rendered = repr(request)
    assert "secret-token" not in rendered
    assert "?q=" not in rendered
    assert _PRIVATE_IP not in rendered
    assert "approved_ip_count=1" in rendered
