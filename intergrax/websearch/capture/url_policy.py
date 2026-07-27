# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

from intergrax.websearch.capture.contracts import (
    WebContentCaptureError,
    WebContentCaptureErrorCode,
)

_RAW_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_REQUEST_TARGET_CONTROL_RE = re.compile(r"[\r\n\t]")
_HEX_DIGITS = frozenset("0123456789ABCDEFabcdef")
_DNS_LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)
_SAFE_DISPLAY_MAX_LEN = 300
_DEFAULT_MAX_URL_LENGTH = 2048
_PATH_SAFE_CHARS = frozenset("-._~/")
_QUERY_SAFE_CHARS = frozenset("-._~&=")


@dataclass(frozen=True, slots=True)
class CanonicalUrl:
    canonical_private_url: str
    safe_display_url: str
    fingerprint: str
    hostname: str
    port: int
    request_target: str
    scheme: str

    def __repr__(self) -> str:
        return (
            f"CanonicalUrl(hostname={self.hostname!r}, port={self.port}, "
            f"fingerprint={self.fingerprint!r})"
        )


@dataclass(frozen=True, slots=True)
class ApprovedTarget:
    hostname: str
    port: int
    request_target: str
    approved_ips: tuple[str, ...]
    canonical: CanonicalUrl

    def __repr__(self) -> str:
        return (
            f"ApprovedTarget(hostname={self.hostname!r}, port={self.port}, "
            f"approved_ip_count={len(self.approved_ips)}, "
            f"fingerprint={self.canonical.fingerprint!r})"
        )


DnsResolver = Callable[[str], Awaitable[tuple[str, ...]]]


def _url_fingerprint(canonical_private_url: str) -> str:
    digest = hashlib.sha256(canonical_private_url.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _default_port_for_scheme(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _format_authority(hostname: str, port: int, scheme: str) -> str:
    if port == _default_port_for_scheme(scheme):
        return hostname
    return f"{hostname}:{port}"


def _build_safe_display_url(scheme: str, hostname: str, port: int, path: str) -> str:
    authority = _format_authority(hostname, port, scheme)
    base = f"{scheme}://{authority}"
    if not path or path == "/":
        display = base
    else:
        display = f"{base}{path}"
    if len(display) <= _SAFE_DISPLAY_MAX_LEN:
        return display
    if len(base) >= _SAFE_DISPLAY_MAX_LEN:
        return base[:_SAFE_DISPLAY_MAX_LEN]
    remaining = _SAFE_DISPLAY_MAX_LEN - len(base) - 3
    if remaining < 1:
        return base
    return f"{base}{path[:remaining]}..."


def _normalize_hostname(raw_host: str) -> str:
    host = raw_host.strip().lower()
    if host.endswith("."):
        host = host[:-1]
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)
    return host


def _normalize_allowlist_host(raw_host: str) -> str:
    host = _normalize_hostname(raw_host)
    if _is_blocked_hostname(host) or not _is_valid_dns_hostname(host):
        raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)
    return host


def _is_blocked_hostname(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    if hostname.endswith(".localhost"):
        return True
    if hostname.endswith(".local"):
        return True
    if hostname.endswith(".internal"):
        return True
    return False


def _is_valid_dns_hostname(hostname: str) -> bool:
    if not hostname or len(hostname) > 253:
        return False
    labels = hostname.split(".")
    if len(labels) < 2:
        return False
    return all(_DNS_LABEL_RE.match(label) for label in labels)


def _is_global_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return (
        addr.is_global
        and not addr.is_private
        and not addr.is_loopback
        and not addr.is_link_local
        and not addr.is_multicast
        and not addr.is_reserved
    )


def _is_global_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return _is_global_address(addr)


def _percent_encode_component(component: str, *, safe: frozenset[str]) -> str:
    if _REQUEST_TARGET_CONTROL_RE.search(component):
        raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)

    result: list[str] = []
    index = 0
    while index < len(component):
        char = component[index]
        if char == "%":
            if index + 2 >= len(component):
                raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)
            hi = component[index + 1]
            lo = component[index + 2]
            if hi not in _HEX_DIGITS or lo not in _HEX_DIGITS:
                raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)
            result.append(f"%{hi.upper()}{lo.upper()}")
            index += 3
            continue
        if char == " ":
            result.append("%20")
        elif char.isascii() and (char.isalnum() or char in safe):
            result.append(char)
        else:
            for byte in char.encode("utf-8"):
                result.append(f"%{byte:02X}")
        index += 1

    encoded = "".join(result)
    encoded.encode("ascii")
    return encoded


def _normalize_path(path: str) -> str:
    if not path:
        return "/"
    if not path.startswith("/"):
        path = f"/{path}"
    return _percent_encode_component(path, safe=_PATH_SAFE_CHARS)


def _normalize_query(query: str) -> str:
    if not query:
        return ""
    return _percent_encode_component(query, safe=_QUERY_SAFE_CHARS)


def _normalize_request_target(path: str, query: str) -> str:
    normalized_path = _normalize_path(path)
    normalized_query = _normalize_query(query)
    if normalized_query:
        request_target = f"{normalized_path}?{normalized_query}"
    else:
        request_target = normalized_path
    request_target.encode("ascii")
    return request_target


def _normalize_resolver_ips(
    resolved_ips: tuple[str, ...],
    *,
    is_redirect: bool,
) -> tuple[str, ...]:
    normalized: set[str] = set()
    for ip_str in resolved_ips:
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESOLUTION_FAILED,
            )
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        if not _is_global_address(addr):
            code = (
                WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED
                if is_redirect
                else WebContentCaptureErrorCode.WEB_URL_NON_GLOBAL_ADDRESS_BLOCKED
            )
            raise WebContentCaptureError(code)
        normalized.add(str(addr))
    return tuple(sorted(normalized))


async def _default_dns_resolver(hostname: str) -> tuple[str, ...]:
    def _resolve() -> list[str]:
        try:
            results = socket.getaddrinfo(
                hostname,
                None,
                type=socket.SOCK_STREAM,
            )
        except OSError:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESOLUTION_FAILED,
            )
        ips: set[str] = set()
        for family, _type, _proto, _canonname, sockaddr in results:
            if family == socket.AF_INET:
                ips.add(sockaddr[0])
            elif family == socket.AF_INET6:
                ips.add(sockaddr[0])
        return sorted(ips)

    ips = await asyncio.to_thread(_resolve)
    return tuple(ips)


class WebUrlAccessPolicy:
    def __init__(
        self,
        *,
        allowed_schemes: frozenset[str] = frozenset({"https"}),
        allowed_ports: frozenset[int] = frozenset({443}),
        max_url_length: int = _DEFAULT_MAX_URL_LENGTH,
        host_allowlist: frozenset[str] = frozenset(),
        dns_resolver: DnsResolver | None = None,
        is_redirect: bool = False,
    ) -> None:
        self._allowed_schemes = allowed_schemes
        self._allowed_ports = allowed_ports
        self._max_url_length = max_url_length
        self._host_allowlist = frozenset(
            _normalize_allowlist_host(host) for host in host_allowlist
        )
        self._dns_resolver = dns_resolver or _default_dns_resolver
        self._is_redirect = is_redirect

    def canonicalize(self, raw_url: str) -> CanonicalUrl:
        if _RAW_CONTROL_CHAR_RE.search(raw_url):
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)

        trimmed = raw_url.strip()
        if not trimmed or len(trimmed) > self._max_url_length:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)

        try:
            parts = urlsplit(trimmed)
        except ValueError:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)

        if not parts.scheme:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)

        scheme = parts.scheme.lower()
        if scheme not in self._allowed_schemes:
            code = (
                WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED
                if self._is_redirect
                else WebContentCaptureErrorCode.WEB_URL_SCHEME_NOT_ALLOWED
            )
            raise WebContentCaptureError(code)

        if "@" in parts.netloc:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_CREDENTIALS_NOT_ALLOWED,
            )

        if not parts.hostname:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)

        hostname = _normalize_hostname(parts.hostname)

        try:
            ipaddress.ip_address(hostname)
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_HOST_NOT_ALLOWED)
        except ValueError:
            pass

        if _is_blocked_hostname(hostname) or not _is_valid_dns_hostname(hostname):
            code = (
                WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED
                if self._is_redirect
                else WebContentCaptureErrorCode.WEB_URL_HOST_NOT_ALLOWED
            )
            raise WebContentCaptureError(code)

        if self._host_allowlist and hostname not in self._host_allowlist:
            code = (
                WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED
                if self._is_redirect
                else WebContentCaptureErrorCode.WEB_URL_HOST_NOT_ALLOWED
            )
            raise WebContentCaptureError(code)

        try:
            port = parts.port
        except ValueError:
            raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_INVALID)

        if port is None:
            port = _default_port_for_scheme(scheme)
        if port not in self._allowed_ports:
            code = (
                WebContentCaptureErrorCode.WEB_URL_REDIRECT_TARGET_BLOCKED
                if self._is_redirect
                else WebContentCaptureErrorCode.WEB_URL_PORT_NOT_ALLOWED
            )
            raise WebContentCaptureError(code)

        path = parts.path or "/"
        request_target = _normalize_request_target(path, parts.query)
        authority = _format_authority(hostname, port, scheme)
        normalized_path = request_target.split("?", 1)[0]
        normalized_query = request_target.split("?", 1)[1] if "?" in request_target else ""
        canonical_private_url = urlunsplit(
            (scheme, authority, normalized_path, normalized_query, ""),
        )
        safe_display_url = _build_safe_display_url(scheme, hostname, port, normalized_path)
        fingerprint = _url_fingerprint(canonical_private_url)

        return CanonicalUrl(
            canonical_private_url=canonical_private_url,
            safe_display_url=safe_display_url,
            fingerprint=fingerprint,
            hostname=hostname,
            port=port,
            request_target=request_target,
            scheme=scheme,
        )

    async def approve_target(self, canonical: CanonicalUrl) -> ApprovedTarget:
        try:
            resolved_ips = await self._dns_resolver(canonical.hostname)
        except WebContentCaptureError:
            raise
        except Exception:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESOLUTION_FAILED,
            )

        if not resolved_ips:
            raise WebContentCaptureError(
                WebContentCaptureErrorCode.WEB_URL_RESOLUTION_FAILED,
            )

        approved_ips = _normalize_resolver_ips(
            resolved_ips,
            is_redirect=self._is_redirect,
        )

        return ApprovedTarget(
            hostname=canonical.hostname,
            port=canonical.port,
            request_target=canonical.request_target,
            approved_ips=approved_ips,
            canonical=canonical,
        )

    def redirect_policy(self) -> WebUrlAccessPolicy:
        return WebUrlAccessPolicy(
            allowed_schemes=self._allowed_schemes,
            allowed_ports=self._allowed_ports,
            max_url_length=self._max_url_length,
            host_allowlist=self._host_allowlist,
            dns_resolver=self._dns_resolver,
            is_redirect=True,
        )
