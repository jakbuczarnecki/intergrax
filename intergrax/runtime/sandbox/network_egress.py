# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed sandbox network egress host-scope contracts (AW-7C P0-1)."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence
from urllib.parse import urlsplit

NetworkEgressScheme = Literal["http", "https"]

_DNS_LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)
_SHELL_OR_PATTERN_CHARS = frozenset("*?[]{}|^$\\")
_METADATA_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "169.254.169.254",
    },
)


class NetworkEgressScopeError(ValueError):
    """Invalid or disallowed network egress host scope."""


@dataclass(frozen=True, slots=True)
class NetworkEgressHost:
    """Canonical exact destination host scope — no wildcards in V1."""

    scheme: NetworkEgressScheme
    hostname: str
    port: int

    def canonical_authority(self) -> str:
        default_port = 443 if self.scheme == "https" else 80
        if self.port == default_port:
            return self.hostname
        return f"{self.hostname}:{self.port}"

    def canonical_form(self) -> str:
        return f"{self.scheme}://{self.canonical_authority()}"

    def fingerprint_part(self) -> str:
        return self.canonical_form()


@dataclass(frozen=True, slots=True)
class NetworkEgressAllowlist:
    """Deterministically ordered exact host allowlist."""

    hosts: tuple[NetworkEgressHost, ...]

    def fingerprint(self) -> str:
        if not self.hosts:
            return "sha256:" + hashlib.sha256(b"").hexdigest()
        canonical = "\n".join(host.fingerprint_part() for host in self.hosts)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


def _default_port_for_scheme(scheme: NetworkEgressScheme) -> int:
    return 443 if scheme == "https" else 80


def _normalize_hostname(raw_host: str) -> str:
    host = raw_host.strip().lower()
    if not host:
        raise NetworkEgressScopeError("hostname must not be empty")
    if host.endswith("."):
        host = host[:-1]
    if any(char in host for char in _SHELL_OR_PATTERN_CHARS):
        raise NetworkEgressScopeError("hostname must not contain shell or pattern syntax")
    if "@" in host or "/" in host or "?" in host or "#" in host:
        raise NetworkEgressScopeError("hostname must not embed credentials or URL fragments")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise NetworkEgressScopeError("hostname is not valid IDNA") from exc
    return host


def _is_blocked_hostname(hostname: str) -> bool:
    if hostname in _METADATA_HOSTNAMES:
        return True
    if hostname == "localhost":
        return True
    if hostname.endswith(".localhost"):
        return True
    if hostname.endswith(".local"):
        return True
    if hostname.endswith(".internal"):
        return True
    return False


def _is_global_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
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


def _is_valid_dns_hostname(hostname: str) -> bool:
    if not hostname or len(hostname) > 253:
        return False
    if hostname.startswith("*.") or hostname.startswith("."):
        return False
    labels = hostname.split(".")
    if len(labels) < 2:
        return False
    return all(_DNS_LABEL_RE.match(label) for label in labels)


def _reject_disallowed_hostname(hostname: str) -> None:
    if _is_blocked_hostname(hostname):
        raise NetworkEgressScopeError(f"hostname is blocked by platform policy: {hostname}")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        if not _is_valid_dns_hostname(hostname):
            raise NetworkEgressScopeError(f"hostname is not a valid exact host scope: {hostname}")
    else:
        if not _is_global_ip(hostname):
            raise NetworkEgressScopeError(f"hostname resolves to a non-global address: {hostname}")


def _normalize_port(raw_port: int | str | None, *, scheme: NetworkEgressScheme) -> int:
    if raw_port is None or raw_port == "":
        return _default_port_for_scheme(scheme)
    port = int(raw_port)
    if port < 1 or port > 65535:
        raise NetworkEgressScopeError("port must be between 1 and 65535")
    return port


def parse_network_egress_host(value: object) -> NetworkEgressHost:
    """Parse and canonicalize one typed host scope entry."""
    if isinstance(value, NetworkEgressHost):
        return value

    if isinstance(value, Mapping):
        scheme_raw = str(value.get("scheme") or "https").strip().lower()
        hostname_raw = str(value.get("hostname") or value.get("host") or "").strip()
        port_raw = value.get("port")
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise NetworkEgressScopeError("host scope must not be empty")
        if "://" not in raw:
            raw = f"https://{raw}"
        parsed = urlsplit(raw)
        scheme_raw = (parsed.scheme or "https").lower()
        hostname_raw = parsed.hostname or ""
        port_raw = parsed.port
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise NetworkEgressScopeError("host scope must not include path, query, or fragment")
        if parsed.username or parsed.password:
            raise NetworkEgressScopeError("host scope must not embed credentials")
    else:
        raise NetworkEgressScopeError(f"unsupported host scope type: {type(value)!r}")

    if scheme_raw not in ("http", "https"):
        raise NetworkEgressScopeError(f"unsupported scheme: {scheme_raw}")
    scheme: NetworkEgressScheme = scheme_raw  # type: ignore[assignment]
    hostname = _normalize_hostname(hostname_raw)
    _reject_disallowed_hostname(hostname)
    port = _normalize_port(port_raw, scheme=scheme)
    return NetworkEgressHost(scheme=scheme, hostname=hostname, port=port)


def canonicalize_network_egress_allowlist(
    values: Sequence[object] | None,
) -> NetworkEgressAllowlist:
    """Canonicalize, deduplicate, and stably order host scopes."""
    if values is None:
        return NetworkEgressAllowlist(hosts=())
    seen: dict[str, NetworkEgressHost] = {}
    for item in values:
        host = parse_network_egress_host(item)
        seen[host.canonical_form()] = host
    ordered = tuple(sorted(seen.values(), key=lambda item: item.canonical_form()))
    return NetworkEgressAllowlist(hosts=ordered)


def allowlist_enforcement_satisfies_request(
    requested: NetworkEgressAllowlist,
    enforced: NetworkEgressAllowlist,
) -> bool:
    """True when substrate enforcement is at least as restrictive as requested (V1)."""
    if not requested.hosts or not enforced.hosts:
        return False
    requested_keys = {host.canonical_form() for host in requested.hosts}
    enforced_keys = {host.canonical_form() for host in enforced.hosts}
    if not enforced_keys.issubset(requested_keys):
        return False
    return True
