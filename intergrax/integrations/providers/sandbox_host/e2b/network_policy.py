# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Intergrax ↔ E2B exact-host network policy mapping and attestation."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from intergrax.runtime.sandbox.network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
    canonicalize_network_egress_allowlist,
    parse_network_egress_host,
)

from intergrax.integrations.providers.sandbox_host.e2b.errors import E2bSandboxSecurityError

_E2B_DEFAULT_DENY = "0.0.0.0/0"
_E2B_DNS_HELPER_IPS = frozenset({"8.8.8.8", "8.8.4.4"})
_WILDCARD_HOST_RE = re.compile(r"^\*\.")
_CIDR_RE = re.compile(r"/\d{1,3}$")


@dataclass(frozen=True, slots=True)
class E2bNetworkCreatePayload:
    """Provider creation payload for exact-host allowlist enforcement."""

    allow_out: tuple[str, ...]
    deny_out: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class E2bProviderNetworkState:
    """Observed provider network policy after sandbox creation."""

    allow_out: tuple[str, ...]
    deny_out: tuple[str, ...]


def validate_security_allowlist_requirements(
    allowlist: NetworkEgressAllowlist | None,
) -> NetworkEgressAllowlist:
    """Reject unsupported Intergrax host scopes before any provider call."""
    if allowlist is None or not allowlist.hosts:
        raise E2bSandboxSecurityError("network egress allowlist must not be empty")
    for host in allowlist.hosts:
        if host.scheme != "https":
            raise E2bSandboxSecurityError(
                f"E2B adapter V1 supports HTTPS only; unsupported scheme: {host.scheme}",
            )
        if host.port != 443:
            raise E2bSandboxSecurityError(
                "E2B domain filtering is enforced on TLS:443 only; "
                f"unsupported port: {host.port}",
            )
    return allowlist


def intergrax_hosts_to_e2b_allow_out(allowlist: NetworkEgressAllowlist) -> E2bNetworkCreatePayload:
    """Map canonical exact-host allowlist to E2B creation-time network config."""
    validated = validate_security_allowlist_requirements(allowlist)
    hostnames: list[str] = []
    seen: set[str] = set()
    for host in validated.hosts:
        hostname = host.hostname.lower()
        if hostname in seen:
            continue
        seen.add(hostname)
        hostnames.append(hostname)
    if not hostnames:
        raise E2bSandboxSecurityError("network egress allowlist must not be empty")
    return E2bNetworkCreatePayload(
        allow_out=tuple(sorted(hostnames)),
        deny_out=(_E2B_DEFAULT_DENY,),
    )


def _is_bare_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _is_cidr(value: str) -> bool:
    if "/" not in value:
        return False
    try:
        ipaddress.ip_network(value, strict=False)
    except ValueError:
        return False
    return True


def _is_wildcard_domain(value: str) -> bool:
    return value.startswith("*.") or _WILDCARD_HOST_RE.match(value) is not None


def _is_global_deny_entry(value: str) -> bool:
    normalized = value.strip()
    if normalized in {"0.0.0.0/0", "::/0"}:
        return True
    if normalized.endswith("/0"):
        try:
            network = ipaddress.ip_network(normalized, strict=False)
        except ValueError:
            return False
        return network.prefixlen == 0
    return False


def _provider_entry_is_disallowed_authority(entry: str) -> str | None:
    normalized = entry.strip().lower()
    if not normalized:
        return "empty allowOut entry"
    if _is_wildcard_domain(normalized):
        return f"wildcard allowOut entry: {entry}"
    if _is_cidr(normalized):
        return f"CIDR allowOut entry: {entry}"
    if _is_bare_ip(normalized):
        if normalized in _E2B_DNS_HELPER_IPS:
            return None
        return f"bare IP allowOut entry: {entry}"
    if ":" in normalized:
        return f"host:port allowOut entry: {entry}"
    return None


def _domain_entry_to_canonical_host(hostname: str) -> NetworkEgressHost:
    return parse_network_egress_host(f"https://{hostname.lower()}:443")


def provider_state_to_enforced_allowlist(
    provider_state: E2bProviderNetworkState,
    *,
    requested: NetworkEgressAllowlist,
) -> NetworkEgressAllowlist:
    """Convert provider-applied allowOut into canonical enforced host evidence."""
    if not provider_state.allow_out:
        raise E2bSandboxSecurityError("provider network state missing allowOut")

    domain_hosts: list[str] = []
    for entry in provider_state.allow_out:
        rejection = _provider_entry_is_disallowed_authority(entry)
        if rejection is not None:
            raise E2bSandboxSecurityError(rejection)
        domain_hosts.append(entry.strip().lower())

    if not domain_hosts:
        raise E2bSandboxSecurityError("provider allowOut contains no exact domain entries")

    deny_entries = tuple(entry.strip() for entry in provider_state.deny_out if entry.strip())
    if not deny_entries:
        raise E2bSandboxSecurityError("provider network state missing default-deny denyOut")
    if not any(_is_global_deny_entry(entry) for entry in deny_entries):
        raise E2bSandboxSecurityError("provider denyOut does not include global default deny")

    enforced = canonicalize_network_egress_allowlist(
        [f"https://{hostname}" for hostname in domain_hosts],
    )
    requested_keys = {host.canonical_form() for host in requested.hosts}
    enforced_keys = {host.canonical_form() for host in enforced.hosts}
    if enforced_keys != requested_keys:
        raise E2bSandboxSecurityError("provider enforced allowOut does not match requested scope")
    return enforced


def parse_provider_network_state(payload: Mapping[str, object] | None) -> E2bProviderNetworkState:
    """Parse provider sandbox info network block."""
    if payload is None:
        raise E2bSandboxSecurityError("provider sandbox info missing network state")
    allow_raw = payload.get("allowOut") or payload.get("allow_out") or []
    deny_raw = payload.get("denyOut") or payload.get("deny_out") or []
    if not isinstance(allow_raw, Sequence) or not isinstance(deny_raw, Sequence):
        raise E2bSandboxSecurityError("provider network state is malformed")
    allow_out = tuple(str(item).strip() for item in allow_raw if str(item).strip())
    deny_out = tuple(str(item).strip() for item in deny_raw if str(item).strip())
    return E2bProviderNetworkState(allow_out=allow_out, deny_out=deny_out)
