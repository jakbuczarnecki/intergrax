# © Artur Czarnecki. All rights reserved.

"""Kernel egress policy application and independent verification."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from intergrax.contracts.sandbox_network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
    NetworkEgressScopeError,
)
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist

from .constants import (
    ALLOWED_ADDR,
    ALLOWED_HOSTNAME,
    ALLOWED_PORT,
    DENIED_ADDR,
    DENIED_HOSTNAME,
    DENIED_PORT,
)

_QUAL_HOST_TO_ENDPOINT: dict[str, tuple[str, int]] = {
    ALLOWED_HOSTNAME: (ALLOWED_ADDR, ALLOWED_PORT),
    DENIED_HOSTNAME: (DENIED_ADDR, DENIED_PORT),
}


@dataclass(frozen=True, slots=True)
class VerifiedEgressPolicy:
    """Kernel-visible egress enforcement derived from nftables/iptables state."""

    enforced_hosts: NetworkEgressAllowlist
    accepted_destinations: frozenset[tuple[str, int]]


class ReferenceSubstratePolicyError(RuntimeError):
    """Reference substrate could not apply or verify kernel egress policy."""


def resolve_qualification_endpoint(host: NetworkEgressHost) -> tuple[str, int]:
    mapped = _QUAL_HOST_TO_ENDPOINT.get(host.hostname)
    if mapped is None:
        raise ReferenceSubstratePolicyError(
            f"hostname {host.hostname!r} is outside reference qualification topology",
        )
    return mapped


def allowlist_to_endpoints(allowlist: NetworkEgressAllowlist) -> list[tuple[str, int]]:
    if not allowlist.hosts:
        raise ReferenceSubstratePolicyError("empty allowlist cannot map to unrestricted egress")
    endpoints: list[tuple[str, int]] = []
    for host in allowlist.hosts:
        addr, default_port = resolve_qualification_endpoint(host)
        port = host.port if host.port else default_port
        endpoints.append((addr, port))
    return endpoints


def apply_egress_policy_netns(
    netns: str,
    *,
    allowlist: NetworkEgressAllowlist,
    use_nftables: bool,
) -> None:
    endpoints = allowlist_to_endpoints(allowlist)
    if use_nftables:
        _apply_nftables(netns, endpoints)
    else:
        _apply_iptables(netns, endpoints)
    verified = read_verified_egress_policy(netns, use_nftables=use_nftables)
    requested = canonicalize_network_egress_allowlist(
        tuple(host.canonical_form() for host in allowlist.hosts),
    )
    if verified.enforced_hosts.fingerprint() != requested.fingerprint():
        raise ReferenceSubstratePolicyError(
            "kernel policy verification failed after apply — scope mismatch",
        )


def read_verified_egress_policy(netns: str, *, use_nftables: bool) -> VerifiedEgressPolicy:
    if use_nftables:
        raw = _run_in_netns(netns, ["nft", "-n", "list", "ruleset"])
        destinations = _parse_nft_accept_destinations(raw)
    else:
        raw = _run_in_netns(netns, ["iptables-save"])
        destinations = _parse_iptables_accept_destinations(raw)
    if not destinations:
        raise ReferenceSubstratePolicyError(
            "no enforced egress destinations found in kernel state",
        )
    canonical_hosts: list[str] = []
    for addr, port in sorted(destinations):
        host = _endpoint_to_canonical_host(addr, port)
        if host is None:
            raise ReferenceSubstratePolicyError(
                f"kernel policy contains non-qualification destination {addr}:{port}",
            )
        canonical_hosts.append(host)
    enforced = canonicalize_network_egress_allowlist(tuple(dict.fromkeys(canonical_hosts)))
    return VerifiedEgressPolicy(
        enforced_hosts=enforced,
        accepted_destinations=frozenset(destinations),
    )


def _endpoint_to_canonical_host(addr: str, port: int) -> str | None:
    for hostname, (mapped_addr, mapped_port) in _QUAL_HOST_TO_ENDPOINT.items():
        if addr == mapped_addr and port == mapped_port:
            return f"http://{hostname}:{mapped_port}"
    return None


def _run_in_netns(netns: str, command: list[str], *, timeout: float = 10.0) -> str:
    label = " ".join(command)
    try:
        completed = subprocess.run(
            ["ip", "netns", "exec", netns, *command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReferenceSubstratePolicyError(
            f"command timed out in netns {netns}: {label} (classification=timeout)",
        ) from exc
    except OSError as exc:
        raise ReferenceSubstratePolicyError(
            f"os error in netns {netns}: {label} (classification=os_error; {exc})",
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ReferenceSubstratePolicyError(
            f"command failed in netns {netns}: {label} (classification=nonzero_exit; {stderr})",
        )
    return completed.stdout


def _apply_nftables(netns: str, endpoints: list[tuple[str, int]]) -> None:
    _run_in_netns(netns, ["nft", "add", "table", "inet", "igxqual"])
    _run_in_netns(
        netns,
        [
            "nft",
            "add",
            "chain",
            "inet",
            "igxqual",
            "output",
            "{",
            "type",
            "filter",
            "hook",
            "output",
            "priority",
            "0",
            ";",
            "policy",
            "drop",
            ";",
            "}",
        ],
    )
    _run_in_netns(
        netns,
        ["nft", "add", "rule", "inet", "igxqual", "output", "ip", "daddr", "127.0.0.0/8", "accept"],
    )
    for addr, port in endpoints:
        _run_in_netns(
            netns,
            [
                "nft",
                "add",
                "rule",
                "inet",
                "igxqual",
                "output",
                "ip",
                "daddr",
                addr,
                "tcp",
                "dport",
                str(port),
                "accept",
            ],
        )


def _apply_iptables(netns: str, endpoints: list[tuple[str, int]]) -> None:
    _run_in_netns(netns, ["iptables", "-P", "OUTPUT", "DROP"])
    _run_in_netns(netns, ["iptables", "-A", "OUTPUT", "-d", "127.0.0.0/8", "-j", "ACCEPT"])
    for addr, port in endpoints:
        _run_in_netns(
            netns,
            [
                "iptables",
                "-A",
                "OUTPUT",
                "-p",
                "tcp",
                "-d",
                addr,
                "--dport",
                str(port),
                "-j",
                "ACCEPT",
            ],
        )


_NFT_DADDR_RE = re.compile(
    r"ip daddr (\d+\.\d+\.\d+\.\d+)(?: tcp dport (\d+))?",
)


def _parse_nft_accept_destinations(raw: str) -> set[tuple[str, int]]:
    destinations: set[tuple[str, int]] = set()
    for line in raw.splitlines():
        if "accept" not in line or "daddr" not in line:
            continue
        match = _NFT_DADDR_RE.search(line)
        if match is None:
            if "daddr" in line:
                raise ReferenceSubstratePolicyError(
                    f"unparseable nftables accept rule line: {line!r}",
                )
            continue
        addr = match.group(1)
        if addr == "127.0.0.0/8" or addr.startswith("127."):
            continue
        port_text = match.group(2)
        if not port_text:
            if "tcp dport" in line:
                raise ReferenceSubstratePolicyError(
                    f"malformed nftables destination port in line: {line!r}",
                )
            continue
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ReferenceSubstratePolicyError(
                f"malformed nftables destination port in line: {line!r}",
            ) from exc
        if port == 0:
            continue
        destinations.add((addr, port))
    return destinations


def _parse_iptables_accept_destinations(raw: str) -> set[tuple[str, int]]:
    destinations: set[tuple[str, int]] = set()
    for line in raw.splitlines():
        if not line.startswith("-A OUTPUT"):
            continue
        if "ACCEPT" not in line or "-d" not in line:
            continue
        parts = line.split()
        try:
            d_index = parts.index("-d")
            addr = parts[d_index + 1]
        except (ValueError, IndexError):
            continue
        if addr.startswith("127."):
            continue
        port = 0
        if "-p" in parts and "tcp" in parts and "--dport" in parts:
            port_index = parts.index("--dport")
            try:
                port = int(parts[port_index + 1])
            except (ValueError, IndexError) as exc:
                raise ReferenceSubstratePolicyError(
                    f"malformed iptables destination port in line: {line!r}",
                ) from exc
        if port == 0:
            continue
        destinations.add((addr, port))
    return destinations


def attestation_from_verified_policy(
    verified: VerifiedEgressPolicy,
    *,
    provider_id: str,
) -> tuple[bool, tuple[str, ...]]:
    hosts = tuple(host.canonical_form() for host in verified.enforced_hosts.hosts)
    return True, hosts


def reject_forged_request_echo(
    requested: NetworkEgressAllowlist,
    *,
    forged_hosts: tuple[str, ...],
) -> None:
    """Adversarial guard — requested scope copied without kernel verification must fail."""
    forged = canonicalize_network_egress_allowlist(forged_hosts)
    if forged.fingerprint() == requested.fingerprint():
        raise NetworkEgressScopeError("forged attestation must not mirror request without kernel proof")
