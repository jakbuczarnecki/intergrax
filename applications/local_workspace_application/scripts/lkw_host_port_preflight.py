# © Artur Czarnecki. All rights reserved.

"""Shared LKW host-port preflight helpers for product and proof runners."""

from __future__ import annotations

import errno
import json
import socket
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_IPV6_UNSUPPORTED_ERRNOS = frozenset(
    error
    for error in (
        getattr(errno, "EAFNOSUPPORT", None),
        getattr(errno, "EPROTONOSUPPORT", None),
        getattr(errno, "ENOPROTOOPT", None),
        getattr(errno, "EADDRNOTAVAIL", None),
        10043,  # WSAEPROTONOSUPPORT
        10047,  # WSAEAFNOSUPPORT
        10049,  # WSAEADDRNOTAVAIL
    )
    if error is not None
)


def is_unsupported_ipv6_error(error: OSError) -> bool:
    return error.errno in _IPV6_UNSUPPORTED_ERRNOS


def is_loopback_tcp_port_reachable(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        return probe.connect_ex(("127.0.0.1", port)) == 0
    finally:
        probe.close()


def probe_host_port_available(port: int) -> bool:
    probes: list[tuple[int, tuple[object, ...]]] = [
        (socket.AF_INET, ("0.0.0.0", port)),
    ]
    ipv6_family = getattr(socket, "AF_INET6", None)
    if ipv6_family is not None:
        probes.append((ipv6_family, ("::", port, 0, 0)))

    for family, address in probes:
        try:
            probe = socket.socket(family, socket.SOCK_STREAM)
        except OSError as error:
            if family == ipv6_family and is_unsupported_ipv6_error(error):
                continue
            return False
        try:
            exclusive_address_use = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive_address_use is not None and hasattr(probe, "setsockopt"):
                try:
                    probe.setsockopt(socket.SOL_SOCKET, exclusive_address_use, 1)
                except OSError:
                    pass
            probe.bind(address)
        except OSError as error:
            if family == ipv6_family and is_unsupported_ipv6_error(error):
                continue
            return False
        finally:
            probe.close()
    return True


def parse_compose_ps_services(stdout: str) -> list[dict[str, Any]] | None:
    text = stdout.strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return None
        services: list[dict[str, Any]] = []
        for item in parsed:
            if isinstance(item, dict):
                services.append(item)
        return services
    services = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            services.append(item)
    return services


def host_ports_from_compose_ports_field(ports_field: str) -> set[int]:
    ports: set[int] = set()
    for fragment in str(ports_field).split(","):
        fragment = fragment.strip()
        if "->" not in fragment:
            continue
        host_mapping = fragment.split("->", 1)[0]
        host_port_text = host_mapping.rsplit(":", 1)[-1].strip()
        if not host_port_text.isdecimal():
            continue
        port = int(host_port_text)
        if 1 <= port <= 65535:
            ports.add(port)
    return ports


def host_ports_from_compose_publishers(service: Mapping[str, Any]) -> set[int]:
    ports: set[int] = set()
    publishers = service.get("Publishers", service.get("publishers"))
    if not isinstance(publishers, list):
        return ports
    for entry in publishers:
        if not isinstance(entry, dict):
            continue
        published = entry.get("PublishedPort", entry.get("publishedPort"))
        if isinstance(published, bool):
            continue
        try:
            port = int(published)
        except (TypeError, ValueError, OverflowError):
            continue
        if 1 <= port <= 65535:
            ports.add(port)
    return ports


def host_ports_from_compose_service(service: Mapping[str, Any]) -> set[int]:
    ports = host_ports_from_compose_publishers(service)
    ports_field = service.get("Ports", service.get("ports"))
    if isinstance(ports_field, str):
        ports.update(host_ports_from_compose_ports_field(ports_field))
    return ports


def canonical_compose_owned_host_ports(
    *,
    compose_exec_args: Callable[..., Sequence[str]],
    run_command: Callable[..., Any],
    cwd: Any,
    timeout: int = 30,
) -> frozenset[int] | None:
    completed = run_command(
        compose_exec_args("ps", "-a", "--format", "json"),
        cwd=cwd,
        timeout=timeout,
    )
    if completed.returncode != 0:
        return None
    services = parse_compose_ps_services(completed.stdout)
    if services is None:
        return None
    owned: set[int] = set()
    for service in services:
        owned.update(host_ports_from_compose_service(service))
    return frozenset(owned)


def _host_port_from_compose_port_entry(entry: object) -> int | None:
    if isinstance(entry, str):
        host_mapping = entry.split("/", 1)[0]
        host_port_text = host_mapping.rsplit(":", 1)[0].strip()
        if host_port_text.isdecimal():
            port = int(host_port_text)
            return port if 1 <= port <= 65535 else None
        return None
    if not isinstance(entry, dict):
        return None
    published = entry.get("published")
    if published is None:
        published = entry.get("Published")
    if isinstance(published, str) and published.isdecimal():
        port = int(published)
        return port if 1 <= port <= 65535 else None
    try:
        port = int(published)
    except (TypeError, ValueError, OverflowError):
        return None
    return port if 1 <= port <= 65535 else None


def published_host_ports_from_compose_config(config: Mapping[str, Any]) -> frozenset[int]:
    services = config.get("services")
    if not isinstance(services, dict):
        return frozenset()
    ports: set[int] = set()
    for service in services.values():
        if not isinstance(service, dict):
            continue
        for key in ("ports", "publish"):
            entries = service.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                parsed = _host_port_from_compose_port_entry(entry)
                if parsed is not None:
                    ports.add(parsed)
    return frozenset(ports)


def resolve_compose_published_host_ports(
    *,
    compose_exec_args: Callable[..., Sequence[str]],
    run_command: Callable[..., Any],
    cwd: Any,
    timeout: int = 120,
) -> frozenset[int]:
    completed = run_command(
        [*compose_exec_args("config", "--format", "json")],
        cwd=cwd,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError("compose_config_failed")
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed, dict):
        raise RuntimeError("compose_config_not_object")
    return published_host_ports_from_compose_config(parsed)
