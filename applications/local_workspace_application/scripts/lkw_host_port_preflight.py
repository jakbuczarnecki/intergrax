# © Artur Czarnecki. All rights reserved.

"""Shared LKW host-port preflight helpers for product and proof runners."""

from __future__ import annotations

import errno
import json
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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


class PortOwnershipKind(str, Enum):
    PRODUCT_STACK = "product_stack"
    KNOWN_INTERGRAX_STACK = "known_intergrax_stack"
    FOREIGN_PROCESS = "foreign_process"
    FREE = "free"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PortOwnership:
    port: int
    kind: PortOwnershipKind
    stack_id: str | None = None
    owner_label: str | None = None


@dataclass(frozen=True)
class KnownIntergraxStackDefinition:
    stack_id: str
    compose_project: str
    compose_file_paths: tuple[str, ...]
    display_label: str
    is_product_stack: bool = False


_PRODUCT_STACK_ID = "lkw-product-quickstart"
_CORE_PLATFORM_PROOF_STACK_ID = "lkw-core-platform-proof"
_TRUSTED_ASK_PROOF_STACK_ID = "lkw-trusted-ask-workspace-proof"

_DESTRUCTIVE_LIFECYCLE_MARKERS = frozenset(
    {
        "-v",
        "--volumes",
        "volume",
        "prune",
    }
)


def known_intergrax_stack_definitions(docker_dir: Path) -> tuple[KnownIntergraxStackDefinition, ...]:
    return (
        KnownIntergraxStackDefinition(
            stack_id=_PRODUCT_STACK_ID,
            compose_project="intergrax_lkw",
            compose_file_paths=(str(docker_dir / "docker-compose.yml"),),
            display_label="LKW Product Quick Start",
            is_product_stack=True,
        ),
        KnownIntergraxStackDefinition(
            stack_id=_CORE_PLATFORM_PROOF_STACK_ID,
            compose_project="lkw-core-platform-proof",
            compose_file_paths=tuple(
                str(docker_dir / name)
                for name in (
                    "docker-compose.yml",
                    "docker-compose.elasticsearch.yml",
                    "docker-compose.kafka.yml",
                    "docker-compose.mongodb.yml",
                    "docker-compose.sentry.yml",
                )
            ),
            display_label="LKW Core Platform Proof",
        ),
        KnownIntergraxStackDefinition(
            stack_id=_TRUSTED_ASK_PROOF_STACK_ID,
            compose_project="lkw-trusted-ask-workspace-proof",
            compose_file_paths=tuple(
                str(docker_dir / name)
                for name in (
                    "docker-compose.yml",
                    "docker-compose.mongodb.yml",
                    "docker-compose.trusted-ask-proof.yml",
                )
            ),
            display_label="LKW Trusted Ask Workspace Proof",
        ),
        KnownIntergraxStackDefinition(
            stack_id="lkw-os-interaction-proof",
            compose_project="lkw-os-interaction-proof",
            compose_file_paths=tuple(
                str(docker_dir / name)
                for name in (
                    "docker-compose.yml",
                    "docker-compose.mongodb.yml",
                )
            ),
            display_label="LKW OS Interaction Proof",
        ),
        KnownIntergraxStackDefinition(
            stack_id="lkw-background-task-proof",
            compose_project="lkw-background-task-proof",
            compose_file_paths=tuple(
                str(docker_dir / name)
                for name in (
                    "docker-compose.yml",
                    "docker-compose.kafka.yml",
                    "docker-compose.mongodb.yml",
                )
            ),
            display_label="LKW Background Task Proof",
        ),
        KnownIntergraxStackDefinition(
            stack_id="lkw-hosting-proof",
            compose_project="lkw-hosting-proof",
            compose_file_paths=tuple(
                str(docker_dir / name)
                for name in (
                    "docker-compose.yml",
                    "docker-compose.mongodb.yml",
                )
            ),
            display_label="LKW Application Hosting Proof",
        ),
        KnownIntergraxStackDefinition(
            stack_id="lkw-file-watcher-e2e-proof",
            compose_project="lkw-file-watcher-e2e-proof",
            compose_file_paths=tuple(
                str(docker_dir / name)
                for name in (
                    "docker-compose.yml",
                    "docker-compose.kafka.yml",
                    "file-watcher-e2e.compose.yml",
                    "docker-compose.mongodb.yml",
                )
            ),
            display_label="LKW File Watcher E2E Proof",
        ),
    )


def compose_exec_args_for_stack(
    stack: KnownIntergraxStackDefinition,
    *compose_command: str,
) -> list[str]:
    args = ["docker", "compose", "-p", stack.compose_project]
    for path in stack.compose_file_paths:
        args.extend(["-f", path])
    args.extend(compose_command)
    return args


def non_destructive_compose_down_args(
    stack: KnownIntergraxStackDefinition,
) -> list[str]:
    return [*compose_exec_args_for_stack(stack, "down"), "--remove-orphans"]


def lifecycle_command_is_non_destructive(command: Sequence[str]) -> bool:
    lowered = [part.lower() for part in command]
    for index, part in enumerate(lowered):
        if part in _DESTRUCTIVE_LIFECYCLE_MARKERS:
            return False
        if part == "volume" and index + 1 < len(lowered) and lowered[index + 1] == "rm":
            return False
        if part == "system" and index + 1 < len(lowered) and lowered[index + 1] == "prune":
            return False
    return True


def collect_stack_owned_ports(
    stack: KnownIntergraxStackDefinition,
    *,
    run_command: Callable[..., Any],
    cwd: Any,
    timeout: int = 30,
) -> frozenset[int] | None:
    return canonical_compose_owned_host_ports(
        compose_exec_args=lambda *command: compose_exec_args_for_stack(stack, *command),
        run_command=run_command,
        cwd=cwd,
        timeout=timeout,
    )


def collect_known_stack_owned_ports_map(
    stacks: Sequence[KnownIntergraxStackDefinition],
    *,
    run_command: Callable[..., Any],
    cwd: Any,
    timeout: int = 30,
) -> dict[str, frozenset[int] | None]:
    owned: dict[str, frozenset[int] | None] = {}
    for stack in stacks:
        owned[stack.stack_id] = collect_stack_owned_ports(
            stack,
            run_command=run_command,
            cwd=cwd,
            timeout=timeout,
        )
    return owned


def classify_port_ownership(
    port: int,
    stack_owned_ports: Mapping[str, frozenset[int] | None],
    *,
    product_stack_id: str = _PRODUCT_STACK_ID,
    stack_labels: Mapping[str, str] | None = None,
) -> PortOwnership:
    labels = stack_labels or {}
    product_owned = stack_owned_ports.get(product_stack_id)
    if product_owned is not None and port in product_owned:
        return PortOwnership(
            port=port,
            kind=PortOwnershipKind.PRODUCT_STACK,
            stack_id=product_stack_id,
            owner_label=labels.get(product_stack_id),
        )

    for stack_id, owned in stack_owned_ports.items():
        if stack_id == product_stack_id:
            continue
        if owned is not None and port in owned:
            return PortOwnership(
                port=port,
                kind=PortOwnershipKind.KNOWN_INTERGRAX_STACK,
                stack_id=stack_id,
                owner_label=labels.get(stack_id),
            )

    if probe_host_port_available(port) and not is_loopback_tcp_port_reachable(port):
        return PortOwnership(port=port, kind=PortOwnershipKind.FREE)

    ownership_unresolved = all(value is None for value in stack_owned_ports.values())
    if ownership_unresolved and is_loopback_tcp_port_reachable(port):
        return PortOwnership(port=port, kind=PortOwnershipKind.UNKNOWN)

    if is_loopback_tcp_port_reachable(port):
        return PortOwnership(port=port, kind=PortOwnershipKind.FOREIGN_PROCESS)

    if not probe_host_port_available(port):
        return PortOwnership(port=port, kind=PortOwnershipKind.FOREIGN_PROCESS)

    return PortOwnership(port=port, kind=PortOwnershipKind.UNKNOWN)


def find_known_stack(
    stacks: Sequence[KnownIntergraxStackDefinition],
    stack_id: str,
) -> KnownIntergraxStackDefinition | None:
    for stack in stacks:
        if stack.stack_id == stack_id:
            return stack
    return None
