# © Artur Czarnecki. All rights reserved.

"""Linux network namespace topology for reference substrate sessions."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

from .constants import (
    ALLOWED_ADDR,
    ALLOWED_HOSTNAME,
    DENIED_ADDR,
    DENIED_HOSTNAME,
    SANDBOX_ADDR,
    VETH_HOST_ADDR,
)
from .firewall import ReferenceSubstratePolicyError


class ReferenceSubstrateTopologyError(RuntimeError):
    """Network namespace topology could not be materialized."""


class _ResidualPresence(Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class NetnsSessionResources:
    netns: str
    veth_host: str
    veth_peer: str


@dataclass(slots=True)
class _PartialTopologyCreation:
    netns: str
    veth_host: str
    veth_peer: str
    mutation_started: bool = False


def _command_label(command: list[str]) -> str:
    return " ".join(command)


def _run(command: list[str], *, timeout: float = 10.0) -> None:
    label = _command_label(command)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReferenceSubstrateTopologyError(
            f"command timed out: {label} (classification=timeout)",
        ) from exc
    except OSError as exc:
        raise ReferenceSubstrateTopologyError(
            f"os error running command: {label} (classification=os_error; {exc})",
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ReferenceSubstrateTopologyError(
            f"command failed: {label} (classification=nonzero_exit; {stderr})",
        )


def _try_run(command: list[str], *, timeout: float = 10.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"timeout expired: {_command_label(command)}"
    except OSError as exc:
        return f"os error running {_command_label(command)}: {exc}"
    if completed.returncode != 0:
        return completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
    return None


def _inspect_netns(netns: str) -> _ResidualPresence:
    try:
        completed = subprocess.run(
            ["ip", "netns", "list"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _ResidualPresence.UNKNOWN
    except OSError:
        return _ResidualPresence.UNKNOWN
    if completed.returncode != 0:
        return _ResidualPresence.UNKNOWN
    for line in completed.stdout.splitlines():
        name = line.split()[0] if line.split() else ""
        if name == netns:
            return _ResidualPresence.PRESENT
    return _ResidualPresence.ABSENT


def _inspect_link(link_name: str) -> _ResidualPresence:
    try:
        completed = subprocess.run(
            ["ip", "link", "show", link_name],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _ResidualPresence.UNKNOWN
    except OSError:
        return _ResidualPresence.UNKNOWN
    if completed.returncode == 0:
        return _ResidualPresence.PRESENT
    if completed.returncode == 1:
        return _ResidualPresence.ABSENT
    return _ResidualPresence.UNKNOWN


def _inspect_netns_hosts_material(netns: str) -> _ResidualPresence:
    hosts_path = Path("/etc/netns") / netns
    try:
        return _ResidualPresence.PRESENT if hosts_path.exists() else _ResidualPresence.ABSENT
    except OSError:
        return _ResidualPresence.UNKNOWN


def _remove_netns_hosts_material(netns: str) -> str | None:
    hosts_path = Path("/etc/netns") / netns
    if not hosts_path.exists():
        return None
    try:
        for child in hosts_path.iterdir():
            child.unlink(missing_ok=True)
        hosts_path.rmdir()
    except OSError as exc:
        return f"/etc/netns/{netns} cleanup: {exc}"
    if _inspect_netns_hosts_material(netns) == _ResidualPresence.PRESENT:
        return f"/etc/netns/{netns} still present after cleanup attempt"
    return None


def _rollback_allocated_topology(partial: _PartialTopologyCreation) -> list[str]:
    """Attempt idempotent cleanup for every resource that could exist after mutation began."""
    if not partial.mutation_started:
        return []
    rollback_errors: list[str] = []
    hosts_err = _remove_netns_hosts_material(partial.netns)
    if hosts_err is not None:
        rollback_errors.append(hosts_err)
    netns_err = _try_run(["ip", "netns", "delete", partial.netns])
    if netns_err is not None:
        rollback_errors.append(f"netns delete: {netns_err}")
    veth_err = _try_run(["ip", "link", "delete", partial.veth_host])
    if veth_err is not None:
        rollback_errors.append(f"veth delete: {veth_err}")
    return rollback_errors


def _write_netns_hosts(netns: str) -> None:
    hosts_dir = Path("/etc/netns") / netns
    try:
        hosts_dir.mkdir(parents=True, exist_ok=True)
        hosts_path = hosts_dir / "hosts"
        hosts_path.write_text(
            "\n".join(
                [
                    f"{ALLOWED_ADDR} {ALLOWED_HOSTNAME}",
                    f"{DENIED_ADDR} {DENIED_HOSTNAME}",
                    "",
                ],
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ReferenceSubstrateTopologyError(
            f"cannot materialize /etc/netns/{netns}/hosts "
            f"(classification=filesystem_os_error; {exc})",
        ) from exc


def create_netns_session() -> NetnsSessionResources:
    suffix = uuid4().hex[:10]
    netns = f"igx-qual-{suffix}"
    veth_host = f"veth{suffix[:6]}h"
    veth_peer = f"veth{suffix[:6]}p"
    partial = _PartialTopologyCreation(netns=netns, veth_host=veth_host, veth_peer=veth_peer)
    try:
        partial.mutation_started = True
        _run(["ip", "netns", "add", netns])
        _run(["ip", "link", "add", veth_host, "type", "veth", "peer", "name", veth_peer])
        _run(["ip", "link", "set", veth_peer, "netns", netns])
        _run(["ip", "addr", "add", f"{VETH_HOST_ADDR}/24", "dev", veth_host])
        _run(["ip", "link", "set", veth_host, "up"])
        _run(["ip", "netns", "exec", netns, "ip", "addr", "add", f"{SANDBOX_ADDR}/24", "dev", veth_peer])
        _run(["ip", "netns", "exec", netns, "ip", "link", "set", veth_peer, "up"])
        _run(["ip", "netns", "exec", netns, "ip", "link", "set", "lo", "up"])
        _run(["ip", "netns", "exec", netns, "ip", "route", "add", "default", "via", VETH_HOST_ADDR])
        _write_netns_hosts(netns)
    except ReferenceSubstrateTopologyError as exc:
        rollback_errors = _rollback_allocated_topology(partial)
        message = str(exc)
        if rollback_errors:
            message += f"; rollback incomplete: {'; '.join(rollback_errors)}"
        raise ReferenceSubstrateTopologyError(message) from exc
    return NetnsSessionResources(netns=netns, veth_host=veth_host, veth_peer=veth_peer)


def destroy_netns_session(resources: NetnsSessionResources) -> None:
    attempt_errors: list[str] = []
    netns_err = _try_run(["ip", "netns", "delete", resources.netns])
    if netns_err is not None:
        attempt_errors.append(f"netns delete: {netns_err}")
    veth_err = _try_run(["ip", "link", "delete", resources.veth_host])
    if veth_err is not None:
        attempt_errors.append(f"veth delete: {veth_err}")
    hosts_err = _remove_netns_hosts_material(resources.netns)
    if hosts_err is not None:
        attempt_errors.append(hosts_err)

    residual_errors: list[str] = []
    netns_state = _inspect_netns(resources.netns)
    veth_state = _inspect_link(resources.veth_host)
    hosts_state = _inspect_netns_hosts_material(resources.netns)
    for label, state in (
        ("netns", netns_state),
        ("host veth", veth_state),
        ("/etc/netns material", hosts_state),
    ):
        if state == _ResidualPresence.UNKNOWN:
            residual_errors.append(f"cannot verify {label} absence for {resources.netns}")
        elif state == _ResidualPresence.PRESENT:
            residual_errors.append(f"{label} still present for {resources.netns}")

    if residual_errors:
        detail_parts = residual_errors
        if attempt_errors:
            detail_parts = attempt_errors + residual_errors
        raise ReferenceSubstratePolicyError(
            f"cleanup incomplete for {resources.netns}: {'; '.join(detail_parts)}",
        )
