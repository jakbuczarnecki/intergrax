# © Artur Czarnecki. All rights reserved.

"""Linux network namespace topology for reference substrate sessions."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
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
    netns_added: bool = False
    veth_added: bool = False
    hosts_material: bool = False


def _run(command: list[str], *, timeout: float = 10.0) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ReferenceSubstrateTopologyError(
            f"command failed: {' '.join(command)} ({stderr})",
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
        return f"timeout expired: {' '.join(command)}"
    except OSError as exc:
        return f"os error running {' '.join(command)}: {exc}"
    if completed.returncode != 0:
        return completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
    return None


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
    return None


def _rollback_partial_topology(partial: _PartialTopologyCreation) -> list[str]:
    rollback_errors: list[str] = []
    if partial.hosts_material:
        hosts_err = _remove_netns_hosts_material(partial.netns)
        if hosts_err is not None:
            rollback_errors.append(hosts_err)
    if partial.netns_added:
        netns_err = _try_run(["ip", "netns", "delete", partial.netns])
        if netns_err is not None:
            rollback_errors.append(f"netns delete: {netns_err}")
    if partial.veth_added:
        veth_err = _try_run(["ip", "link", "delete", partial.veth_host])
        if veth_err is not None:
            rollback_errors.append(f"veth delete: {veth_err}")
    return rollback_errors


def _write_netns_hosts(netns: str) -> None:
    hosts_dir = Path("/etc/netns") / netns
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


def create_netns_session() -> NetnsSessionResources:
    suffix = uuid4().hex[:10]
    netns = f"igx-qual-{suffix}"
    veth_host = f"veth{suffix[:6]}h"
    veth_peer = f"veth{suffix[:6]}p"
    partial = _PartialTopologyCreation(netns=netns, veth_host=veth_host, veth_peer=veth_peer)
    try:
        _run(["ip", "netns", "add", netns])
        partial.netns_added = True
        _run(["ip", "link", "add", veth_host, "type", "veth", "peer", "name", veth_peer])
        partial.veth_added = True
        _run(["ip", "link", "set", veth_peer, "netns", netns])
        _run(["ip", "addr", "add", f"{VETH_HOST_ADDR}/24", "dev", veth_host])
        _run(["ip", "link", "set", veth_host, "up"])
        _run(["ip", "netns", "exec", netns, "ip", "addr", "add", f"{SANDBOX_ADDR}/24", "dev", veth_peer])
        _run(["ip", "netns", "exec", netns, "ip", "link", "set", veth_peer, "up"])
        _run(["ip", "netns", "exec", netns, "ip", "link", "set", "lo", "up"])
        _run(["ip", "netns", "exec", netns, "ip", "route", "add", "default", "via", VETH_HOST_ADDR])
        partial.hosts_material = True
        _write_netns_hosts(netns)
    except ReferenceSubstrateTopologyError as exc:
        rollback_errors = _rollback_partial_topology(partial)
        message = str(exc)
        if rollback_errors:
            message += f"; rollback incomplete: {'; '.join(rollback_errors)}"
        raise ReferenceSubstrateTopologyError(message) from exc
    return NetnsSessionResources(netns=netns, veth_host=veth_host, veth_peer=veth_peer)


def _netns_exists(netns: str) -> bool:
    completed = subprocess.run(
        ["ip", "netns", "list"],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    return netns in completed.stdout


def _link_exists(link_name: str) -> bool:
    completed = subprocess.run(
        ["ip", "link", "show", link_name],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    return completed.returncode == 0


def _topology_material_present(resources: NetnsSessionResources) -> bool:
    if _netns_exists(resources.netns):
        return True
    if _link_exists(resources.veth_host):
        return True
    hosts_path = Path("/etc/netns") / resources.netns
    return hosts_path.exists()


def destroy_netns_session(resources: NetnsSessionResources) -> None:
    errors: list[str] = []
    netns_err = _try_run(["ip", "netns", "delete", resources.netns])
    if netns_err is not None:
        errors.append(f"netns delete: {netns_err}")
    veth_err = _try_run(["ip", "link", "delete", resources.veth_host])
    if veth_err is not None:
        errors.append(f"veth delete: {veth_err}")
    hosts_err = _remove_netns_hosts_material(resources.netns)
    if hosts_err is not None:
        errors.append(hosts_err)
    if not errors:
        return
    if not _topology_material_present(resources):
        return
    raise ReferenceSubstratePolicyError(
        f"cleanup incomplete for {resources.netns}: {'; '.join(errors)}",
    )
