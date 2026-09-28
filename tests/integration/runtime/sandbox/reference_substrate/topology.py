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


def create_netns_session() -> NetnsSessionResources:
    suffix = uuid4().hex[:10]
    netns = f"igx-qual-{suffix}"
    veth_host = f"veth{suffix[:6]}h"
    veth_peer = f"veth{suffix[:6]}p"
    _run(["ip", "netns", "add", netns])
    _run(["ip", "link", "add", veth_host, "type", "veth", "peer", "name", veth_peer])
    _run(["ip", "link", "set", veth_peer, "netns", netns])
    _run(["ip", "addr", "add", f"{VETH_HOST_ADDR}/24", "dev", veth_host])
    _run(["ip", "link", "set", veth_host, "up"])
    _run(["ip", "netns", "exec", netns, "ip", "addr", "add", f"{SANDBOX_ADDR}/24", "dev", veth_peer])
    _run(["ip", "netns", "exec", netns, "ip", "link", "set", veth_peer, "up"])
    _run(["ip", "netns", "exec", netns, "ip", "link", "set", "lo", "up"])
    _run(["ip", "netns", "exec", netns, "ip", "route", "add", "default", "via", VETH_HOST_ADDR])
    _configure_hosts(netns)
    return NetnsSessionResources(netns=netns, veth_host=veth_host, veth_peer=veth_peer)


def _configure_hosts(netns: str) -> None:
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


def destroy_netns_session(resources: NetnsSessionResources) -> None:
    errors: list[str] = []
    for command in (
        ["ip", "netns", "delete", resources.netns],
        ["ip", "link", "delete", resources.veth_host],
    ):
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        if completed.returncode != 0:
            errors.append(completed.stderr.strip() or completed.stdout.strip())
    hosts_path = Path("/etc/netns") / resources.netns
    if hosts_path.exists():
        for child in hosts_path.iterdir():
            child.unlink(missing_ok=True)
        hosts_path.rmdir()
    if errors and not _netns_exists(resources.netns):
        return
    if errors:
        raise ReferenceSubstratePolicyError(
            f"cleanup incomplete for {resources.netns}: {'; '.join(errors)}",
        )


def _netns_exists(netns: str) -> bool:
    completed = subprocess.run(
        ["ip", "netns", "list"],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    return netns in completed.stdout
