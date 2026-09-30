# © Artur Czarnecki. All rights reserved.

"""Bounded Linux/WSL preflight for reference substrate physical qualification."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReferenceSubstratePreflight:
    """Environment readiness for kernel-level qualification."""

    ok: bool
    block_reason: str | None
    wsl_detected: bool
    linux_kernel: str
    ip_available: bool
    netns_usable: bool
    firewall_backend: str | None
    privileged: bool
    python_available: bool

    def skip_reason(self) -> str:
        if self.block_reason:
            return self.block_reason
        return "reference substrate preflight blocked"


def _detect_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def _run_ip(args: list[str], *, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ip", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _inspect_netns_usable() -> tuple[bool, str | None]:
    try:
        probe = _run_ip(["netns", "list"])
    except subprocess.TimeoutExpired:
        return False, "`ip netns list` timed out during environment inspection"
    except OSError as exc:
        return False, f"`ip netns list` failed with OS error: {exc}"
    if probe.returncode != 0:
        stderr = probe.stderr.strip() or probe.stdout.strip() or f"exit {probe.returncode}"
        return False, f"`ip netns list` failed: {stderr}"
    return True, None


def evaluate_reference_substrate_preflight() -> ReferenceSubstratePreflight:
    kernel = platform.release()
    wsl = _detect_wsl()
    ip_bin = shutil.which("ip") is not None
    netns_usable = False
    netns_block: str | None = None
    if ip_bin:
        netns_usable, netns_block = _inspect_netns_usable()
    firewall: str | None = None
    if shutil.which("nft"):
        firewall = "nftables"
    elif shutil.which("iptables"):
        firewall = "iptables"
    geteuid = getattr(os, "geteuid", None)
    privileged = geteuid() == 0 if geteuid is not None else False
    python_available = shutil.which("python3") is not None

    block: str | None = None
    if platform.system() != "Linux":
        block = "Linux kernel environment required (WSL2 recommended)"
    elif not ip_bin:
        block = "`ip` utility unavailable"
    elif not netns_usable:
        block = netns_block or "`ip netns` not usable (missing privilege or kernel support)"
    elif firewall is None:
        block = "neither nftables nor iptables available"
    elif not privileged:
        block = "root/CAP_NET_ADMIN privilege required for network namespace qualification"
    elif not python_available:
        block = "python3 unavailable in qualification environment"

    ok = block is None
    return ReferenceSubstratePreflight(
        ok=ok,
        block_reason=block,
        wsl_detected=wsl,
        linux_kernel=kernel,
        ip_available=ip_bin,
        netns_usable=netns_usable,
        firewall_backend=firewall,
        privileged=privileged,
        python_available=python_available,
    )
