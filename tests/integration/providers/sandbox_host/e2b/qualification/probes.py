# © Artur Czarnecki. All rights reserved.

"""Reusable sandbox network probe abstraction for provider qualification."""

from __future__ import annotations

import re
import textwrap
from typing import Protocol, runtime_checkable

from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession

from .models import NetworkProbeResult


@runtime_checkable
class SandboxNetworkProbe(Protocol):
    """Provider-neutral network probe executed inside a sandbox session."""

    def execute(self, session: HostedSandboxSession, target: str) -> NetworkProbeResult:
        """Execute a network probe against ``target`` inside ``session``."""


def _probe_python_code(url: str) -> str:
    return textwrap.dedent(
        f"""
        import urllib.error
        import urllib.request

        url = {url!r}
        redirected = False
        status_code = None
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=20) as response:
                status_code = int(response.status)
                final_url = response.geturl()
                redirected = final_url.rstrip("/") != url.rstrip("/")
                body = response.read(256)
                print("status", status_code)
                print("final_url", final_url)
                print("redirected", redirected)
                print("bytes", len(body))
                print("reachable", True)
        except Exception as exc:
            print("error_type", type(exc).__name__)
            print("reachable", False)
            raise SystemExit(17)
        """,
    ).strip()


def _parse_probe_stdout(stdout: str) -> NetworkProbeResult:
    status_match = re.search(r"^status\s+(\d+)", stdout, flags=re.MULTILINE)
    redirected_match = re.search(r"^redirected\s+(True|False)", stdout, flags=re.MULTILINE)
    reachable_match = re.search(r"^reachable\s+(True|False)", stdout, flags=re.MULTILINE)
    status_code = int(status_match.group(1)) if status_match else None
    redirected = redirected_match.group(1) == "True" if redirected_match else False
    reachable = reachable_match.group(1) == "True" if reachable_match else False
    return NetworkProbeResult(
        reachable=reachable,
        status_code=status_code,
        redirected=redirected,
    )


class HostedPythonNetworkProbe:
    """Execute urllib-based probes via ``HostedSandboxSession.run_python``."""

    def execute(self, session: HostedSandboxSession, target: str) -> NetworkProbeResult:
        result = session.execute("run_python", {"code": _probe_python_code(target)})
        stdout = str((result.output or {}).get("stdout", ""))
        if not result.success:
            return NetworkProbeResult(reachable=False, status_code=None, redirected=False)
        return _parse_probe_stdout(stdout)
