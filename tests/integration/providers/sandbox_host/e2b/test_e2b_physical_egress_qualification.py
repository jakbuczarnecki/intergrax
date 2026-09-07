# © Artur Czarnecki. All rights reserved.

"""Physical E2B exact-host egress qualification (AW-7C P0-3A)."""

from __future__ import annotations

import os
import textwrap
import warnings

import pytest

from intergrax.integrations.providers.sandbox_host.e2b.bundle import create_e2b_sandbox_host
from intergrax.integrations.providers.sandbox_host.e2b.config import E2bSandboxHostConfig
from intergrax.runtime.sandbox.contracts import SandboxSecurityRequirements
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist

pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.sandbox_provider,
    pytest.mark.qualification,
]

_ALLOWED_HOST = os.environ.get("INTERGRAX_E2B_QUAL_ALLOWED_HOST", "https://httpbin.org").rstrip("/")
_DENIED_HOST = os.environ.get("INTERGRAX_E2B_QUAL_DENIED_HOST", "https://www.google.com").rstrip("/")
_REDIRECT_URL = os.environ.get(
    "INTERGRAX_E2B_QUAL_REDIRECT_URL",
    "https://httpbin.org/redirect-to?url=https://www.google.com",
)


def _credential_available() -> bool:
    try:
        E2bSandboxHostConfig.from_env().resolved_api_key()
    except Exception:
        return False
    return True


def _fetch_code(url: str) -> str:
    return textwrap.dedent(
        f"""
        import urllib.error
        import urllib.request

        url = {url!r}
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                body = response.read(256)
                print("status", response.status)
                print("bytes", len(body))
        except Exception as exc:
            print("error_type", type(exc).__name__)
            print("error", exc)
            raise SystemExit(17)
        """,
    ).strip()


@pytest.fixture(scope="module")
def qualified_backend():
    if not _credential_available():
        pytest.skip("E2B_API_KEY / INTERGRAX_E2B_API_KEY unavailable for physical qualification")
    try:
        backend = create_e2b_sandbox_host().client
    except Exception as exc:
        pytest.skip(f"E2B backend unavailable: {type(exc).__name__}")
    yield backend


@pytest.fixture(scope="module")
def qualified_session(qualified_backend):
    allowed = canonicalize_network_egress_allowlist([_ALLOWED_HOST])
    requirements = SandboxSecurityRequirements(
        isolation_tier="cloud",
        network_egress="allowlist",
        network_egress_allowlist=allowed,
    )
    session = HostedSandboxSession.open(
        qualified_backend,
        tenant_id="qual-tenant",
        task_id="qual-task",
        security_requirements=requirements,
        allowed_operations=frozenset({"run_python"}),
    )
    if session is None:
        pytest.fail("qualified E2B session was not admitted")
    try:
        yield session
    finally:
        cleanup_error: Exception | None = None
        try:
            qualified_backend.destroy_session(session.session_id)
        except Exception as exc:  # noqa: BLE001 — qualification cleanup boundary
            cleanup_error = exc
        if cleanup_error is not None:
            warnings.warn(
                f"E2B physical qualification sandbox cleanup failed: {cleanup_error}",
                UserWarning,
                stacklevel=1,
            )


def test_physical_allowed_host_reachable(qualified_session: HostedSandboxSession) -> None:
    result = qualified_session.execute("run_python", {"code": _fetch_code(_ALLOWED_HOST)})
    assert result.success is True
    assert "error_type" not in (result.output or {}).get("stdout", "")


def test_physical_denied_host_blocked(qualified_session: HostedSandboxSession) -> None:
    result = qualified_session.execute("run_python", {"code": _fetch_code(_DENIED_HOST)})
    assert result.success is False
    stdout = str((result.output or {}).get("stdout", ""))
    assert "error_type" in stdout or "exit_code" in (result.output or {})


def test_physical_redirect_escape_denied(qualified_session: HostedSandboxSession) -> None:
    result = qualified_session.execute("run_python", {"code": _fetch_code(_REDIRECT_URL)})
    assert result.success is False
