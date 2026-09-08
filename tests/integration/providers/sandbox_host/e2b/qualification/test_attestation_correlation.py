# © Artur Czarnecki. All rights reserved.

"""Unit-style attestation correlation tests — no physical provider required."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist
from tests.integration.providers.sandbox_host.e2b.qualification import (
    HostProbeEvidence,
    NetworkProbeResult,
    ObservedNetworkScope,
    ProviderAttestationCorrelation,
    ProviderAttestationCorrelationEvidence,
    ProviderAttestationEvidence,
)

pytestmark = pytest.mark.qualification

_ALLOWED = "https://api.allowed.com"
_DENIED = "https://blocked.com"
_REQUESTED = canonicalize_network_egress_allowlist([_ALLOWED])

_SECRET_PATTERNS = (
    re.compile(r"sk-[a-zA-Z0-9]{8,}"),
    re.compile(r"E2B_API_KEY", re.IGNORECASE),
    re.compile(r"INTERGRAX_E2B_API_KEY", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{8,}"),
)


def _probe(target: str, *, reachable: bool) -> HostProbeEvidence:
    return HostProbeEvidence(
        target=target,
        result=NetworkProbeResult(
            reachable=reachable,
            status_code=200 if reachable else None,
            redirect_target=None,
            latency_ms=0.0,
            redirected=False,
        ),
    )


def _attestation(
    *,
    enforced_hosts: tuple[str, ...] | None,
    allowlist_enforced: bool | None = True,
) -> ProviderAttestationEvidence:
    return ProviderAttestationEvidence(
        provider_id="test-provider",
        network_egress_allowlist_enforced=allowlist_enforced,
        enforced_network_hosts=enforced_hosts,
    )


def test_provider_attestation_matches_execution() -> None:
    evidence = ProviderAttestationCorrelation.evaluate(
        requested_allowlist=_REQUESTED,
        provider_attestation=_attestation(enforced_hosts=(_ALLOWED,)),
        allowed_probe=_probe(_ALLOWED, reachable=True),
        denied_probe=_probe(_DENIED, reachable=False),
    )
    assert evidence.correlation_result == "PASS"
    assert evidence.attestation_verified is True
    assert evidence.execution_verified is True
    assert evidence.passes() is True


def test_provider_attestation_scope_mismatch_denied() -> None:
    evidence = ProviderAttestationCorrelation.evaluate(
        requested_allowlist=_REQUESTED,
        provider_attestation=_attestation(enforced_hosts=("https://other.example.com",)),
        allowed_probe=_probe(_ALLOWED, reachable=True),
        denied_probe=_probe(_DENIED, reachable=False),
    )
    assert evidence.correlation_result == "DENIED"
    assert evidence.attestation_verified is False
    assert evidence.execution_verified is True


def test_observed_execution_mismatch_denied() -> None:
    evidence = ProviderAttestationCorrelation.evaluate(
        requested_allowlist=_REQUESTED,
        provider_attestation=_attestation(enforced_hosts=(_ALLOWED,)),
        allowed_probe=_probe(_ALLOWED, reachable=True),
        denied_probe=_probe(_DENIED, reachable=True),
    )
    assert evidence.correlation_result == "DENIED"
    assert evidence.attestation_verified is True
    assert evidence.execution_verified is False


def test_missing_attestation_fails_closed() -> None:
    evidence = ProviderAttestationCorrelation.evaluate(
        requested_allowlist=_REQUESTED,
        provider_attestation=None,
        allowed_probe=_probe(_ALLOWED, reachable=True),
        denied_probe=_probe(_DENIED, reachable=False),
    )
    assert evidence.correlation_result == "DENIED"
    assert evidence.attested_scope is None
    assert evidence.attestation_verified is False


def test_attestation_evidence_is_secret_free() -> None:
    evidence = ProviderAttestationCorrelationEvidence(
        requested_scope=("https://api.allowed.com",),
        attested_scope=("https://api.allowed.com",),
        observed_scope=ObservedNetworkScope(
            allowed_target=_ALLOWED,
            allowed_reachable=True,
            denied_target=_DENIED,
            denied_reachable=False,
        ),
        attestation_verified=True,
        execution_verified=True,
        correlation_result="PASS",
    )
    serialized = json.dumps(evidence.to_mapping())
    for pattern in _SECRET_PATTERNS:
        assert pattern.search(serialized) is None


_QUALIFICATION_ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "path",
    sorted(_QUALIFICATION_ROOT.rglob("*.py")),
    ids=lambda path: path.relative_to(_QUALIFICATION_ROOT).as_posix(),
)
def test_no_nexus_dependency(path: Path) -> None:
    import ast

    forbidden = (
        "intergrax.nexus",
        "intergrax.runtime.nexus",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported).lower()
    for prefix in forbidden:
        assert prefix.lower() not in joined, f"{path} imports forbidden surface {prefix}"
