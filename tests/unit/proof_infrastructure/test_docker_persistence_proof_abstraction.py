# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER_PATH = (
    _REPO_ROOT / "proof_infrastructure/governed_hybrid_knowledge_proof/docker_persistence_proof.py"
)


def test_docker_persistence_proof_has_no_forbidden_low_level_imports() -> None:
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    forbidden = (
        "httpx",
        "pymongo",
        "MongoClient",
        "MongoSecurityStatusStore",
        "SecurityStatusStore",
        "HttpxSecurityStatusReadClient",
        "SecurityStatusIntegration",
        "live_call_failure_reason_for_error_code",
        "LiveErrorCodeV1",
    )
    for token in forbidden:
        assert token not in source, (
            f"docker_persistence_proof.py contains forbidden import token: {token}"
        )


def test_docker_persistence_proof_depends_on_proof_abstractions() -> None:
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    required = (
        "GovernedHybridDockerEnvironmentV1",
        "build_governed_security_docker_scenario",
    )
    for token in required:
        assert token in source, (
            f"docker_persistence_proof.py must depend on proof abstraction: {token}"
        )
