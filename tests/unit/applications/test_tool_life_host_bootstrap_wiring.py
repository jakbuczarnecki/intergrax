# © Artur Czarnecki. All rights reserved.

"""G-A07 host bootstrap wiring — default-off and topology gates."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from intergrax.applications._shared.tool_life_host_bootstrap_wiring import (
    is_tool_life_durable_wiring_configuration_enabled,
    resolve_tool_life_host_bootstrap_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.persistence_topology import DeploymentTopology, PersistenceTopology

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ENV = "INTERGRAX_TOOL_LIFE_DURABLE_WIRING_ENABLED"
_MAC = "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY"


def _env_profile(topology: PersistenceTopology) -> ApplicationEnvironmentProfile:
    deployment = {
        PersistenceTopology.DURABLE_SINGLE_HOST: DeploymentTopology.SINGLE_HOST,
        PersistenceTopology.PROCESS_LOCAL: DeploymentTopology.PROCESS_LOCAL,
        PersistenceTopology.SHARED_MULTI_HOST: DeploymentTopology.MULTI_HOST,
    }[topology]
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="tool-life.bootstrap.qual")
    return profile.model_copy(
        update={
            "meta": profile.meta.model_copy(
                update={"deployment_topology": deployment},
            ),
        },
    )


@pytest.fixture(autouse=True)
def _clear_tool_life_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.delenv(_MAC, raising=False)


def test_wiring_disabled_by_default() -> None:
    assert is_tool_life_durable_wiring_configuration_enabled() is False
    resolution = resolve_tool_life_host_bootstrap_wiring(
        _env_profile(PersistenceTopology.DURABLE_SINGLE_HOST),
        idempotency_db_path=Path("/tmp/idempotency.db"),
    )
    assert resolution.configuration_enabled is False
    assert resolution.wiring is None


def test_wiring_fail_closed_wrong_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    resolution = resolve_tool_life_host_bootstrap_wiring(
        _env_profile(PersistenceTopology.PROCESS_LOCAL),
        idempotency_db_path=Path("/tmp/idempotency.db"),
    )
    assert resolution.configuration_enabled is True
    assert resolution.topology_eligible is False
    assert resolution.wiring is None


def test_wiring_materializes_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setenv(_MAC, "test-mac-key-g-a07-bootstrap")
    db_parent = tmp_path / "sqlite"
    db_parent.mkdir()
    resolution = resolve_tool_life_host_bootstrap_wiring(
        _env_profile(PersistenceTopology.DURABLE_SINGLE_HOST),
        idempotency_db_path=db_parent / "idempotency.db",
    )
    assert resolution.wiring is not None
    assert resolution.wiring.composition.commit_authority_loss_decision_authority is not None
