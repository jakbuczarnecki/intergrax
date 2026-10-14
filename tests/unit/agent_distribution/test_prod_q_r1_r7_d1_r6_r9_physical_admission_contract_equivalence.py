# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6-R9 — physical admission provider contract equivalence."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.agent_distribution.deployment import (
    FakeInMemoryRuntimeDeploymentAdapter,
    RuntimeDeploymentAdapter,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from testing_support.agent_distribution.venv_bundle_deployment_fixtures import (
    file_artifact_locator,
    write_minimal_venv_bundle,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter import (
    _revision_for,
)

pytestmark = [pytest.mark.unit]


def test_fake_adapter_does_not_expose_physical_execution_reservation() -> None:
    adapter = FakeInMemoryRuntimeDeploymentAdapter()
    assert not hasattr(adapter, "reserve_unit_execution")
    assert not hasattr(adapter, "release_unit_execution")
    assert not hasattr(adapter, "in_flight_count")


def test_runtime_deployment_port_is_lifecycle_not_physical_admission() -> None:
    protocol_members = {
        name
        for name, member in inspect.getmembers(RuntimeDeploymentAdapter)
        if not name.startswith("_")
    }
    assert "reserve_unit_execution" not in protocol_members
    assert "prepare" in protocol_members
    assert "attest_serving_unit" in protocol_members


def test_venv_adapter_requires_coordinator_before_physical_reserve(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    digest, lock = write_minimal_venv_bundle(root)
    locator = file_artifact_locator(root)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-r9", digest=digest, lock=lock)
    prepared = adapter.prepare(revision, artifact_locator=locator)
    adapter.check_readiness(revision, serving_unit_ref=prepared.serving_unit_ref)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="not admitting"):
        adapter.reserve_unit_execution(prepared.serving_unit_ref)
