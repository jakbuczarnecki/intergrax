# © Artur Czarnecki. All rights reserved.

"""Behavioral INTERNAL RPL proofs (tier ownership and PROD-Q adapter gate)."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingLauncherError,
    _require_venv_production_adapter,
)
from tests.qualification.ebh_5._closed_world_discovery import discover_composition_root_paths
from tests.qualification.ebh_5.test_ebh_5_l2_integrity_gates import test_ebh5_q08_composition_roots_exist

_REPO = Path(__file__).resolve().parents[3]


def prove_application_composition_root_internal_positive() -> None:
    adr = _REPO / "docs/project/maintainers/architecture/ADR/ADR-EBH-7-INTERNAL-APPLICATION-COMPOSITION-ROOT.md"
    assert adr.is_file()
    test_ebh5_q08_composition_roots_exist()
    roots = discover_composition_root_paths()
    assert "intergrax/applications/_shared/memory_wiring.py" in roots


def prove_application_composition_root_internal_negative() -> None:
    factory = (_REPO / "intergrax/integrations/registry/factory.py").read_text(encoding="utf-8")
    assert "from applications." not in factory
    assert "import applications." not in factory


def prove_prodq_colocated_adapter_internal_positive() -> None:
    adr = _REPO / "docs/project/maintainers/architecture/ADR/ADR-EBH-7-INTERNAL-PRODQ-COLOCATED-ADAPTER-GATE.md"
    assert adr.is_file()
    trusted = (Path.cwd(),)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(trusted_artifact_parent_roots=trusted)
    accepted = _require_venv_production_adapter(adapter)
    assert isinstance(accepted, VenvBundleColocatedRuntimeDeploymentAdapter)


def prove_prodq_colocated_adapter_internal_negative() -> None:
    with pytest.raises(
        StrictProductionServingLauncherError,
        match="FakeInMemoryRuntimeDeploymentAdapter",
    ):
        _require_venv_production_adapter(FakeInMemoryRuntimeDeploymentAdapter())

    class _NotVenvAdapter:
        """Wrong runtime deployment authority for strict production."""

    with pytest.raises(
        StrictProductionServingLauncherError,
        match="VenvBundleColocatedRuntimeDeploymentAdapter",
    ):
        _require_venv_production_adapter(_NotVenvAdapter())  # type: ignore[arg-type]
