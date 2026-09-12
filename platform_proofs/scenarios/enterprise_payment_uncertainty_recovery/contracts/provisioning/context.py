"""Typed provisioning context — scenario identity, variant selection, execution scope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ScenarioIdentity:
    """Stable qualification identity for the scenario dataset package."""

    qualification_id: str
    scenario_slug: str


@dataclass(frozen=True, slots=True)
class ScenarioVariantSelection:
    """Harness-selected variant; provisioning materializes logical truth for this id only."""

    variant_id: str


@dataclass(frozen=True, slots=True)
class ProvisioningExecutionContext:
    """Single proof-run execution scope for provisioning (not a storage technology choice)."""

    run_id: str
    dataset_package_root: Path


@dataclass(frozen=True, slots=True)
class ProvisioningContext:
    """Full input to every provisioning lifecycle phase."""

    identity: ScenarioIdentity
    variant: ScenarioVariantSelection
    execution: ProvisioningExecutionContext
