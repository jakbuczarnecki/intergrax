# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed physical runtime binding for one co-located VENV_BUNDLE serving unit (ADR A1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ColocatedVenvUnitRuntimeBinding:
    """Adapter-issued evidence that one unit's warmed resources are bound to a revision."""

    serving_unit_ref: str
    application_id: str
    application_environment_id: str
    runtime_revision_id: str
    materialization_artifact_digest: str
    artifact_root: Path
    site_packages: Path
    physical_runtime_warmup_evidence: str


__all__ = ["ColocatedVenvUnitRuntimeBinding"]
