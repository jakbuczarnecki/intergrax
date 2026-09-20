# © Artur Czarnecki. All rights reserved.

"""Reference-agent harness context (Tier-2 neutral lab settings).

Runtime materialization lives in
``intergrax.runtime.nexus.agents.reference_harness_runtime`` (host/Nexus composition).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.runtime.modality.modality_profile import ModalityProfile
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.registry.read import ToolRegistryRead
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True, slots=True)
class LabHarnessContext:
    """Policy and strict-mode options injected into reference agents from Tier-3 hosts."""

    policy_bundle: RuntimePolicyBundle
    strict_harness: bool = False
    trace_db_path: Path | None = None
    modality_profile: ModalityProfile | None = None
    tool_wiring_context: ToolWiringContext | None = None
    tool_registry: ToolRegistryRead | None = None


def default_reference_harness() -> LabHarnessContext:
    """Minimal harness defaults when no Tier-3 host injects a bundle."""
    return LabHarnessContext(policy_bundle=RuntimePolicyBundle())
