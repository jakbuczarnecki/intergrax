# © Artur Czarnecki. All rights reserved.

"""Reference-agent harness context (Tier-2 neutral lab settings).

Runtime materialization lives in
``intergrax.runtime.nexus.agents.reference_harness_runtime`` (host/Nexus composition).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from intergrax.runtime.modality.modality_profile import ModalityProfile
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.registry.read import ToolRegistryRead
from intergrax.tools.registry.wiring import ToolWiringContext

_RUNTIME_MODULE = "intergrax.runtime.nexus.agents.reference_harness_runtime"


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


def build_lab_agent_runtime_config_from_merged(*args, **kwargs):
    mod = importlib.import_module(_RUNTIME_MODULE)
    return mod.build_lab_agent_runtime_config_from_merged(*args, **kwargs)


def build_lab_agent_runtime_context_from_merged(*args, **kwargs):
    mod = importlib.import_module(_RUNTIME_MODULE)
    return mod.build_lab_agent_runtime_context_from_merged(*args, **kwargs)


def build_lab_agent_runtime_config(*args, **kwargs):
    mod = importlib.import_module(_RUNTIME_MODULE)
    return mod.build_lab_agent_runtime_config(*args, **kwargs)


def build_lab_agent_runtime_context(*args, **kwargs):
    mod = importlib.import_module(_RUNTIME_MODULE)
    return mod.build_lab_agent_runtime_context(*args, **kwargs)


def default_lab_modality_profile():
    mod = importlib.import_module(_RUNTIME_MODULE)
    return mod.default_lab_modality_profile()


def lab_harness_context_from_modality_tooling(*args, **kwargs):
    mod = importlib.import_module(_RUNTIME_MODULE)
    return mod.lab_harness_context_from_modality_tooling(*args, **kwargs)
