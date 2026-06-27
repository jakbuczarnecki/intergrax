# © Artur Czarnecki. All rights reserved.

"""Reference-agent harness context and Nexus runtime bridge (Tier-2 safe imports).

``strict_harness`` here is a **neutral Tier-2 fallback**: it sets ``production_mode``,
injects a minimal ``LabAllowGovernanceService``, and optionally wires trace/modality/tool
slices from ``LabHarnessContext``. It does **not** materialize a full
``ApplicationEnvironmentProfile`` (memory, security, reliability, LLM routing, etc.).

Full strict application wiring lives in Tier-3:
``intergrax.applications._shared.runtime_config_bridge.materialize_runtime_config`` /
``build_runtime_context_from_environment``, and ACP hosts via ``ACPSessionHostContext``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from intergrax.agents.run_environment import EffectiveAgentRunEnvironment
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.modality.modality_profile import ModalityProfile, lab_default_modality_profile
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.wiring.harness_governance import create_lab_allow_governance_service
from intergrax.runtime.wiring.policy_runtime_bridge import apply_policy_bundle_to_runtime_config
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True, slots=True)
class LabHarnessContext:
    """Policy and strict-mode options injected into reference agents from Tier-3 hosts."""

    policy_bundle: RuntimePolicyBundle
    strict_harness: bool = False
    trace_db_path: Path | None = None
    modality_profile: ModalityProfile | None = None
    tool_wiring_context: ToolWiringContext | None = None


def default_reference_harness() -> LabHarnessContext:
    """Minimal harness defaults when no Tier-3 host injects a bundle."""
    return LabHarnessContext(policy_bundle=RuntimePolicyBundle())


def build_lab_agent_runtime_config_from_merged(
    *,
    request: RuntimeRequest,
    llm_adapter: LLMAdapter,
    harness: LabHarnessContext,
    merged: EffectiveAgentRunEnvironment,
) -> RuntimeConfig:
    """ACP-CFG — compose RuntimeConfig from merged environment slices."""
    return build_lab_agent_runtime_config(
        request=request,
        llm_adapter=llm_adapter,
        harness=harness,
        enable_rag=merged.enable_rag,
        enable_websearch=merged.enable_websearch,
    )


def build_lab_agent_runtime_context_from_merged(
    *,
    request: RuntimeRequest,
    llm_adapter: LLMAdapter,
    harness: LabHarnessContext,
    merged: EffectiveAgentRunEnvironment,
) -> RuntimeContext:
    """ACP-CFG — build RuntimeContext using merged profile flags."""
    config = build_lab_agent_runtime_config_from_merged(
        request=request,
        llm_adapter=llm_adapter,
        harness=harness,
        merged=merged,
    )
    governance: GovernanceService | None = None
    if harness.strict_harness:
        governance = cast(
            GovernanceService,
            create_lab_allow_governance_service(),
        )
    return RuntimeContext.build(
        config=config,
        session_manager=SessionManager(storage=InMemorySessionStorage()),
        governance_service=governance,
    )


def build_lab_agent_runtime_config(
    *,
    request: RuntimeRequest,
    llm_adapter: LLMAdapter,
    harness: LabHarnessContext,
    enable_rag: bool = False,
    enable_websearch: bool = False,
) -> RuntimeConfig:
    """Compose ``RuntimeConfig`` with policy bundle and optional strict production mode."""
    trace_path: str | None = None
    if harness.strict_harness and harness.trace_db_path is not None:
        trace_path = str(harness.trace_db_path)

    config = RuntimeConfig(
        llm_adapter=llm_adapter,
        enable_rag=enable_rag,
        enable_websearch=enable_websearch,
        production_mode=harness.strict_harness,
        tenant_id=request.tenant_id,
        trace_db_path=trace_path,
        modality_profile=harness.modality_profile,
        tool_wiring_context=harness.tool_wiring_context,
    )
    return apply_policy_bundle_to_runtime_config(config, harness.policy_bundle)


def build_lab_agent_runtime_context(
    *,
    request: RuntimeRequest,
    llm_adapter: LLMAdapter,
    harness: LabHarnessContext,
    enable_rag: bool = False,
    enable_websearch: bool = False,
) -> RuntimeContext:
    """Build ``RuntimeContext`` for reference agents with policy and strict governance."""
    config = build_lab_agent_runtime_config(
        request=request,
        llm_adapter=llm_adapter,
        harness=harness,
        enable_rag=enable_rag,
        enable_websearch=enable_websearch,
    )
    governance: GovernanceService | None = None
    if harness.strict_harness:
        governance = cast(
            GovernanceService,
            create_lab_allow_governance_service(),
        )
    return RuntimeContext.build(
        config=config,
        session_manager=SessionManager(storage=InMemorySessionStorage()),
        governance_service=governance,
    )


def default_lab_modality_profile() -> ModalityProfile:
    return lab_default_modality_profile()


def lab_harness_context_from_modality_tooling(
    *,
    policy_bundle: RuntimePolicyBundle,
    strict_harness: bool = False,
    trace_db_path: Path | None = None,
    tool_wiring_context: ToolWiringContext | None = None,
) -> LabHarnessContext:
    """Build context for lab builders without importing Tier-3 ``ApplicationBuildContext``."""
    modality_profile = None
    if tool_wiring_context is not None:
        from intergrax.runtime.modality.modality_profile import MODALITY_PROFILE_EXTRA_KEY

        raw_profile = tool_wiring_context.extras.get(MODALITY_PROFILE_EXTRA_KEY)
        if isinstance(raw_profile, ModalityProfile):
            modality_profile = raw_profile
    if modality_profile is None and strict_harness:
        modality_profile = lab_default_modality_profile()
    return LabHarnessContext(
        policy_bundle=policy_bundle,
        strict_harness=strict_harness,
        trace_db_path=trace_db_path,
        modality_profile=modality_profile,
        tool_wiring_context=tool_wiring_context,
    )
