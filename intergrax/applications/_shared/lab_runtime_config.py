# © Artur Czarnecki. All rights reserved.

"""Bridge lab harness context into Nexus ``RuntimeConfig`` (Phase U-Pol.1)."""

from __future__ import annotations

from typing import cast

from intergrax.applications._shared.harness_governance import create_lab_allow_governance_service
from intergrax.applications._shared.lab_harness_context import LabHarnessContext
from intergrax.applications._shared.runtime_config_bridge import apply_policy_bundle_to_runtime_config
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


def build_lab_agent_runtime_config(
    *,
    request: RuntimeRequest,
    llm_adapter: LLMAdapter,
    harness: LabHarnessContext,
    pipeline: RuntimePipeline | None = None,
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
    )
    if pipeline is not None:
        config.pipeline = pipeline
    return apply_policy_bundle_to_runtime_config(config, harness.policy_bundle)


def build_lab_agent_runtime_context(
    *,
    request: RuntimeRequest,
    llm_adapter: LLMAdapter,
    harness: LabHarnessContext,
    pipeline: RuntimePipeline | None = None,
    enable_rag: bool = False,
) -> RuntimeContext:
    """Build ``RuntimeContext`` for lab reference agents with policy + strict governance."""
    config = build_lab_agent_runtime_config(
        request=request,
        llm_adapter=llm_adapter,
        harness=harness,
        pipeline=pipeline,
        enable_rag=enable_rag,
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
