# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5-R2 — strict production governance composition fixtures."""

from __future__ import annotations

from pathlib import Path

from intergrax.agents.echo.echo_agent import EchoAgent
from intergrax.applications._shared.application_composition_context import (
    composition_for_factory_context,
)
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    PolicyRulesProfile,
    SandboxProfile,
)
from intergrax.contracts.policy_enforcement_mode import PolicyEnforcementMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID

_UCA6C_WORKER_ID = "worker-uca6c-qualified"
_UCA6C_ECHO_ONLY_WORKER_ID = "worker-uca6c-echo-only"
_UCA6C_SANDBOX_CAPABILITY = "sandbox"
_ECHO_BASIC_CAPABILITY = "echo.basic"


class Uca6cQualifiedSandboxWorkerAgent(EchoAgent):
    """Test worker with sandbox governance capability for strict catalog execution."""

    def get_contract(self):
        return (
            super()
            .get_contract()
            .model_copy(
                update={
                    "id": _UCA6C_WORKER_ID,
                    "capabilities": [_UCA6C_SANDBOX_CAPABILITY],
                },
            )
        )


class Uca6cEchoOnlyWorkerAgent(EchoAgent):
    """Strict harness worker without sandbox capability (governance deny fixtures)."""

    def get_contract(self):
        return (
            super()
            .get_contract()
            .model_copy(
                update={
                    "id": _UCA6C_ECHO_ONLY_WORKER_ID,
                    "capabilities": [_ECHO_BASIC_CAPABILITY],
                },
            )
        )


def uca6c_strict_sandbox_env_profile() -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="uca6c-r5-r2-strict"
    )
    profile = profile.model_copy(update={"execution_mode": ExecutionMode.STRICT})
    return profile.model_copy(
        update={
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=True),
            ),
            "policy_rules": PolicyRulesProfile(
                inline_rules=[],
                policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
            ),
        },
    )


def uca6c_strict_worker_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="uca6c_r5_r2",
        name="UCA6C R5 R2",
        route_prefix="/v1/uca6c_r5_r2",
        env_prefix="UCA6C_R5_R2_",
        agents=[
            AgentBinding.mount(
                Uca6cQualifiedSandboxWorkerAgent,
                contract_id=_UCA6C_WORKER_ID,
                capabilities=[_UCA6C_SANDBOX_CAPABILITY],
            ),
        ],
    )


def uca6c_strict_echo_only_worker_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="uca6c_r5_r2_echo_only",
        name="UCA6C R5 R2 Echo Only",
        route_prefix="/v1/uca6c_r5_r2_echo",
        env_prefix="UCA6C_R5_R2_ECHO_",
        agents=[
            AgentBinding.mount(
                Uca6cEchoOnlyWorkerAgent,
                contract_id=_UCA6C_ECHO_ONLY_WORKER_ID,
                capabilities=[_ECHO_BASIC_CAPABILITY],
            ),
        ],
    )


def uca6c_strict_worker_registry(manifest: ApplicationManifest) -> AgentRegistryRead:
    ctx = ApplicationBuildContext.for_manifest(manifest, strict_harness=True)
    composition = composition_for_factory_context(
        ctx,
        policy_bundle=RuntimePolicyBundle(),
    )
    return build_application_registry(manifest, ctx, composition=composition)


def uca6c_test_bound_catalog_step_id(execution_request_id: str) -> str:
    """Test-only catalog step correlation (not production uca6c-scope transport)."""
    return f"uca6c.bound:{execution_request_id}"


def uca6c_high_risk_tool_approval_evidence_for_execution_request(
    *,
    execution_request_id: str,
    tenant_id: str,
    task_id: str,
    run_id: str,
    agent_id: str = _UCA6C_WORKER_ID,
    tool_id: str = CODE_EXEC_TOOL_ID,
) -> None:
    """Removed — legacy TIGAE transport is not part of UCA production graph."""
    raise RuntimeError("TIGAE transport removed from UCA; use DeclarativeHitlApprovalGrant")


def uca6c_high_risk_tool_approval_evidence(
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
    step_id: str,
    agent_id: str = _UCA6C_WORKER_ID,
    tool_id: str = CODE_EXEC_TOOL_ID,
) -> None:
    raise RuntimeError("TIGAE transport removed from UCA; use DeclarativeHitlApprovalGrant")


def uca6c_high_risk_tool_approval_grant(
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
    step_id: str,
    agent_id: str = _UCA6C_WORKER_ID,
    tool_id: str = CODE_EXEC_TOOL_ID,
) -> DeclarativeHitlApprovalGrant:
    """Post-HITL approval artifact with bridge-style invocation scope."""
    return DeclarativeHitlApprovalGrant(
        grant_id=f"dhr_test_grant:{tool_id}:{step_id}",
        invocation_scope_id=f"dhr_test_scope:{step_id}",
        task_id=task_id,
        run_id=run_id,
        step_id=step_id,
        tool_id=tool_id,
        agent_id=agent_id,
        idempotency_key=None,
        matched_rule_ids=("governance.high_risk_approval",),
        human_request_id="uca6c-r5-r3-human",
        policy_provenance_digest=None,
        pause_id="uca6c-r5-r3-pause",
        approved_at="2026-09-22T00:00:00+00:00",
    )


def uca6c_attach_catalog_hitl_grant(catalog_invoker, grant: DeclarativeHitlApprovalGrant) -> None:
    from intergrax.runtime.nexus.agents.catalog_declarative_invoker import (
        CatalogDeclarativeRunBinding,
    )

    catalog_invoker.binding = CatalogDeclarativeRunBinding(
        user_id=catalog_invoker.binding.user_id,
        declarative_hitl_grant=grant,
    )


def uca6c_strict_r6_durable_wiring(
    tmp_path: Path | None = None,
) -> dict[str, object]:
    """STRICT UCA-6C-R6 production-shaped continuation + suspended-operation backing."""
    from intergrax.runtime.execution.continuation.composition import (
        wire_execution_engine_continuation_dependencies,
    )
    from intergrax.runtime.sandbox.durable_sandbox_wiring_binding_resolver import (
        as_durable_wiring_binding_resolver,
    )
    from intergrax.runtime.sandbox.manager import SandboxSessionManager
    from testing_support.uca6c_process_restart_durable_document_store import (
        ProcessRestartQualificationDocumentStore,
    )

    document_store = ProcessRestartQualificationDocumentStore()
    continuation_dependencies = wire_execution_engine_continuation_dependencies()
    kwargs: dict[str, object] = {
        "document_store": document_store,
        "continuation_dependencies": continuation_dependencies,
    }
    if tmp_path is not None:
        manager = SandboxSessionManager(root=tmp_path)
        kwargs["durable_wiring_binding_resolver"] = as_durable_wiring_binding_resolver(
            manager,
        )
        kwargs["sandbox_session_manager"] = manager
    return kwargs


def build_sandbox_session(tmp_path: Path, *, tenant_id: str, task_id: str):
    from intergrax.runtime.sandbox.session import SandboxSession

    return SandboxSession.create(
        tmp_path,
        tenant_id=tenant_id,
        task_id=task_id,
        allowed_operations=frozenset(
            {"echo", "write_file", "read_file", "list_files", "run_python"},
        ),
    )


__all__ = [
    "Uca6cEchoOnlyWorkerAgent",
    "Uca6cQualifiedSandboxWorkerAgent",
    "build_sandbox_session",
    "uca6c_attach_catalog_hitl_grant",
    "uca6c_high_risk_tool_approval_grant",
    "uca6c_strict_echo_only_worker_manifest",
    "uca6c_strict_r6_durable_wiring",
    "uca6c_strict_sandbox_env_profile",
    "uca6c_strict_worker_manifest",
    "uca6c_strict_worker_registry",
    "uca6c_test_bound_catalog_step_id",
]
