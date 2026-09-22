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
from intergrax.contracts.tool_invocation_governance_approval_evidence import (
    ToolInvocationGovernanceApprovalEvidence,
    tool_invocation_governance_approval_evidence_from_declarative_hitl,
)
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


def uca6c_high_risk_tool_approval_evidence(
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
    step_id: str,
    agent_id: str = _UCA6C_WORKER_ID,
    tool_id: str = CODE_EXEC_TOOL_ID,
) -> ToolInvocationGovernanceApprovalEvidence:
    """Neutral post-HITL approval evidence for HIGH-risk catalog tools."""
    return tool_invocation_governance_approval_evidence_from_declarative_hitl(
        uca6c_high_risk_tool_approval_grant(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            step_id=step_id,
            agent_id=agent_id,
            tool_id=tool_id,
        ),
    )


def uca6c_high_risk_tool_approval_grant(
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
    step_id: str,
    agent_id: str = _UCA6C_WORKER_ID,
    tool_id: str = CODE_EXEC_TOOL_ID,
) -> DeclarativeHitlApprovalGrant:
    """Post-HITL approval artifact for HIGH-risk catalog tools (e.g. code.exec)."""
    return DeclarativeHitlApprovalGrant(
        grant_id=f"uca6c-hitl-grant:{tool_id}:{step_id}",
        invocation_scope_id=f"uca6c-scope:{step_id}",
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
    "uca6c_high_risk_tool_approval_evidence",
    "uca6c_high_risk_tool_approval_grant",
    "uca6c_strict_echo_only_worker_manifest",
    "uca6c_strict_sandbox_env_profile",
    "uca6c_strict_worker_manifest",
    "uca6c_strict_worker_registry",
]
