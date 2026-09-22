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
from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.runtime.agent_governance.audit import (
    GovernanceAuditRecorder,
    InMemoryGovernanceAuditSink,
)
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.agent_governance.capability_resolver import (
    InMemoryCapabilityGrantResolver,
)
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    AllowAllPolicyProvider,
)
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead

_UCA6C_WORKER_ID = "worker-uca6c-qualified"
_UCA6C_SANDBOX_CAPABILITY = "sandbox"


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


def uca6c_strict_worker_registry(manifest: ApplicationManifest) -> AgentRegistryRead:
    ctx = ApplicationBuildContext.for_manifest(manifest, strict_harness=True)
    composition = composition_for_factory_context(
        ctx,
        policy_bundle=RuntimePolicyBundle(),
    )
    return build_application_registry(manifest, ctx, composition=composition)


def allow_all_agent_governance_for_worker(
    *,
    tenant_id: str,
    agent_id: str = _UCA6C_WORKER_ID,
) -> AgentRuntimeGovernanceBoundary:
    grant = CapabilityGrant(
        tenant_id=tenant_id,
        agent_id=agent_id,
        allowed_capabilities=frozenset({_UCA6C_SANDBOX_CAPABILITY}),
    )
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine((AllowAllPolicyProvider(),)),
        audit_recorder=GovernanceAuditRecorder(InMemoryGovernanceAuditSink()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)


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
    "Uca6cQualifiedSandboxWorkerAgent",
    "allow_all_agent_governance_for_worker",
    "build_sandbox_session",
    "uca6c_strict_sandbox_env_profile",
    "uca6c_strict_worker_manifest",
    "uca6c_strict_worker_registry",
]
