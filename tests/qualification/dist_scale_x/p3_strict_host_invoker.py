# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X strict-host production-effective invoker wiring for DIST-SCALE-X P3 WL-4."""

from __future__ import annotations

from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.contracts.persistence_topology import DeploymentTopology, PersistenceTopology
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
from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
    materialize_tool_dependency_attempt_boundary,
)
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.tools.providers.http.idempotent_post_qual import HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID
from testing_support.dependency_concurrency_admission_config import (
    tool_dependency_concurrency_admission_configuration,
)
from tests.unit.runtime.governance.test_gr3_r1_explicit_inner_guard_composition import (
    _AllowingTestGuard,
)
from tests.unit.runtime.nexus.tools.test_gr10_r9_orchestration_mse import _RecordingMseBoundary


def strict_durable_application_environment() -> ApplicationEnvironmentProfile:
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="dist-scale-x.p3.wl4.strict")
    return base.model_copy(
        update={
            "meta": base.meta.model_copy(
                update={
                    "execution_mode": ExecutionMode.STRICT,
                    "deployment_topology": DeploymentTopology.SINGLE_HOST,
                    "required_persistence_topology": PersistenceTopology.DURABLE_SINGLE_HOST,
                },
            ),
        },
    )


def http_capability_governance(
    *,
    tenant_id: str,
    agent_id: str,
) -> AgentRuntimeGovernanceBoundary:
    grant = CapabilityGrant(
        tenant_id=tenant_id,
        agent_id=agent_id,
        allowed_capabilities=frozenset({"http"}),
    )
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver((grant,)),
        policy_engine=AgentRuntimePolicyEngine((AllowAllPolicyProvider(),)),
        audit_recorder=GovernanceAuditRecorder(InMemoryGovernanceAuditSink()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)


def build_p3_strict_production_invoker_bundle(
    tool_wiring: ApplicationToolWiring,
    *,
    idempotency_db_path: str,
    tool_life_execution_recovery: object,
    tenant_id: str,
    agent_id: str,
    meaningful_side_effect_authorization: _RecordingMseBoundary,
):
    admission = tool_dependency_concurrency_admission_configuration(
        HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
        max_concurrent_calls=8,
    )
    dependency_boundary = materialize_tool_dependency_attempt_boundary(
        admission,
        production_mode=True,
    )
    assert dependency_boundary is not None
    invoker_bundle = build_declarative_invoker_from_tool_wiring(
        tool_wiring,
        idempotency_store=SQLiteIdempotencyStore(idempotency_db_path),
        agent_runtime_governance=http_capability_governance(
            tenant_id=tenant_id,
            agent_id=agent_id,
        ),
        canonical_inner_execution_guard=_AllowingTestGuard(),
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        dependency_attempt_boundary=dependency_boundary,
        tool_life_execution_recovery=tool_life_execution_recovery,
        production_mode=True,
    )
    assert invoker_bundle is not None
    return invoker_bundle, dependency_boundary


__all__ = [
    "build_p3_strict_production_invoker_bundle",
    "http_capability_governance",
    "strict_durable_application_environment",
    "_RecordingMseBoundary",
]
