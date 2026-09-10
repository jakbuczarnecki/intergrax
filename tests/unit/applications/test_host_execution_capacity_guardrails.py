# © Artur Czarnecki. All rights reserved.

"""W0 — host capacity guardrails (strict mode, process-local semantics)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.applications._shared.host_execution_capacity_policy import (
    HostExecutionCapacityPolicyError,
    validate_strict_host_execution_capacity,
)
from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications._shared.orchestration_wiring import (
    resolve_max_inflight_nodes,
    resolve_max_parallel_nodes,
    resolve_orchestration_runtime_settings,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_strict_mode_requires_explicit_host_capacity_caps() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="w0.strict.missing").model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(long_running_enabled=True),
        },
    )
    with pytest.raises(HostExecutionCapacityPolicyError, match="max_parallel_nodes"):
        validate_strict_host_execution_capacity(env)


def test_balanced_lab_profile_allows_unset_capacity_caps() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    assert env.execution_mode is not ExecutionMode.STRICT
    validate_strict_host_execution_capacity(env)
    assert resolve_max_parallel_nodes(env) is None
    assert resolve_max_inflight_nodes(env) is None


def test_product_defaults_include_explicit_capacity_caps() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="w0.product.template")
    assert env.execution_mode is ExecutionMode.STRICT
    assert env.orchestration_profile.max_parallel_nodes == 8
    assert env.orchestration_profile.max_inflight_nodes == 8
    validate_strict_host_execution_capacity(env)


def test_orchestration_profile_platform_maximum_is_256() -> None:
    with pytest.raises(ValidationError):
        OrchestrationProfile(max_parallel_nodes=257)
    with pytest.raises(ValidationError):
        OrchestrationProfile(max_inflight_nodes=0)


def test_nexus_factory_canonical_path_receives_resolved_caps() -> None:
    env = ApplicationEnvironmentProfile.async_batch_defaults(max_parallel_nodes=5)
    settings = resolve_orchestration_runtime_settings(env)
    assert settings.max_parallel_nodes == 5
    assert settings.max_inflight_nodes == 5
    loop = build_nexus_loop_from_environment(AgentRegistry(), env=env)
    assert loop.graph_executor is not None


def test_w0_policy_documents_process_local_not_cluster_wide() -> None:
    from intergrax.applications._shared import host_execution_capacity_policy as policy

    doc = policy.__doc__ or ""
    assert "process-local" in doc.lower()
    assert "cluster" in doc.lower()
