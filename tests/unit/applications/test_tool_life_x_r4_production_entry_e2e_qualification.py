# © Artur Czarnecki. All rights reserved.

"""R4 EXT-L4 — canonical harness host production entry wires binding on tool invoke."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.tool_life_execution_recovery_wiring import (
    wire_tool_life_durable_single_host_execution_recovery,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.persistence_topology import DeploymentTopology, PersistenceTopology
from lab_application.manifest import build_lab_manifest
from lab_application.host.settings import LabApplicationSettings
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import PolicyRulesProfile
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ENV = "INTERGRAX_TOOL_LIFE_DURABLE_WIRING_ENABLED"
_MAC = "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY"
_RUN_SEED = "tool-life-r4-prod-entry"
_RUN_ID = canonical_run_id_for_tests(_RUN_SEED)


def _env_profile(topology: PersistenceTopology) -> ApplicationEnvironmentProfile:
    deployment = {
        PersistenceTopology.DURABLE_SINGLE_HOST: DeploymentTopology.SINGLE_HOST,
        PersistenceTopology.PROCESS_LOCAL: DeploymentTopology.PROCESS_LOCAL,
        PersistenceTopology.SHARED_MULTI_HOST: DeploymentTopology.MULTI_HOST,
    }[topology]
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="tool-life.r4.prod.entry")
    return profile.model_copy(
        update={
            "meta": profile.meta.model_copy(
                update={
                    "deployment_topology": deployment,
                    "required_persistence_topology": topology,
                },
            ),
        },
    )


class _EffectIn(BaseModel):
    value: int


class _EffectOut(BaseModel):
    result: int


class _EffectHandler:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[_EffectIn]) -> _EffectOut:
        self.calls += 1
        return _EffectOut(result=request.input.value + 1)


@pytest.fixture(autouse=True)
def _clear_tool_life_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.delenv(_MAC, raising=False)


def test_harness_production_entry_wires_durable_external_operation_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setenv(_MAC, "test-mac-key-r4-prod-entry")
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = _env_profile(PersistenceTopology.DURABLE_SINGLE_HOST)
    idem = tmp_path / "sqlite" / "idempotency.db"
    idem.parent.mkdir(parents=True)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        tenant_id="tenant-r4-prod",
        idempotency_db_path=idem,
        use_in_memory_trace=True,
    )
    assert runtime.tool_life_execution_recovery is not None
    assert runtime.tool_life_bootstrap_resolution is not None
    assert runtime.tool_life_bootstrap_resolution.configuration_enabled is True
    assert runtime.tool_life_bootstrap_resolution.wiring is not None
    runtime.close()


def test_production_invoker_records_trusted_binding_before_physical_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical composition root: bind runs inside RuntimeToolInvoker._execute_once."""
    monkeypatch.setenv(_MAC, "test-mac-key-r4-prod-bind")
    wiring = wire_tool_life_durable_single_host_execution_recovery(
        required_persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        db_path=tmp_path / "tool_life_prod_bind.db",
    )
    assert wiring is not None
    store = wiring.composition.bundle.external_operation_state

    handler = _EffectHandler()
    registry = ToolRegistry()
    contract = ToolContract(
        tool_id="custom.side_effect",
        name="side",
        description="",
        input_schema=_EffectIn,
        output_schema=_EffectOut,
        error_mapping={},
        side_effects=True,
    )
    registry.register(contract, handler)

    from intergrax.applications._shared.declarative_tool_wiring import (
        build_declarative_invoker_from_tool_wiring,
    )
    from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
    from intergrax.tools.contracts.tool_profile import ToolProfile
    from intergrax.tools.registry.wiring import ToolWiringContext
    from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
    from tests.unit.runtime.nexus.tools.test_gr10_r9_orchestration_mse import (
        _RecordingMseBoundary,
    )

    env = _env_profile(PersistenceTopology.DURABLE_SINGLE_HOST)
    tool_wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=["custom.side_effect"]),
        wiring_context=ToolWiringContext(),
        registry=registry,
    )
    idem_store = InMemoryIdempotencyStore()
    invoker_bundle = build_declarative_invoker_from_tool_wiring(
        tool_wiring,
        idempotency_store=idem_store,
        meaningful_side_effect_authorization=_RecordingMseBoundary(allow=True),
        tool_life_execution_recovery=wiring,
        production_mode=False,
    )
    assert invoker_bundle is not None
    invoker = invoker_bundle.tool_invoker
    state = build_runtime_state_for_tests(run_id=_RUN_SEED)
    state.context.config.policy_bundle = wire_policy_bundle(
        env.model_copy(
            update={
                "policy_rules": PolicyRulesProfile(
                    inline_rules=[],
                    policy_enforcement_mode="enforce",
                ),
            },
        ),
    )
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        tool_id="custom.side_effect",
        input=_EffectIn(value=1),
        step_id="step-1",
        idempotency_key="idem-r4-prod-1",
    )
    from intergrax.runtime.external_operations.tool_external_operation_attempt import (
        tool_external_operation_identity,
    )

    with canonical_governed_execution_scope(
        _RUN_SEED,
        governance_tenant_id=state.tenant_id,
        governance_principal_id="tool-life-r4-prod-entry",
    ):
        identity = tool_external_operation_identity(
            tenant_id=state.tenant_id,
            tool_id="custom.side_effect",
            step_id="step-1",
            physical_attempt_sequence=1,
        )
        result = invoker.invoke(
            state=state,
            agent_id="echo",
            request=request,
        )
    assert result.success is True
    assert handler.calls == 1
    assert identity.operation_id is not None
    bound = store.load(state.tenant_id, identity.operation_id)
    assert bound is not None
    assert bound.bound_redrive_effect_contract_id is None
    assert bound.bound_redrive_effect_contract_digest is None
    assert bound.bound_physical_idempotency_key is None
    invoker.close()
