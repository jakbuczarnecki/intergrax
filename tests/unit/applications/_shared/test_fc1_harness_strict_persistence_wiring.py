"""FC-1 Parts B and C1: STRICT harness and worker composition state their persistence posture.

NT-FC1-19..22, 24 and 26 of T-IGX-043-A (EQOA-A4-037, A4-024, A4-015b): a profile that disables task
memory is not re-enabled by the environment, a STRICT harness or worker without a durable human
decision store or the typed acknowledgement is refused, reject and escalate verdicts reach a durable
decision record through materialization, and Tier-2 agents take the runtime posture from the host.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from intergrax.agents.authoring.acp_stub_reflex import build_agent_runtime_context
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
)
from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
    resolve_reference_production_strict_host_environment,
)
from intergrax.applications._shared.roster_agent_contract_authority import (
    materialize_manifest_contract_authority_lab_compat,
)
from intergrax.applications._shared.task_memory_wiring import wire_task_memory_from_profile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.integrations.providers.relational_store.sqlite.paths import ENV_TASK_MEMORY_DB
from intergrax.runtime.execution.agent_runtime_context import RuntimeContext
from intergrax.runtime.execution.agent_runtime_io import RuntimeRequest
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    backing_execution_continuation_state_store,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
)
from intergrax.runtime.execution.environment_orchestration_materialization import (
    materialize_host_orchestration_backend,
)
from intergrax.runtime.execution.host_orchestration_loop_init_spec import HostOrchestrationLoopInitSpec
from intergrax.runtime.execution.host_runtime_config import RuntimeConfig
from intergrax.runtime.execution.worker_host_task_execution_composition import (
    build_worker_host_task_execution_from_registry,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.persistence_contract import (
    HumanDecisionPersistence,
    InMemoryHumanDecisionPersistence,
)
from intergrax.runtime.human.persistence_errors import HumanDecisionPersistenceNotConfiguredError
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
    ToolDependencyAttemptBoundaryMaterializationError,
)
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskHumanInput,
    TaskLongRunningOptions,
)
from testing_support.dependency_concurrency_admission_config import (
    tool_dependency_concurrency_admission_configuration,
)
from testing_support.nexus_hitl_test_agent import (
    NexusBasicHitlTestAgent,
    prepare_nexus_hitl_resume_task,
)
from testing_support.nexus_host_task_execution import (
    admit_certified_internal_harness_root_governance_identity,
    build_reference_allowing_root_execution_authority_admission,
)
from testing_support.nexus_lab_task_execution import run_lab_nexus_task

from external_contractor_adapter.external_contractor_adapter_agent import ExternalContractorAdapterAgent
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.orchestration_decision_requirement_policy import (
    default_governed_contractor_harness_orchestration_decision_requirement_policy,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from organization_worker.organization_worker_agent import OrganizationWorkerAgent

pytestmark = pytest.mark.unit


class _InMemoryCollaborativeWorkStoreOwner:
    def close(self) -> None:
        return None


def _strict_harness(
    tmp_path: Path,
    *,
    human_decision_store: HumanDecisionPersistence | None = None,
    disable_human_decision_persistence: bool = False,
    enabled_bundles: tuple[str, ...] = (),
) -> HarnessHostRuntime:
    """Real STRICT governed-contractor ``build_harness_host_runtime`` with legal prerequisites."""
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    base_env = manifest.environment or build_governed_contractor_environment_profile(settings)
    env = resolve_reference_production_strict_host_environment(base_env)
    if env.reliability_profile.dependency_concurrency_admission is None:
        env = env.model_copy(
            update={
                "reliability_profile": env.reliability_profile.model_copy(
                    update={
                        "dependency_concurrency_admission": tool_dependency_concurrency_admission_configuration(
                            "fc1-harness-tool",
                        ),
                    },
                ),
            },
        )
    if enabled_bundles:
        merged = [*env.tool_profile.enabled_bundles, *enabled_bundles]
        env = env.model_copy(
            update={"tool_profile": env.tool_profile.model_copy(update={"enabled_bundles": merged})},
        )
    manifest_for_runtime = manifest.model_copy(update={"environment": env})
    platform = build_reference_production_platform_persistence(db_path=tmp_path / "fc1-kv.db")
    backing = ExecutionContinuationDurableBacking()
    backing_execution_continuation_state_store(backing)
    continuation_store = execution_continuation_state_store_from_durable_export(
        export_durable_continuation_state(backing),
    )
    return build_harness_host_runtime(
        manifest_for_runtime,
        env,
        settings=settings,
        tenant_id=manifest.app_id,
        registry_projection=build_governed_contractor_test_registry_projection(),
        key_value_cache=platform.kv_store,
        document_store=platform.document_store,
        execution_continuation_state_store=continuation_store,
        collaborative_work_repositories=CollaborativeWorkRepositories(
            membership=InMemoryWorkspaceMembershipRepository(),
            delegation=InMemoryAuthorityDelegationRepository(),
            principal_authority=InMemoryPrincipalAuthorityRepository(),
            policy=InMemoryCollaborativePolicyRepository(),
            operation_profile=InMemoryCollaborativeOperationPolicyProfileRepository(),
            store=_InMemoryCollaborativeWorkStoreOwner(),
        ),
        orchestration_decision_requirement_policy=(
            default_governed_contractor_harness_orchestration_decision_requirement_policy()
        ),
        agent_contract_authority=materialize_manifest_contract_authority_lab_compat(manifest_for_runtime),
        human_decision_store=human_decision_store,
        disable_human_decision_persistence=disable_human_decision_persistence,
    )


def test_profile_disabling_task_memory_is_not_reenabled_by_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NT-FC1-19 (V4 / CK-08)."""
    ambient = tmp_path / "ambient_task_memory.db"
    monkeypatch.setenv(ENV_TASK_MEMORY_DB, str(ambient))
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc1.nt19")
    env.execution_mode = ExecutionMode.STRICT
    memory = env.memory_profile
    for flag in ("enable_task_memory", "enable_user_memory", "enable_org_memory", "enable_long_term_memory"):
        setattr(memory, flag, False)
    wiring = wire_task_memory_from_profile(env)
    assert wiring.store is None
    assert wiring.db_path is None
    assert not ambient.exists()


def test_strict_harness_runtime_refuses_missing_human_store_without_ack(tmp_path: Path) -> None:
    """NT-FC1-20: refused before the loop is returned."""
    with pytest.raises(HumanDecisionPersistenceNotConfiguredError):
        _strict_harness(tmp_path)


def test_strict_harness_runtime_accepts_the_typed_acknowledgement(tmp_path: Path) -> None:
    """Control for NT-FC1-20: the acknowledgement path composes."""
    runtime = _strict_harness(tmp_path, disable_human_decision_persistence=True)
    assert runtime.environment.execution_mode is ExecutionMode.STRICT


def test_strict_worker_composition_refuses_missing_persistence() -> None:
    """NT-FC1-21."""
    with pytest.raises(HumanDecisionPersistenceNotConfiguredError):
        build_worker_host_task_execution_from_registry(
            AgentRegistry(),
            production_mode=True,
            disable_task_memory_persistence=True,
            disable_runtime_event_persistence=True,
            admit_root_governance_identity=admit_certified_internal_harness_root_governance_identity,
            root_authority_admission=build_reference_allowing_root_execution_authority_admission(),
        )


def test_acknowledgement_is_refused_when_hitl_tools_are_wired(tmp_path: Path) -> None:
    """NT-FC1-26 (case 1): the acknowledgement and wired HITL tools are mutually exclusive."""
    with pytest.raises(ValueError, match="HITL tool bundle"):
        _strict_harness(
            tmp_path,
            disable_human_decision_persistence=True,
            enabled_bundles=("hitl",),
        )


def test_tier2_build_context_takes_posture_from_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """NT-FC1-24 (Part C1): the three Tier-2 sites report production_mode from the composing host.

    ``RuntimeContext.build`` is replaced by a recorder: the production runtime context needs much more
    host machinery (dependency admission, agent runtime governance) than a Tier-2 stub can supply, which
    is recorded as a residual in the ticket; this test pins only which posture each site hands over.
    """
    captured: list[RuntimeConfig] = []

    def _record(cls: object, *, config: RuntimeConfig, **_: object) -> object:
        captured.append(config)
        return object()

    monkeypatch.setattr(RuntimeContext, "build", classmethod(_record))
    request = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s",
        message="m",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        tenant_id="t1",
    )
    ExternalContractorAdapterAgent(require_sanctioned_external_work_mutations=True).build_context(request)
    ExternalContractorAdapterAgent().build_context(request)
    OrganizationWorkerAgent(production_mode=True).build_context(request)
    OrganizationWorkerAgent().build_context(request)
    build_agent_runtime_context(request, MagicMock(), production_mode=True)
    build_agent_runtime_context(request, MagicMock())
    assert [config.production_mode for config in captured] == [True, False, True, False, True, False]


def test_strict_tier2_context_without_host_admission_configuration_fails_closed() -> None:
    """The production runtime context needs the host's dependency admission configuration."""
    request = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s",
        message="m",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        tenant_id="t1",
    )
    strict_contractor = ExternalContractorAdapterAgent(require_sanctioned_external_work_mutations=True)
    with pytest.raises(ToolDependencyAttemptBoundaryMaterializationError):
        strict_contractor.build_context(request)


class _RecordingHumanDecisionStore(InMemoryHumanDecisionPersistence):
    """Recording store port: keeps every persisted decision in order."""


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [HumanResponseVerdict.REJECT, HumanResponseVerdict.ESCALATE])
async def test_reject_and_escalate_verdicts_are_recorded_through_materialization(
    tmp_path: Path,
    verdict: HumanResponseVerdict,
) -> None:
    """NT-FC1-22 (V5 / NT-05): the verdict scenarios of test_nexus_loop_governance, run through the spec."""
    registry = AgentRegistry()
    registry.register(NexusBasicHitlTestAgent())
    store = _RecordingHumanDecisionStore()
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    spec = HostOrchestrationLoopInitSpec(
        production_mode=False,
        human_decision_store=store,
        checkpoint_store=checkpoint_store,
    )
    loop = materialize_host_orchestration_backend(registry, spec)._backend  # noqa: SLF001
    task_id = mint_task_id()
    run_id = mint_run_id()
    paused = await run_lab_nexus_task(
        loop,
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            task_id=task_id,
            options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
        ),
        run_id=run_id,
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    checkpoint = checkpoint_store.get_latest(task_id, "t1")
    assert checkpoint is not None and checkpoint.runtime is not None
    resume_token = paused.summary.resume_token
    assert resume_token
    word = verdict.value
    resumed = Task(
        tenant_id="t1",
        user_id="u1",
        message="sensitive action",
        context=TaskContext(capability="hitl.basic"),
        task_id=task_id,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token=resume_token),
            human=TaskHumanInput(response_text=word, verdict=verdict),
        ),
        metadata={"human_response": word, "resume_token": resume_token},
    )
    prepare_nexus_hitl_resume_task(
        resumed,
        loaded=checkpoint,
        run_id=run_id,
        human_rejected=verdict is HumanResponseVerdict.REJECT,
    )
    await run_lab_nexus_task(
        loop,
        resumed,
        run_id=run_id,
        attempt_id=checkpoint.runtime.attempt_id,
        resume_checkpoint=checkpoint,
    )
    records = store.list_for_task(task_id, "t1")
    assert [r.verdict for r in records] == [verdict]
