# © Artur Czarnecki. All rights reserved.

"""PDC-D1-R011 (upgrade), R014 (retire), R017 (transition evidence) — ADR-PDC-01.

Real ``AgentPlatformAdminService`` over in-memory stores; no mocks of the unit under test.
"""

from __future__ import annotations

import pytest
from intergrax.agent_distribution.admin_models import (
    AgentUpgradeOutcome,
    RetireAgentRequest,
    RollbackRuntimeRevisionRequest,
)
from intergrax.agent_distribution.agent_manager_command_facade import (
    AgentManagerCommandFacade,
)
from intergrax.agent_distribution.errors import (
    AgentDistributionNotFoundError,
    InstallationLifecycleError,
    InstallationSlotConflict,
)
from intergrax.agent_distribution.events import (
    EVENT_INSTALLATION_ACTIVATED,
    EVENT_INSTALLATION_CREATED,
    EVENT_INSTALLATION_FAILED_CANDIDATE,
    EVENT_INSTALLATION_RETIRED,
    EVENT_INSTALLATION_VERIFIED,
    EVENT_RUNTIME_REVISION_CREATED,
    EVENT_RUNTIME_REVISION_VALIDATED,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentInstallationStore,
)
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.installation_service import (
    INSTALLATION_STATE_EVIDENCE_EVENT,
    InstallationService,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.upgrade_admission import (
    AgentUpgradeRejectionReason,
)
from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.agent_routing_policy import evaluate_agent_routing
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager
from tests.qualification.pdc_x.d1 import _d1_harness as h

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_V1 = "1.0.0"
_V11 = "1.1.0"


def _upgrade(stack: h.D1Stack, candidate: h.Release, *, expected: str = h.installation_id(_V1)):
    stack.register(candidate)
    return stack.service.upgrade_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=candidate.upgrade_request(expected_active=expected),
        principal=h.principal(),
    )


def _slot_states(stack: h.D1Stack) -> dict[str, InstallationState]:
    return {
        record.installation_id: record.installation_state
        for record in stack.installation_store.list_installations_for_slot(h.ENV, h.SLOT)
    }


def _stack_with_v1(*, tool_allowlist: tuple[str, ...] = ()) -> h.D1Stack:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(_V1), tool_allowlist=tool_allowlist)
    return stack


# --------------------------------------------------------------------------- R011 upgrade


def test_r011_upgrade_accepts_compatible_candidate_and_refreshes_binding() -> None:
    stack = _stack_with_v1()
    result = _upgrade(stack, h.release(_V11, contract_version="1.1.0"))

    assert result.outcome is AgentUpgradeOutcome.UPGRADED
    assert result.previous_installation_id == h.installation_id(_V1)
    assert result.active_installation.installation_id == h.installation_id(_V11)
    assert _slot_states(stack) == {
        h.installation_id(_V1): InstallationState.INSTALLED_PREVIOUS,
        h.installation_id(_V11): InstallationState.INSTALLED_ACTIVE,
    }
    active = stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT)
    assert active is not None and active.previous_installation_ref == h.installation_id(_V1)
    bindings = stack.service.list_bindings(application_id=h.APP, application_environment_id=h.ENV)
    assert bindings.bindings[0].active_installation_id == h.installation_id(_V11)
    assert bindings.bindings[0].enablement is True
    assert result.refreshed_binding_ids == ("bind-search",)
    assert result.audit_event_types == (
        EVENT_INSTALLATION_CREATED,
        EVENT_INSTALLATION_VERIFIED,
        EVENT_INSTALLATION_ACTIVATED,
        "binding.updated",
    )


def test_r011_upgrade_replay_is_idempotent_and_conflicting_reuse_is_refused() -> None:
    stack = _stack_with_v1()
    candidate = h.release(_V11, contract_version="1.1.0")
    first = _upgrade(stack, candidate)
    replay = _upgrade(stack, candidate)
    assert first.outcome is AgentUpgradeOutcome.UPGRADED
    assert replay.outcome is AgentUpgradeOutcome.ALREADY_UPGRADED
    assert replay.audit_event_types == ()

    with pytest.raises(InstallationSlotConflict):
        stack.service.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=candidate.upgrade_request(expected_active="inst-stale"),
            principal=h.principal(),
        )


@pytest.mark.parametrize(
    ("candidate_kwargs", "tool_allowlist", "expected_reason"),
    [
        (
            {"version": "2.0.0", "contract_version": "2.0.0"},
            (),
            AgentUpgradeRejectionReason.PACKAGE_VERSION_INCOMPATIBLE,
        ),
        (
            {"version": "0.9.0", "contract_version": "1.0.0"},
            (),
            AgentUpgradeRejectionReason.NOT_AN_UPGRADE,
        ),
        (
            {"version": _V11, "contract_version": "1.1.0", "capabilities": ()},
            (),
            AgentUpgradeRejectionReason.CAPABILITY_REMOVED,
        ),
        (
            {"version": _V11, "contract_version": "0.9.0"},
            (),
            AgentUpgradeRejectionReason.CONTRACT_VERSION_INCOMPATIBLE,
        ),
        (
            {"version": _V11, "contract_version": "1.1.0", "tools": ("tool.other",)},
            ("tool.search",),
            AgentUpgradeRejectionReason.BOUND_TOOL_REMOVED,
        ),
        (
            {"version": _V11, "contract_version": "1.1.0", "line": "some-other-package"},
            (),
            AgentUpgradeRejectionReason.PACKAGE_LINE_MISMATCH,
        ),
    ],
)
def test_r011_upgrade_rejects_incompatible_candidate_and_keeps_previous_active(
    candidate_kwargs: dict[str, object],
    tool_allowlist: tuple[str, ...],
    expected_reason: AgentUpgradeRejectionReason,
) -> None:
    stack = _stack_with_v1(tool_allowlist=tool_allowlist)
    kwargs = dict(candidate_kwargs)
    version = str(kwargs.pop("version"))
    candidate = h.release(version, **kwargs)  # type: ignore[arg-type]

    result = _upgrade(stack, candidate)

    assert result.outcome is AgentUpgradeOutcome.REJECTED
    assert result.rejection_reason is expected_reason
    assert result.violations and result.violations[0].reason is expected_reason
    # typed rejection is recorded: candidate kept as failed_candidate, one event per transition
    assert result.audit_event_types == (
        EVENT_INSTALLATION_CREATED,
        EVENT_INSTALLATION_VERIFIED,
        EVENT_INSTALLATION_FAILED_CANDIDATE,
    )
    assert _slot_states(stack) == {
        h.installation_id(_V1): InstallationState.INSTALLED_ACTIVE,
        h.installation_id(version): InstallationState.FAILED_CANDIDATE,
    }
    assert result.active_installation.installation_id == h.installation_id(_V1)
    assert result.installation.installation_state is InstallationState.FAILED_CANDIDATE
    active = stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT)
    assert active is not None and active.installation_id == h.installation_id(_V1)
    binding = stack.service.list_bindings(application_id=h.APP, application_environment_id=h.ENV).bindings[0]
    assert binding.active_installation_id == h.installation_id(_V1)
    assert binding.binding_revision == 1  # untouched since enable


def test_r011_upgrade_requires_active_installation_and_is_on_manager_facade() -> None:
    stack = h.build_stack()
    candidate = h.release(_V11, contract_version="1.1.0")
    stack.register(candidate)
    facade = AgentManagerCommandFacade(stack.service)
    with pytest.raises(AgentDistributionNotFoundError):
        facade.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=candidate.upgrade_request(expected_active="inst-none"),
            principal=h.principal(),
        )
    h.install_bind_enable(stack, h.release(_V1))
    result = facade.upgrade_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=candidate.upgrade_request(expected_active=h.installation_id(_V1)),
        principal=h.principal(),
    )
    assert result.outcome is AgentUpgradeOutcome.UPGRADED


def test_r011_upgrade_then_rollback_restores_previous_revision_and_installation() -> None:
    stack = _stack_with_v1()
    first = h.build_revision(stack, "rev-1")
    h.activate_revision(stack, "rev-1")
    assert first.runtime_revision_id == "rev-1"

    assert _upgrade(stack, h.release(_V11, contract_version="1.1.0")).outcome is AgentUpgradeOutcome.UPGRADED
    h.build_revision(stack, "rev-2")
    rev2 = stack.service.inspect_revision(
        application_id=h.APP, application_environment_id=h.ENV, runtime_revision_id="rev-2"
    )
    assert h.DIGESTS[_V11] in rev2.installed_agent_package_digests
    h.activate_revision(stack, "rev-2", expected_prior="rev-1", expected_pointer=1)

    rolled = stack.service.rollback_revision(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=RollbackRuntimeRevisionRequest(
            mutation_id="mut-rollback",
            expected_current_traffic_revision_id="rev-2",
            expected_serving_pointer_revision=2,
        ),
        principal=h.principal(),
    )
    assert rolled.restored_revision_id == "rev-1"
    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    assert serving.traffic_serving_revision_id == "rev-1"
    restored = stack.service.inspect_revision(
        application_id=h.APP, application_environment_id=h.ENV, runtime_revision_id="rev-1"
    )
    assert restored.installed_agent_package_digests == (h.DIGESTS[_V1],)

    # installation-level rollback restores the previous release as the active one
    back = stack.installation_service.rollback_slot_to_previous(
        h.ENV, h.SLOT, expected_active_installation_id=h.installation_id(_V11)
    )
    assert back.value.installation_id == h.installation_id(_V1)
    assert _slot_states(stack) == {
        h.installation_id(_V1): InstallationState.INSTALLED_ACTIVE,
        h.installation_id(_V11): InstallationState.INSTALLED_PREVIOUS,
    }
    assert [event.event_type for event in back.events] == [EVENT_INSTALLATION_ACTIVATED]
    assert back.events[0].attributes["rollback_from"] == h.installation_id(_V11)


# --------------------------------------------------------------------------- R014 retire


def _retire_request(*, expected: str = h.installation_id(_V1), mutation_id: str = "mut-retire") -> RetireAgentRequest:
    return RetireAgentRequest(
        mutation_id=mutation_id,
        installation_slot_id=h.SLOT,
        expected_active_installation_id=expected,
        reason="end of life",
    )


def _retire(stack: h.D1Stack, **kwargs: object):
    return stack.service.retire_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=_retire_request(**kwargs),  # type: ignore[arg-type]
        principal=h.principal(),
    )


def test_r014_retire_removes_routing_but_keeps_installation_and_evidence() -> None:
    stack = _stack_with_v1()
    h.build_revision(stack, "rev-1")
    h.activate_revision(stack, "rev-1")

    result = _retire(stack)

    assert result.lifecycle_state is AgentLifecycleState.RETIRED
    assert result.installation.installation_state is InstallationState.RETIRED
    assert result.affected_logical_agent_ids == (h.CONTRACT_ID,)
    assert result.disabled_binding_ids == ("bind-search",)
    assert result.audit_event_types == ("binding.enablement_changed", EVENT_INSTALLATION_RETIRED)

    # installation and evidence records are kept (no tombstone)
    record = stack.installation_store.get_installation(h.installation_id(_V1))
    assert record is not None
    assert record.installation_state is InstallationState.RETIRED
    assert record.retired_at is not None and record.tombstoned_at is None
    assert record.trust_record is not None and record.artifact_store_ref is not None
    assert stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT) is None

    # routing eligibility is gone: binding disabled, roster entry disabled, status retired
    status = stack.service.inspect_agent_status(
        application_id=h.APP, application_environment_id=h.ENV, logical_agent_id=h.CONTRACT_ID
    )
    assert stack.installation_service.is_slot_retired(h.ENV, h.SLOT)
    assert status.installed is False
    assert status.enabled_in_desired_state is False
    roster = stack.service.inspect_effective_roster(application_id=h.APP, application_environment_id=h.ENV)
    assert [entry.effective_enablement for entry in roster.entries] == [False]

    # the next revision does not include the retired package
    h.build_revision(stack, "rev-2")
    rev2 = stack.service.inspect_revision(
        application_id=h.APP, application_environment_id=h.ENV, runtime_revision_id="rev-2"
    )
    assert h.DIGESTS[_V1] not in rev2.installed_agent_package_digests


def test_r014_retire_lets_inflight_work_finish_under_its_pinned_revision() -> None:
    stack = _stack_with_v1()
    h.build_revision(stack, "rev-1")
    h.activate_revision(stack, "rev-1")
    before = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)

    _retire(stack)

    after = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    assert after.traffic_serving_revision_id == before.traffic_serving_revision_id == "rev-1"
    assert after.serving_pointer_revision == before.serving_pointer_revision
    pinned = stack.service.inspect_revision(
        application_id=h.APP, application_environment_id=h.ENV, runtime_revision_id="rev-1"
    )
    assert pinned.revision_state is RuntimeRevisionState.ACTIVE
    assert pinned.installed_agent_package_digests == (h.DIGESTS[_V1],)


def test_r014_retire_refuses_new_installs_activations_and_rollback_on_the_slot() -> None:
    stack = _stack_with_v1()
    # a verified candidate prepared before the retire must not become active afterwards
    pending = h.release(_V11, contract_version="1.1.0")
    stack.register(pending)
    stack.installation_service.create_candidate_installation(
        installation_id=h.installation_id(_V11),
        installation_slot_id=h.SLOT,
        environment_id=h.ENV,
        package_identity=pending.identity,
    )
    stack.installation_service.mark_verified(
        h.installation_id(_V11),
        artifact_store_ref="store://artifacts/pending",
        trust_record=h.trust(_V11),
    )

    _retire(stack)

    with pytest.raises(InstallationLifecycleError):
        stack.installation_service.promote_verified_to_active(h.installation_id(_V11))
    other = h.release("1.2.0", contract_version="1.2.0")
    stack.register(other)
    with pytest.raises(InstallationLifecycleError):
        stack.service.install_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=other.install_request(),
            principal=h.principal(),
        )
    with pytest.raises(AgentDistributionNotFoundError):
        stack.installation_service.rollback_slot_to_previous(
            h.ENV, h.SLOT, expected_active_installation_id=h.installation_id(_V1)
        )
    with pytest.raises(AgentDistributionNotFoundError):
        stack.service.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=other.upgrade_request(expected_active=h.installation_id(_V1)),
            principal=h.principal(),
        )
    assert stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT) is None


def test_r014_retire_is_idempotent_and_guarded_by_expected_active() -> None:
    stack = _stack_with_v1()
    with pytest.raises(InstallationSlotConflict):
        _retire(stack, expected="inst-stale")
    assert stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT) is not None  # nothing changed

    first = _retire(stack)
    replay = _retire(stack)
    assert first.already_done is False
    assert replay.already_done is True
    assert replay.audit_event_types == ()


def test_r014_retire_is_on_manager_facade_and_scoped_to_the_owning_application() -> None:
    stack = _stack_with_v1()
    with pytest.raises(AgentDistributionNotFoundError):
        stack.service.retire_agent(
            application_id=h.APP_B,
            application_environment_id=h.ENV,
            request=_retire_request(),
            principal=h.principal(),
        )
    facade = AgentManagerCommandFacade(stack.service)
    result = facade.retire_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=_retire_request(),
        principal=h.principal(),
    )
    assert result.installation.installation_state is InstallationState.RETIRED


class _StubAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=h.CONTRACT_ID,
            name="Researcher",
            description="stub",
            capabilities=["search"],
        )

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        raise NotImplementedError

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
        return RuntimeContext.build(config=config, session_manager=build_in_memory_session_manager())


def test_r014_registry_projection_drops_retired_agent_from_routing_but_keeps_contract() -> None:
    registry = AgentRegistry()
    registry.register(_StubAgent())
    assert registry.is_routable(h.CONTRACT_ID)
    assert registry.list_routable_agent_ids() == [h.CONTRACT_ID]

    retired = registry.mark_retired(h.CONTRACT_ID)

    assert retired.lifecycle_state is AgentLifecycleState.RETIRED
    assert registry.mark_retired(h.CONTRACT_ID) is registry.get_contract(h.CONTRACT_ID)  # idempotent
    assert not registry.is_routable(h.CONTRACT_ID)
    assert registry.list_routable_agent_ids() == []
    assert registry.find_by_capability("search") == []
    assert registry.has(h.CONTRACT_ID)  # evidence and in-flight resolution stay possible
    decision = evaluate_agent_routing(registry.get_contract(h.CONTRACT_ID), production_mode=False)
    assert decision.routable is False and "retired" in (decision.reason or "")


# --------------------------------------------------------------------------- R017 evidence


def test_r017_every_installation_state_has_a_canonical_evidence_event() -> None:
    assert set(INSTALLATION_STATE_EVIDENCE_EVENT) == set(InstallationState)
    for event_type in INSTALLATION_STATE_EVIDENCE_EVENT.values():
        assert event_type.startswith("installation.")


def test_r017_each_service_transition_emits_exactly_one_canonical_event() -> None:
    store = InMemoryAgentInstallationStore(AgentDistributionStoreState())
    service = InstallationService(store)
    seen: dict[InstallationState, str] = {}

    def _step(result, state: InstallationState) -> None:
        assert len(result.events) == 1, (state, result.events)
        event = result.events[0]
        assert result.value.installation_state is state
        assert event.event_type == INSTALLATION_STATE_EVIDENCE_EVENT[state]
        seen[state] = event.event_type

    v1, v11, v12 = h.package(_V1), h.package(_V11), h.package("1.2.0")
    _step(
        service.create_candidate_installation(
            installation_id="i1",
            installation_slot_id="s",
            environment_id="e",
            package_identity=v1,
        ),
        InstallationState.CANDIDATE,
    )
    _step(
        service.mark_verified("i1", artifact_store_ref="store://i1", trust_record=h.trust(_V1)),
        InstallationState.VERIFIED,
    )
    _step(service.promote_verified_to_active("i1"), InstallationState.INSTALLED_ACTIVE)

    service.create_candidate_installation(
        installation_id="i2",
        installation_slot_id="s",
        environment_id="e",
        package_identity=v11,
    )
    service.mark_verified("i2", artifact_store_ref="store://i2", trust_record=h.trust(_V11))
    promoted = service.promote_verified_to_active("i2")
    assert promoted.events[0].attributes["prior_active_installation_id"] == "i1"
    seen[InstallationState.INSTALLED_PREVIOUS] = promoted.events[0].event_type
    assert store.get_installation("i1").installation_state is InstallationState.INSTALLED_PREVIOUS  # type: ignore[union-attr]

    _step(service.revoke_installation("i1"), InstallationState.REVOKED)
    _step(service.tombstone_installation("i1"), InstallationState.REMOVED_TOMBSTONE)

    service.create_candidate_installation(
        installation_id="i3",
        installation_slot_id="s",
        environment_id="e",
        package_identity=v12,
    )
    failed = service.mark_failed_candidate("i3", reason_code="contract_removed")
    _step(failed, InstallationState.FAILED_CANDIDATE)
    assert failed.events[0].attributes["reason_code"] == "contract_removed"

    retired = service.retire_installation("i2", expected_active_installation_id="i2", reason="eol")
    _step(retired, InstallationState.RETIRED)
    assert retired.events[0].attributes["reason"] == "eol"

    assert set(seen) == set(InstallationState)


def test_r017_admin_lifecycle_trace_covers_every_state_it_leaves_in_the_store() -> None:
    stack = h.build_stack()
    first = h.release(_V1)
    stack.register(first)
    trace: list[str] = []

    install = stack.service.install_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=first.install_request(),
        principal=h.principal(),
    )
    assert install.audit_event_types == (
        EVENT_INSTALLATION_CREATED,
        EVENT_INSTALLATION_VERIFIED,
        EVENT_INSTALLATION_ACTIVATED,
    )
    trace.extend(install.audit_event_types)
    bound = stack.service.bind_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=h.bind_request(),
        principal=h.principal(),
    )
    trace.extend(bound.audit_event_types)

    upgraded = _upgrade(stack, h.release(_V11, contract_version="1.1.0"))
    trace.extend(upgraded.audit_event_types)
    rejected = _upgrade(
        stack,
        h.release("1.2.0", contract_version="1.2.0", capabilities=()),
        expected=h.installation_id(_V11),
    )
    assert rejected.outcome is AgentUpgradeOutcome.REJECTED
    trace.extend(rejected.audit_event_types)
    retired = _retire(stack, expected=h.installation_id(_V11))
    trace.extend(retired.audit_event_types)

    states_in_store = {
        record.installation_state for record in stack.installation_store.list_installations_for_slot(h.ENV, h.SLOT)
    }
    assert states_in_store == {
        InstallationState.INSTALLED_PREVIOUS,
        InstallationState.RETIRED,
        InstallationState.FAILED_CANDIDATE,
    }
    for state in states_in_store:
        assert INSTALLATION_STATE_EVIDENCE_EVENT[state] in trace
    assert trace.count(EVENT_INSTALLATION_FAILED_CANDIDATE) == 1
    assert trace.count(EVENT_INSTALLATION_RETIRED) == 1
    assert "binding.created" in trace


def test_r017_revision_build_emits_created_and_validated_evidence() -> None:
    stack = _stack_with_v1()
    built = h.build_revision(stack, "rev-1")
    assert built.audit_event_types == (
        EVENT_RUNTIME_REVISION_CREATED,
        EVENT_RUNTIME_REVISION_VALIDATED,
    )
