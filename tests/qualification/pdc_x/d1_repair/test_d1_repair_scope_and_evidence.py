# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-A / F-D1-05, F-D1-06 and the F-D1-12 / F-D1-14 by-product pins.

F-D1-05: application-ownership guard on install into an occupied slot and on upgrade.
F-D1-06: no leftover VERIFIED candidate after a failed upgrade; optional inert event sink.
F-D1-12: DENY boundary on upgrade/retire (M2), same-version upgrade (M7), reason_code on the
failed-candidate event (M10). F-D1-14: duplicate agent id rejected by the registry.
"""

from __future__ import annotations

import inspect

import pytest
from intergrax.agent_distribution.admin_models import (
    AgentPlatformAdminGovernanceBlockedError,
    AgentUpgradeOutcome,
    RetireAgentRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.errors import AgentDistributionNotFoundError
from intergrax.agent_distribution.events import (
    EVENT_INSTALLATION_ACTIVATED,
    EVENT_INSTALLATION_CREATED,
    EVENT_INSTALLATION_FAILED_CANDIDATE,
    EVENT_INSTALLATION_VERIFIED,
    AgentDistributionEvent,
)
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.upgrade_admission import AgentUpgradeRejectionReason
from tests.qualification.pdc_x.d1 import _d1_harness as h
from tests.qualification.pdc_x.d1_repair import _support as s

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentDistributionEvent] = []

    def __call__(self, events: tuple[AgentDistributionEvent, ...]) -> None:
        self.events.extend(events)

    def types(self) -> list[str]:
        return [event.event_type for event in self.events]


def _stack_with_v1() -> h.D1Stack:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(s.V1))
    return stack


def _upgrade(stack: h.D1Stack, candidate: h.Release, *, application_id: str = h.APP):
    stack.register(candidate)
    return stack.service.upgrade_agent(
        application_id=application_id,
        application_environment_id=h.ENV,
        request=candidate.upgrade_request(expected_active=h.installation_id(s.V1)),
        principal=h.principal(),
    )


def _slot_states(stack: h.D1Stack) -> dict[str, InstallationState]:
    return {record.installation_id: record.installation_state for record in stack.state.installations.values()}


def _with_sink(stack: h.D1Stack, monkeypatch: pytest.MonkeyPatch) -> _RecordingSink:
    sink = _RecordingSink()
    monkeypatch.setattr(stack.service, "_lifecycle_event_sink", sink)
    return sink


# --------------------------------------------------------------------------- F-D1-05 ownership


def test_upgrade_by_an_application_that_does_not_own_the_slot_is_refused_without_mutation() -> None:
    stack = _stack_with_v1()
    before = _slot_states(stack)

    with pytest.raises(AgentDistributionNotFoundError):
        _upgrade(stack, h.release(s.V11, contract_version="1.1.0"), application_id=h.APP_B)

    assert _slot_states(stack) == before
    active = stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT)
    assert active is not None and active.installation_id == h.installation_id(s.V1)


def test_upgrade_by_the_owning_application_still_works() -> None:
    stack = _stack_with_v1()

    result = _upgrade(stack, h.release(s.V11, contract_version="1.1.0"))

    assert result.outcome is AgentUpgradeOutcome.UPGRADED


def test_install_into_a_slot_occupied_by_another_application_is_refused_without_mutation() -> None:
    stack = _stack_with_v1()
    before = _slot_states(stack)
    other = h.release(s.V11, contract_version="1.1.0")
    stack.register(other)

    with pytest.raises(AgentDistributionNotFoundError):
        stack.service.install_agent(
            application_id=h.APP_B,
            application_environment_id=h.ENV,
            request=other.install_request(slot=h.SLOT),
            principal=h.principal(),
        )

    assert _slot_states(stack) == before


def test_install_into_a_fresh_slot_is_unchanged_for_any_application() -> None:
    stack = _stack_with_v1()
    other = h.release(s.V11, contract_version="1.1.0")
    stack.register(other)

    result = stack.service.install_agent(
        application_id=h.APP_B,
        application_environment_id=h.ENV,
        request=other.install_request(slot="slot-fresh"),
        principal=h.principal(),
    )

    assert result.installation.installation_id == h.installation_id(s.V11)


# --------------------------------------------------------------------------- F-D1-06 candidate state


def test_failure_after_verification_never_leaves_a_verified_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _stack_with_v1()

    def _boom(installation: object) -> object:
        raise RuntimeError("resolve-failed")

    monkeypatch.setattr(stack.service, "_resolve_package_contracts", _boom)

    with pytest.raises(RuntimeError, match="resolve-failed"):
        _upgrade(stack, h.release(s.V11, contract_version="1.1.0"))

    states = _slot_states(stack)
    assert states[h.installation_id(s.V11)] is InstallationState.FAILED_CANDIDATE
    assert states[h.installation_id(s.V1)] is InstallationState.INSTALLED_ACTIVE


# --------------------------------------------------------------------------- F-D1-06 event sink


def test_service_accepts_an_optional_event_sink_that_defaults_to_none() -> None:
    parameter = inspect.signature(AgentPlatformAdminService.__init__).parameters["lifecycle_event_sink"]
    assert parameter.default is None


def test_sink_receives_every_install_and_upgrade_transition_including_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = h.build_stack()
    sink = _with_sink(stack, monkeypatch)
    h.install_bind_enable(stack, h.release(s.V1))
    assert sink.types() == [
        EVENT_INSTALLATION_CREATED,
        EVENT_INSTALLATION_VERIFIED,
        EVENT_INSTALLATION_ACTIVATED,
    ]
    sink.events.clear()

    def _boom(installation: object) -> object:
        raise RuntimeError("resolve-failed")

    monkeypatch.setattr(stack.service, "_resolve_package_contracts", _boom)
    with pytest.raises(RuntimeError):
        _upgrade(stack, h.release(s.V11, contract_version="1.1.0"))

    assert EVENT_INSTALLATION_FAILED_CANDIDATE in sink.types()
    failed = next(e for e in sink.events if e.event_type == EVENT_INSTALLATION_FAILED_CANDIDATE)
    assert failed.aggregate_id == h.installation_id(s.V11)
    assert failed.attributes["reason_code"]


def test_rejected_upgrade_event_carries_the_typed_reason_code(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _stack_with_v1()
    sink = _with_sink(stack, monkeypatch)
    sink.events.clear()

    result = _upgrade(stack, h.release(s.V11, contract_version="1.1.0", capabilities=()))

    assert result.outcome is AgentUpgradeOutcome.REJECTED
    failed = next(e for e in sink.events if e.event_type == EVENT_INSTALLATION_FAILED_CANDIDATE)
    assert failed.attributes["reason_code"] == result.rejection_reason.value  # type: ignore[union-attr]


def test_without_a_sink_results_are_unchanged() -> None:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(s.V1))

    result = _upgrade(stack, h.release(s.V11, contract_version="1.1.0"))

    assert result.outcome is AgentUpgradeOutcome.UPGRADED
    assert result.audit_event_types


# --------------------------------------------------------------------------- F-D1-12 by-product pins


def test_deny_boundary_blocks_upgrade_and_retire_without_touching_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack_with_v1()
    before = _slot_states(stack)
    monkeypatch.setattr(stack.service, "_mutation_authorization_boundary", s.deny_mutation_boundary())

    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        _upgrade(stack, h.release(s.V11, contract_version="1.1.0"))
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.retire_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=RetireAgentRequest(
                mutation_id="mut-retire-denied",
                installation_slot_id=h.SLOT,
                expected_active_installation_id=h.installation_id(s.V1),
            ),
            principal=h.principal(),
        )

    assert _slot_states(stack) == before


def test_upgrade_to_the_same_version_is_not_an_upgrade() -> None:
    stack = _stack_with_v1()
    same = h.release(s.V1)
    stack.register(same)
    request = same.upgrade_request(expected_active=h.installation_id(s.V1)).model_copy(
        update={"installation_id": "inst-1.0.0-again", "mutation_id": "mut-upgrade-same-version"}
    )

    result = stack.service.upgrade_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=request,
        principal=h.principal(),
    )

    assert result.outcome is AgentUpgradeOutcome.REJECTED
    assert result.rejection_reason is AgentUpgradeRejectionReason.NOT_AN_UPGRADE


# --------------------------------------------------------------------------- F-D1-14 by-product pin


def test_registry_rejects_a_second_agent_with_the_same_contract_id() -> None:
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from tests.unit.runtime.registry.test_agent_registry import _StubAgent

    registry = AgentRegistry()
    registry.register(_StubAgent())

    with pytest.raises(ValueError, match="Agent already registered"):
        registry.register(_StubAgent())
