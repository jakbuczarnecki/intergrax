# © Artur Czarnecki. All rights reserved.

"""ME-RB4 — typed marketplace lifecycle handoff boundary."""

from __future__ import annotations

import pytest

from intergrax.contracts.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.marketplace import (
    AgentLifecycleHandoffPayload,
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
    MarketplaceCapabilitySelection,
    MarketplaceLifecycleDomainPayload,
    MarketplaceLifecycleHandoffHandler,
    MarketplaceLifecycleHandoffIntent,
    MarketplaceLifecycleHandoffOutcome,
    MarketplaceLifecycleHandoffReasonCode,
    MarketplaceLifecycleHandoffRequest,
    MarketplaceLifecycleHandoffStatus,
    SkillLifecycleHandoffPayload,
    ToolLifecycleHandoffPayload,
    selection_identity_key,
)
from intergrax.contracts.tools.marketplace_lifecycle_handoff import (
    ToolLifecycleHandoffUnavailableError,
)
from intergrax.marketplace.handoff import (
    AgentMarketplaceLifecycleHandoffHandler,
    LifecycleHandoffResolver,
    MarketplaceLifecycleHandoffService,
    SkillMarketplaceLifecycleHandoffHandler,
    ToolMarketplaceLifecycleHandoffHandler,
)
from intergrax.marketplace.handoff.errors import MarketplaceLifecycleHandlerError

pytestmark = pytest.mark.unit

_SOURCE = CapabilitySourceIdentity(
    source_id="marketplace.test",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _entry(kind: CapabilityKind, logical_id: str) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(
            source=_SOURCE,
            version_label="1.0.0",
            publisher="test",
        ),
        display_label=logical_id,
    )


def _identity_key(kind: CapabilityKind, logical_id: str) -> CapabilityIdentityKey:
    return CapabilityIdentityKey.from_discovery_identity(
        _entry(kind, logical_id).identity,
    )


def _handoff_request(
    kind: CapabilityKind,
    logical_id: str,
    *,
    request_id: str = "req-1",
    mismatch_payload_logical: str | None = None,
) -> MarketplaceLifecycleHandoffRequest:
    entry = _entry(kind, logical_id)
    payload_logical = mismatch_payload_logical or logical_id
    payload_key = _identity_key(kind, payload_logical)
    domain = MarketplaceLifecycleDomainPayload(
        agent=(
            AgentLifecycleHandoffPayload(
                operation_id="op-1",
                application_id="app-1",
                application_environment_id="env-1",
                catalog_entry_id="cat-1",
                capability_identity_key=payload_key,
            )
            if kind is CapabilityKind.AGENT
            else None
        ),
        tool=(
            ToolLifecycleHandoffPayload(
                operation_id="op-1",
                host_profile_id="host-1",
                capability_identity_key=payload_key,
            )
            if kind is CapabilityKind.TOOL
            else None
        ),
        skill=(
            SkillLifecycleHandoffPayload(
                operation_id="op-1",
                host_profile_id="host-1",
                capability_identity_key=payload_key,
            )
            if kind is CapabilityKind.SKILL
            else None
        ),
    )
    return MarketplaceLifecycleHandoffRequest(
        request_id=request_id,
        selection=MarketplaceCapabilitySelection(
            listing_id=f"listing-{logical_id}",
            capability=entry,
        ),
        intent=MarketplaceLifecycleHandoffIntent.REQUEST_LIFECYCLE,
        domain_payload=domain,
        correlation_id="corr-1",
    )


class _RecordingAgentPort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, AgentLifecycleHandoffPayload]] = []

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: AgentLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck:
        self.calls.append((request_id, payload))
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference="agent:ref-1",
        )


class _RecordingToolPort:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: ToolLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck:
        del payload, correlation_id
        self.calls.append(request_id)
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference="tool:ref-1",
        )


class _RecordingSkillPort:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: SkillLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck:
        del payload, correlation_id
        self.calls.append(request_id)
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference="skill:ref-1",
        )


def _service_with_builtin_verticals() -> tuple[MarketplaceLifecycleHandoffService, object, object, object]:
    agent_port = _RecordingAgentPort()
    tool_port = _RecordingToolPort()
    skill_port = _RecordingSkillPort()
    resolver = LifecycleHandoffResolver(
        {
            CapabilityKind.AGENT: AgentMarketplaceLifecycleHandoffHandler(agent_port),
            CapabilityKind.TOOL: ToolMarketplaceLifecycleHandoffHandler(tool_port),
            CapabilityKind.SKILL: SkillMarketplaceLifecycleHandoffHandler(skill_port),
        },
    )
    return MarketplaceLifecycleHandoffService(resolver), agent_port, tool_port, skill_port


def test_marketplace_handoff_routes_agent_tool_skill_through_common_contract() -> None:
    service, agent_port, tool_port, skill_port = _service_with_builtin_verticals()

    agent_outcome = service.handoff(_handoff_request(CapabilityKind.AGENT, "agents.a"))
    tool_outcome = service.handoff(_handoff_request(CapabilityKind.TOOL, "tools.t"))
    skill_outcome = service.handoff(_handoff_request(CapabilityKind.SKILL, "skills.s"))

    assert agent_outcome.status is MarketplaceLifecycleHandoffStatus.ACCEPTED
    assert tool_outcome.status is MarketplaceLifecycleHandoffStatus.ACCEPTED
    assert skill_outcome.status is MarketplaceLifecycleHandoffStatus.ACCEPTED
    assert len(agent_port.calls) == 1
    assert len(tool_port.calls) == 1
    assert len(skill_port.calls) == 1
    assert isinstance(agent_outcome, MarketplaceLifecycleHandoffOutcome)
    assert agent_outcome.domain_reference == "agent:ref-1"


class _CustomToolHandler:
    """External handler — not a subclass of default vertical adapter."""

    @property
    def capability_kind(self) -> CapabilityKind:
        return CapabilityKind.TOOL

    @property
    def domain_authority_id(self) -> str:
        return "custom.tool.authority"

    def handoff(
        self,
        request: MarketplaceLifecycleHandoffRequest,
    ) -> MarketplaceLifecycleHandoffOutcome:
        return MarketplaceLifecycleHandoffOutcome(
            request_id=request.request_id,
            status=MarketplaceLifecycleHandoffStatus.ACCEPTED,
            domain_authority_id=self.domain_authority_id,
            domain_reference="custom:tool",
        )


def test_custom_lifecycle_handoff_handler_plugs_in_without_core_changes() -> None:
    custom = _CustomToolHandler()
    assert isinstance(custom, MarketplaceLifecycleHandoffHandler)
    resolver = LifecycleHandoffResolver({CapabilityKind.TOOL: custom})
    service = MarketplaceLifecycleHandoffService(resolver)
    outcome = service.handoff(_handoff_request(CapabilityKind.TOOL, "tools.custom"))
    assert outcome.domain_authority_id == "custom.tool.authority"
    assert outcome.domain_reference == "custom:tool"


def test_missing_handler_fail_closed() -> None:
    service = MarketplaceLifecycleHandoffService(LifecycleHandoffResolver({}))
    outcome = service.handoff(_handoff_request(CapabilityKind.AGENT, "agents.x"))
    assert outcome.status is MarketplaceLifecycleHandoffStatus.REJECTED
    assert outcome.reason_code is MarketplaceLifecycleHandoffReasonCode.HANDLER_MISSING


def test_identity_mismatch_rejected() -> None:
    service, _, _, _ = _service_with_builtin_verticals()
    outcome = service.handoff(
        _handoff_request(
            CapabilityKind.AGENT,
            "agents.a",
            mismatch_payload_logical="agents.other",
        ),
    )
    assert outcome.status is MarketplaceLifecycleHandoffStatus.REJECTED
    assert outcome.reason_code is MarketplaceLifecycleHandoffReasonCode.IDENTITY_MISMATCH


def test_known_domain_unavailable_maps_to_domain_unavailable() -> None:
    class _UnavailableToolPort:
        def submit_marketplace_lifecycle_handoff(
            self,
            payload: ToolLifecycleHandoffPayload,
            *,
            request_id: str,
            correlation_id: str | None,
        ) -> DomainLifecycleHandoffAck:
            del payload, request_id, correlation_id
            raise ToolLifecycleHandoffUnavailableError("tool authority down")

    resolver = LifecycleHandoffResolver(
        {CapabilityKind.TOOL: ToolMarketplaceLifecycleHandoffHandler(_UnavailableToolPort())},
    )
    service = MarketplaceLifecycleHandoffService(resolver)
    outcome = service.handoff(_handoff_request(CapabilityKind.TOOL, "tools.fail"))
    assert outcome.reason_code is MarketplaceLifecycleHandoffReasonCode.DOMAIN_UNAVAILABLE


def test_unexpected_domain_port_error_propagates() -> None:
    class _BrokenToolPort:
        def submit_marketplace_lifecycle_handoff(
            self,
            payload: ToolLifecycleHandoffPayload,
            *,
            request_id: str,
            correlation_id: str | None,
        ) -> DomainLifecycleHandoffAck:
            del payload, request_id, correlation_id
            raise RuntimeError("programming defect")

    resolver = LifecycleHandoffResolver(
        {CapabilityKind.TOOL: ToolMarketplaceLifecycleHandoffHandler(_BrokenToolPort())},
    )
    service = MarketplaceLifecycleHandoffService(resolver)
    with pytest.raises(RuntimeError, match="programming defect"):
        service.handoff(_handoff_request(CapabilityKind.TOOL, "tools.boom"))


def test_wrong_kind_rejected_at_request_construction() -> None:
    entry = _entry(CapabilityKind.AGENT, "agents.a")
    with pytest.raises(ValueError, match="must match domain payload kind"):
        MarketplaceLifecycleHandoffRequest(
            request_id="req-bad",
            selection=MarketplaceCapabilitySelection(
                listing_id="listing-1",
                capability=entry,
            ),
            intent=MarketplaceLifecycleHandoffIntent.REQUEST_LIFECYCLE,
            domain_payload=MarketplaceLifecycleDomainPayload(
                tool=ToolLifecycleHandoffPayload(
                    operation_id="op-1",
                    host_profile_id="host-1",
                    capability_identity_key=selection_identity_key(
                        MarketplaceCapabilitySelection(
                            listing_id="listing-1",
                            capability=entry,
                        ),
                    ),
                ),
            ),
        )


class _TypedFailingHandler:
    @property
    def capability_kind(self) -> CapabilityKind:
        return CapabilityKind.AGENT

    @property
    def domain_authority_id(self) -> str:
        return "agent_distribution"

    def handoff(
        self,
        request: MarketplaceLifecycleHandoffRequest,
    ) -> MarketplaceLifecycleHandoffOutcome:
        raise MarketplaceLifecycleHandlerError("handler internal error")


def test_known_handler_error_maps_to_handler_failed() -> None:
    service = MarketplaceLifecycleHandoffService(
        LifecycleHandoffResolver({CapabilityKind.AGENT: _TypedFailingHandler()}),
    )
    outcome = service.handoff(_handoff_request(CapabilityKind.AGENT, "agents.boom"))
    assert outcome.reason_code is MarketplaceLifecycleHandoffReasonCode.HANDLER_FAILED


class _ExplodingHandler:
    @property
    def capability_kind(self) -> CapabilityKind:
        return CapabilityKind.AGENT

    @property
    def domain_authority_id(self) -> str:
        return "agent_distribution"

    def handoff(
        self,
        request: MarketplaceLifecycleHandoffRequest,
    ) -> MarketplaceLifecycleHandoffOutcome:
        raise TypeError("unexpected handler defect")


def test_unexpected_handler_error_propagates() -> None:
    service = MarketplaceLifecycleHandoffService(
        LifecycleHandoffResolver({CapabilityKind.AGENT: _ExplodingHandler()}),
    )
    with pytest.raises(TypeError, match="unexpected handler defect"):
        service.handoff(_handoff_request(CapabilityKind.AGENT, "agents.boom"))
