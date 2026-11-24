# © Artur Czarnecki. All rights reserved.

"""GCA hosting-boundary component (Package B strict production)."""

from __future__ import annotations

from datetime import datetime

from intergrax.hosting import (
    HostedApplicationComponentHealth,
    HostedApplicationComponentState,
    HostedApplicationContext,
)

GOVERNED_CONTRACTOR_HOSTING_BOUNDARY_COMPONENT_ID = "governed_contractor_hosting_boundary"

GOVERNED_CONTRACTOR_HOSTING_BOUNDARY_COMPONENT_TYPE_ID = "governed_contractor_application.hosting.boundary_component.v1"

GOVERNED_CONTRACTOR_BEFORE_READY_HOOK_ID = "gca_hosted_before_ready_boundary"

GOVERNED_CONTRACTOR_BEFORE_READY_HANDLER_ID = "governed_contractor_application.hosting.before_ready_boundary.v1"

GOVERNED_CONTRACTOR_HOSTING_SOURCE_ID = "governed_contractor"

_EXPECTED_APPLICATION_ID = "governed_contractor"


class _GovernedContractorHostingBoundary:
    """Required GCA product component proving before_ready hook completion."""

    def __init__(self) -> None:
        self._state = HostedApplicationComponentState.CREATED
        self._started = False
        self._before_ready_verified = False
        self._last_transition_at: datetime | None = None

    @property
    def component_id(self) -> str:
        return GOVERNED_CONTRACTOR_HOSTING_BOUNDARY_COMPONENT_ID

    def _validate_application_id(self, context: HostedApplicationContext) -> None:
        if context.application_id != _EXPECTED_APPLICATION_ID:
            raise RuntimeError(
                "governed contractor hosting boundary received unexpected application id",
            )

    async def start(
        self,
        context: HostedApplicationContext,
    ) -> None:
        self._validate_application_id(context)
        if self._started:
            raise RuntimeError("governed contractor hosting boundary already started")
        self._started = True
        self._before_ready_verified = False
        self._state = HostedApplicationComponentState.STARTING
        self._last_transition_at = context.clock.now()

    async def mark_before_ready(
        self,
        context: HostedApplicationContext,
    ) -> None:
        self._validate_application_id(context)
        if not self._started:
            raise RuntimeError("governed contractor hosting boundary is not started")
        self._before_ready_verified = True
        self._state = HostedApplicationComponentState.READY
        self._last_transition_at = context.clock.now()

    async def health(
        self,
        context: HostedApplicationContext,
    ) -> HostedApplicationComponentHealth:
        checked_at = context.clock.now()
        if self._state is HostedApplicationComponentState.CREATED:
            return HostedApplicationComponentHealth(
                component_id=self.component_id,
                enabled=True,
                required=True,
                state=HostedApplicationComponentState.CREATED,
                healthy=False,
                ready=False,
                detail_code="not_started",
                safe_message="hosted boundary not started",
                last_transition_at=self._last_transition_at,
                last_check_at=checked_at,
            )
        if self._state is HostedApplicationComponentState.STOPPED:
            return HostedApplicationComponentHealth(
                component_id=self.component_id,
                enabled=True,
                required=True,
                state=HostedApplicationComponentState.STOPPED,
                healthy=True,
                ready=False,
                detail_code="stopped",
                safe_message="hosted boundary stopped",
                last_transition_at=self._last_transition_at,
                last_check_at=checked_at,
            )
        if self._state is HostedApplicationComponentState.READY and self._before_ready_verified:
            return HostedApplicationComponentHealth(
                component_id=self.component_id,
                enabled=True,
                required=True,
                state=HostedApplicationComponentState.READY,
                healthy=True,
                ready=True,
                detail_code="hosted_boundary_ready",
                safe_message="before_ready hook completed",
                last_transition_at=self._last_transition_at,
                last_check_at=checked_at,
            )
        return HostedApplicationComponentHealth(
            component_id=self.component_id,
            enabled=True,
            required=True,
            state=HostedApplicationComponentState.STARTING,
            healthy=True,
            ready=False,
            detail_code="waiting_before_ready",
            safe_message="waiting for before_ready hook",
            last_transition_at=self._last_transition_at,
            last_check_at=checked_at,
        )

    async def stop(
        self,
        context: HostedApplicationContext,
    ) -> None:
        self._validate_application_id(context)
        self._started = False
        self._before_ready_verified = False
        self._state = HostedApplicationComponentState.STOPPED
        self._last_transition_at = context.clock.now()
