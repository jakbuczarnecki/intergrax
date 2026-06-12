# © Artur Czarnecki. All rights reserved.

"""Application host operational ownership (APP-OPS-2 · architecture §50.2)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EscalationChannel(StrEnum):
    """Supported escalation channels for application incidents."""

    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"


class ApplicationOwner(BaseModel):
    """Business/accountable party for a Tier-3 application host."""

    model_config = ConfigDict(extra="forbid")

    name: str
    team: str
    contact: str


class ApplicationMaintainer(BaseModel):
    """Engineering team shipping and operating the host."""

    model_config = ConfigDict(extra="forbid")

    team: str
    primary_contact: str
    repo_path: str


class ApplicationEscalationContact(BaseModel):
    """Incident escalation routing for the application host."""

    model_config = ConfigDict(extra="forbid")

    channel: EscalationChannel
    target: str
    severity_routing: dict[str, str] = Field(default_factory=dict)


class ApplicationOperationalOwnership(BaseModel):
    """
    Operational ownership contract for a deployable application environment.

    Lives on :class:`~intergrax.applications.contracts.manifest.ApplicationManifest`.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: str
    owner: ApplicationOwner
    maintainer: ApplicationMaintainer
    escalation: ApplicationEscalationContact
    on_call_rotation: str | None = None
    runbook_ref: str
    architecture_ref: str
    status_page_component: str | None = None
