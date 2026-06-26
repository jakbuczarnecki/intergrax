# © Artur Czarnecki. All rights reserved.

"""Application operational ownership validation (APP-OPS-2 · §50.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.operational_ownership import (
    ApplicationEscalationContact,
    ApplicationMaintainer,
    ApplicationOperationalOwnership,
    ApplicationOwner,
    EscalationChannel,
)

if TYPE_CHECKING:
    from intergrax.applications.contracts.manifest import ApplicationManifest

DEFAULT_OWNER_TEAM = "Intergrax Platform"
DEFAULT_OWNER_CONTACT = "platform@intergrax.local"
DEFAULT_ON_CALL_ROTATION = "intergrax-platform-oncall"


def standard_product_operational_ownership(
    app_id: str,
    *,
    package_name: str | None = None,
    owner_name: str = "Intergrax Platform",
    owner_team: str = DEFAULT_OWNER_TEAM,
    owner_contact: str = DEFAULT_OWNER_CONTACT,
    maintainer_team: str = DEFAULT_OWNER_TEAM,
    maintainer_contact: str = DEFAULT_OWNER_CONTACT,
    escalation_channel: EscalationChannel = EscalationChannel.SLACK,
    escalation_target: str = "#intergrax-platform-oncall",
    on_call_rotation: str | None = DEFAULT_ON_CALL_ROTATION,
    runbook_ref: str | None = None,
    architecture_ref: str | None = None,
    status_page_component: str | None = None,
) -> ApplicationOperationalOwnership:
    """Build canonical ownership metadata for reference product hosts."""
    pkg = package_name or f"{app_id}_application"
    return ApplicationOperationalOwnership(
        app_id=app_id,
        owner=ApplicationOwner(name=owner_name, team=owner_team, contact=owner_contact),
        maintainer=ApplicationMaintainer(
            team=maintainer_team,
            primary_contact=maintainer_contact,
            repo_path=f"applications/{pkg}/",
        ),
        escalation=ApplicationEscalationContact(
            channel=escalation_channel,
            target=escalation_target,
            severity_routing={
                "sev1": escalation_target,
                "sev2": escalation_target,
                "sev3": maintainer_contact,
            },
        ),
        on_call_rotation=on_call_rotation,
        runbook_ref=runbook_ref or f"applications/{pkg}/BUILD_AND_DEPLOY.md",
        architecture_ref=architecture_ref or f"applications/{pkg}/docs/ARCHITECTURE.md",
        status_page_component=status_page_component,
    )


@dataclass(frozen=True, slots=True)
class ApplicationOwnershipDecision:
    """Outcome of application ownership evaluation."""

    approved: bool
    reasons: tuple[str, ...] = ()


def evaluate_application_ownership(manifest: ApplicationManifest) -> ApplicationOwnershipDecision:
    """Return whether a manifest satisfies product host ownership rules."""
    if manifest.profile is not ApplicationProfile.PRODUCT:
        return ApplicationOwnershipDecision(approved=True)

    violations = check_manifest_operational_ownership(manifest.app_id, manifest)
    return ApplicationOwnershipDecision(approved=not violations, reasons=tuple(violations))


def check_manifest_operational_ownership(
    product_id: str,
    manifest: ApplicationManifest,
) -> list[str]:
    """Validate ``ApplicationOperationalOwnership`` on product manifests."""
    if manifest.profile is not ApplicationProfile.PRODUCT:
        return []

    ownership = manifest.ownership
    if ownership is None:
        return [f"{product_id}: ownership must be declared on PRODUCT manifests"]

    violations: list[str] = []
    if ownership.app_id != manifest.app_id:
        violations.append(
            f"{product_id}: ownership.app_id {ownership.app_id!r} must match manifest.app_id {manifest.app_id!r}"
        )

    if not ownership.owner.name.strip():
        violations.append(f"{product_id}: ownership.owner.name is required")
    if not ownership.owner.team.strip():
        violations.append(f"{product_id}: ownership.owner.team is required")
    if not ownership.owner.contact.strip():
        violations.append(f"{product_id}: ownership.owner.contact is required")

    if not ownership.maintainer.team.strip():
        violations.append(f"{product_id}: ownership.maintainer.team is required")
    if not ownership.maintainer.primary_contact.strip():
        violations.append(f"{product_id}: ownership.maintainer.primary_contact is required")
    if not ownership.maintainer.repo_path.strip():
        violations.append(f"{product_id}: ownership.maintainer.repo_path is required")
    if not ownership.maintainer.repo_path.startswith("applications/"):
        violations.append(f"{product_id}: ownership.maintainer.repo_path must start with applications/")

    if not ownership.escalation.target.strip():
        violations.append(f"{product_id}: ownership.escalation.target is required")

    if not ownership.runbook_ref.strip():
        violations.append(f"{product_id}: ownership.runbook_ref is required")
    if not ownership.architecture_ref.strip():
        violations.append(f"{product_id}: ownership.architecture_ref is required")

    return violations
