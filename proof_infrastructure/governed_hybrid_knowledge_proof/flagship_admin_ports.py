# © Artur Czarnecki. All rights reserved.

"""Typed proof administration ports for flagship multi-vendor Docker scenarios."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from proof_infrastructure.controlled_change_approval_service.models import (
    ChangeApprovalResponseV1,
)
from proof_infrastructure.controlled_governance_approval_service.models import (
    GovernanceApprovalResponseV1,
    GovernanceApprovalSeedControlV1,
)
from proof_infrastructure.controlled_project_status_service.models import (
    ProjectBlockerStatusV1,
    ProjectBlockerV1,
    ProjectStatusControlUpdateV1,
    ProjectStatusResponseV1,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_BLOCKER_ID,
    ORION_FIXTURE_PROJECT_ID,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.admin_port import (
    ControlledSecurityStatusAdminPort,
    HttpxControlledSecurityStatusAdminPort,
)


class ProjectStatusFixtureIdentityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1, max_length=64)
    readiness_score: int = Field(..., ge=0, le=100)
    updated_at: datetime


class ChangeApprovalFixtureIdentityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    change_id: str = Field(..., min_length=1, max_length=64)
    approved: bool
    updated_at: datetime


class GovernanceApprovalFixtureIdentityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_id: str = Field(..., min_length=1, max_length=64)
    approved: bool
    updated_at: datetime
    valid_from: datetime | None = None
    valid_until: datetime | None = None


@runtime_checkable
class ControlledProjectStatusAdminPort(Protocol):
    def seed_project_status(self) -> ProjectStatusFixtureIdentityV1:
        """Seed canonical ORION project readiness through vendor control plane."""

    def close_readiness_blocker(self) -> ProjectStatusFixtureIdentityV1:
        """Mark readiness blocker closed for admissible deployment evidence."""

    def wait_until_ready(self, *, timeout_seconds: float = 60.0) -> None:
        """Wait until vendor reports healthy readiness."""

    def read_request_count(self) -> int:
        """Return vendor-side live read request count."""

    def reset_read_request_count(self) -> None:
        """Reset vendor-side live read request counter."""

    def read_safe_fixture_identity(self, *, project_id: str) -> ProjectStatusFixtureIdentityV1:
        """Read fixture identity without incrementing live read counters."""


@runtime_checkable
class ControlledChangeApprovalAdminPort(Protocol):
    def seed_change_approval(self) -> ChangeApprovalFixtureIdentityV1:
        """Seed canonical ORION change approval through vendor control plane."""

    def wait_until_ready(self, *, timeout_seconds: float = 60.0) -> None:
        """Wait until vendor reports healthy readiness."""

    def read_request_count(self) -> int:
        """Return vendor-side live read request count."""

    def reset_read_request_count(self) -> None:
        """Reset vendor-side live read request counter."""

    def read_safe_fixture_identity(self, *, change_id: str) -> ChangeApprovalFixtureIdentityV1:
        """Read fixture identity without incrementing live read counters."""


@runtime_checkable
class ControlledGovernanceApprovalAdminPort(Protocol):
    def seed_governance_approval(
        self,
        *,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
    ) -> GovernanceApprovalFixtureIdentityV1:
        """Seed canonical ORION governance approval through vendor control plane."""

    def wait_until_ready(self, *, timeout_seconds: float = 60.0) -> None:
        """Wait until vendor reports healthy readiness."""

    def read_request_count(self) -> int:
        """Return vendor-side live read request count."""

    def reset_read_request_count(self) -> None:
        """Reset vendor-side live read request counter."""

    def read_safe_fixture_identity(
        self,
        *,
        subject_id: str,
    ) -> GovernanceApprovalFixtureIdentityV1:
        """Read fixture identity without incrementing live read counters."""


def _project_identity_from_response(
    response: ProjectStatusResponseV1,
) -> ProjectStatusFixtureIdentityV1:
    return ProjectStatusFixtureIdentityV1(
        project_id=response.project_id,
        readiness_score=response.readiness_score,
        updated_at=response.updated_at,
    )


def _change_identity_from_response(
    response: ChangeApprovalResponseV1,
) -> ChangeApprovalFixtureIdentityV1:
    return ChangeApprovalFixtureIdentityV1(
        change_id=response.change_id,
        approved=response.approved,
        updated_at=response.updated_at,
    )


def _governance_identity_from_response(
    response: GovernanceApprovalResponseV1,
) -> GovernanceApprovalFixtureIdentityV1:
    return GovernanceApprovalFixtureIdentityV1(
        subject_id=response.subject_id,
        approved=response.approved,
        updated_at=response.updated_at,
        valid_from=response.valid_from,
        valid_until=response.valid_until,
    )


@dataclass(frozen=True, slots=True)
class HttpxControlledProjectStatusAdminPort:
    base_url: str
    timeout_seconds: float = 5.0

    def seed_project_status(self) -> ProjectStatusFixtureIdentityV1:
        response = httpx.post(
            f"{self.base_url}/control/seed-orion",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _project_identity_from_response(
            ProjectStatusResponseV1.model_validate(response.json())
        )

    def close_readiness_blocker(self) -> ProjectStatusFixtureIdentityV1:
        update = ProjectStatusControlUpdateV1(
            blockers=[
                ProjectBlockerV1(
                    id=ORION_FIXTURE_BLOCKER_ID,
                    status=ProjectBlockerStatusV1.CLOSED,
                ),
            ],
        )
        response = httpx.put(
            f"{self.base_url}/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
            json=update.model_dump(mode="json"),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _project_identity_from_response(
            ProjectStatusResponseV1.model_validate(response.json())
        )

    def wait_until_ready(self, *, timeout_seconds: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "startup_timeout"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                continue
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        raise RuntimeError(f"project_vendor_unavailable: {last_error}")

    def read_request_count(self) -> int:
        response = httpx.get(
            f"{self.base_url}/control/request-count",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return int(response.json()["read_request_count"])

    def reset_read_request_count(self) -> None:
        response = httpx.post(
            f"{self.base_url}/control/request-count/reset",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def read_safe_fixture_identity(self, *, project_id: str) -> ProjectStatusFixtureIdentityV1:
        response = httpx.get(
            f"{self.base_url}/control/fixture/{project_id}",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _project_identity_from_response(
            ProjectStatusResponseV1.model_validate(response.json())
        )


@dataclass(frozen=True, slots=True)
class HttpxControlledChangeApprovalAdminPort:
    base_url: str
    timeout_seconds: float = 5.0

    def seed_change_approval(self) -> ChangeApprovalFixtureIdentityV1:
        response = httpx.post(
            f"{self.base_url}/control/seed-orion",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _change_identity_from_response(
            ChangeApprovalResponseV1.model_validate(response.json())
        )

    def wait_until_ready(self, *, timeout_seconds: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "startup_timeout"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                continue
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        raise RuntimeError(f"change_vendor_unavailable: {last_error}")

    def read_request_count(self) -> int:
        response = httpx.get(
            f"{self.base_url}/control/request-count",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return int(response.json()["read_request_count"])

    def reset_read_request_count(self) -> None:
        response = httpx.post(
            f"{self.base_url}/control/request-count/reset",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def read_safe_fixture_identity(self, *, change_id: str) -> ChangeApprovalFixtureIdentityV1:
        response = httpx.get(
            f"{self.base_url}/control/fixture/{change_id}",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _change_identity_from_response(
            ChangeApprovalResponseV1.model_validate(response.json())
        )


@dataclass(frozen=True, slots=True)
class HttpxControlledGovernanceApprovalAdminPort:
    base_url: str
    timeout_seconds: float = 5.0

    def seed_governance_approval(
        self,
        *,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
    ) -> GovernanceApprovalFixtureIdentityV1:
        control = GovernanceApprovalSeedControlV1(
            valid_from=valid_from,
            valid_until=valid_until,
        )
        response = httpx.post(
            f"{self.base_url}/control/seed-orion",
            json=control.model_dump(mode="json"),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _governance_identity_from_response(
            GovernanceApprovalResponseV1.model_validate(response.json())
        )

    def wait_until_ready(self, *, timeout_seconds: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "startup_timeout"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                continue
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        raise RuntimeError(f"governance_vendor_unavailable: {last_error}")

    def read_request_count(self) -> int:
        response = httpx.get(
            f"{self.base_url}/control/request-count",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return int(response.json()["read_request_count"])

    def reset_read_request_count(self) -> None:
        response = httpx.post(
            f"{self.base_url}/control/request-count/reset",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def read_safe_fixture_identity(
        self,
        *,
        subject_id: str,
    ) -> GovernanceApprovalFixtureIdentityV1:
        response = httpx.get(
            f"{self.base_url}/control/fixture/{subject_id}",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _governance_identity_from_response(
            GovernanceApprovalResponseV1.model_validate(response.json())
        )


@dataclass(frozen=True, slots=True)
class FlagshipVendorAdminFacadeV1:
    """Typed composite proof administration facade for flagship Docker vendors."""

    project: ControlledProjectStatusAdminPort
    security: ControlledSecurityStatusAdminPort
    change: ControlledChangeApprovalAdminPort
    governance: ControlledGovernanceApprovalAdminPort

    @classmethod
    def from_base_urls(
        cls,
        *,
        project_base_url: str,
        security_base_url: str,
        change_base_url: str,
        governance_base_url: str,
    ) -> FlagshipVendorAdminFacadeV1:
        return cls(
            project=HttpxControlledProjectStatusAdminPort(base_url=project_base_url),
            security=HttpxControlledSecurityStatusAdminPort(base_url=security_base_url),
            change=HttpxControlledChangeApprovalAdminPort(base_url=change_base_url),
            governance=HttpxControlledGovernanceApprovalAdminPort(
                base_url=governance_base_url,
            ),
        )

    def wait_until_all_ready(self, *, timeout_seconds: float = 90.0) -> None:
        self.project.wait_until_ready(timeout_seconds=timeout_seconds)
        self.security.wait_until_ready(timeout_seconds=timeout_seconds)
        self.change.wait_until_ready(timeout_seconds=timeout_seconds)
        self.governance.wait_until_ready(timeout_seconds=timeout_seconds)
