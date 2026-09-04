# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Collaborative Work intake port and adapters (AW-4C).

Stable AW→Collaborative Work boundary. Canonical WorkItem semantics remain
owned by Collaborative Work MP-2.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.collaborative_work_bridge import (
    CollaborativeWorkRequest,
    CollaborativeWorkRequestIdentity,
    CollaborativeWorkSubmissionDisposition,
    CollaborativeWorkSubmissionResult,
    resolve_collaborative_work_submission_replay,
)
from intergrax.contracts.autonomous_work.references import WorkReference


class CollaborativeWorkIntakeUnavailable(Exception):
    """Canonical Collaborative Work intake is not available — fail closed."""


@runtime_checkable
class CollaborativeWorkIntakePort(Protocol):
    """Smallest stable port for submitting collaborative work requests."""

    def submit(
        self,
        request: CollaborativeWorkRequest,
    ) -> CollaborativeWorkSubmissionResult:
        """Submit one collaborative work request to canonical intake."""
        ...


class UnavailableCollaborativeWorkIntake:
    """Production-safe fail-closed adapter when MP-2 intake is absent."""

    def submit(
        self,
        request: CollaborativeWorkRequest,
    ) -> CollaborativeWorkSubmissionResult:
        return CollaborativeWorkSubmissionResult(
            disposition=CollaborativeWorkSubmissionDisposition.UNAVAILABLE,
            request_identity=request.request_identity,
        )


class RecordingCollaborativeWorkIntake:
    """Reference-only in-memory adapter for bridge unit tests."""

    def __init__(self) -> None:
        self._submissions: dict[str, CollaborativeWorkRequest] = {}
        self._refs: dict[str, WorkReference] = {}

    @property
    def submissions(self) -> tuple[CollaborativeWorkRequest, ...]:
        return tuple(self._submissions.values())

    def submit(
        self,
        request: CollaborativeWorkRequest,
    ) -> CollaborativeWorkSubmissionResult:
        key = request.request_identity.identity_key
        existing = self._submissions.get(key)
        disposition = resolve_collaborative_work_submission_replay(
            existing=existing,
            incoming=request,
        )
        if disposition is CollaborativeWorkSubmissionDisposition.ACCEPTED:
            self._submissions[key] = request
            work_ref = WorkReference(f"work/test/{key}")
            self._refs[key] = work_ref
            return CollaborativeWorkSubmissionResult(
                disposition=CollaborativeWorkSubmissionDisposition.ACCEPTED,
                request_identity=request.request_identity,
                collaborative_work_ref=work_ref,
            )
        if disposition is CollaborativeWorkSubmissionDisposition.ALREADY_EXISTS:
            return CollaborativeWorkSubmissionResult(
                disposition=CollaborativeWorkSubmissionDisposition.ALREADY_EXISTS,
                request_identity=request.request_identity,
                collaborative_work_ref=self._refs.get(key),
            )
        return CollaborativeWorkSubmissionResult(
            disposition=CollaborativeWorkSubmissionDisposition.CONFLICT,
            request_identity=request.request_identity,
            collaborative_work_ref=self._refs.get(key),
        )

    def submission_for(
        self,
        identity: CollaborativeWorkRequestIdentity,
    ) -> CollaborativeWorkRequest | None:
        return self._submissions.get(identity.identity_key)
