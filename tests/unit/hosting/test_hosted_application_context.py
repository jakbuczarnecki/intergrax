# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.hosting import HostedApplicationProcessIdentity
from tests.unit.hosting._helpers import SampleComponent, build_minimal_profile, build_sample_context

pytestmark = pytest.mark.unit


def test_safe_context_construction() -> None:
    context = build_sample_context()
    assert context.application_id == "my_application"
    assert context.instance_id == "instance-001"
    assert context.profile.identity.application_id == "my_application"
    assert context.profile_digest.startswith("sha256:")


def test_public_view_redaction() -> None:
    context = build_sample_context()
    public_view = context.public_view()
    dumped = public_view.model_dump()
    assert set(dumped) == {
        "application_id",
        "instance_id",
        "profile_digest",
        "profile_spec_version",
        "process_identity",
        "lifecycle",
        "closed",
    }
    assert "paths" not in dumped
    assert "services" not in dumped


def test_no_global_sharing_between_contexts() -> None:
    left = build_sample_context(instance_id="left")
    right = build_sample_context(instance_id="right")
    assert left.services is not right.services


def test_context_close_marks_closed_and_closes_registry() -> None:
    context = build_sample_context()
    registry = context.services
    registry.register(SampleComponent, SampleComponent())
    context.close()
    assert context.closed is True
    assert registry.is_closed is True


def test_context_field_reassignment_rejected() -> None:
    context = build_sample_context()
    with pytest.raises(FrozenInstanceError):
        context.application_id = "other_application"  # type: ignore[misc]


def test_context_reassignment_rejected_after_close() -> None:
    context = build_sample_context()
    context.close()
    with pytest.raises(FrozenInstanceError):
        context.instance_id = "other-instance"  # type: ignore[misc]


def test_context_repeated_close_is_safe() -> None:
    context = build_sample_context()
    context.close()
    context.close()
    assert context.closed is True


def test_context_rejects_application_profile_identity_mismatch() -> None:
    from intergrax.hosting import (
        HostedApplicationContext,
        HostedApplicationLifecycleSnapshot,
        HostedApplicationLifecycleState,
        HostedApplicationPaths,
        HostedApplicationProcessIdentity,
    )
    from intergrax.hosting.services import HostedApplicationServiceRegistry
    from tests.unit.hosting._helpers import (
        _FixedClock,
        _FixedLifecycle,
        _NoopLogger,
        _NoopPublisher,
        _NoopShutdown,
        build_minimal_profile,
    )

    profile = build_minimal_profile()
    moment = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="application_id must match"):
        HostedApplicationContext(
            application_id="other_application",
            instance_id="instance-001",
            profile=profile.public_view(),
            profile_digest=profile.profile_digest(),
            paths=HostedApplicationPaths(
                data_home=Path("data") / profile.application_id,
                run_directory=Path("data") / profile.application_id / "run",
            ),
            process_identity=HostedApplicationProcessIdentity(
                process_id=1,
                started_at=moment,
            ),
            services=HostedApplicationServiceRegistry(),
            clock=_FixedClock(moment),
            logger=_NoopLogger(),
            event_publisher=_NoopPublisher(),
            shutdown=_NoopShutdown(),
            lifecycle=_FixedLifecycle(
                HostedApplicationLifecycleSnapshot(
                    state=HostedApplicationLifecycleState.READY,
                    accepting_new_work=True,
                    shutdown_requested=False,
                    last_transition_at=moment,
                )
            ),
        )


def test_context_rejects_invalid_profile_digest() -> None:
    profile = build_minimal_profile()
    moment = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="profile_digest"):
        from intergrax.hosting import HostedApplicationContext, HostedApplicationPaths, HostedApplicationProcessIdentity
        from intergrax.hosting import HostedApplicationLifecycleSnapshot, HostedApplicationLifecycleState
        from intergrax.hosting.services import HostedApplicationServiceRegistry
        from tests.unit.hosting._helpers import (
            _FixedClock,
            _FixedLifecycle,
            _NoopLogger,
            _NoopPublisher,
            _NoopShutdown,
        )

        HostedApplicationContext(
            application_id=profile.application_id,
            instance_id="instance-001",
            profile=profile.public_view(),
            profile_digest="sha256:NOT_VALID",
            paths=HostedApplicationPaths(
                data_home=Path("data") / profile.application_id,
                run_directory=Path("data") / profile.application_id / "run",
            ),
            process_identity=HostedApplicationProcessIdentity(
                process_id=1,
                started_at=moment,
            ),
            services=HostedApplicationServiceRegistry(),
            clock=_FixedClock(moment),
            logger=_NoopLogger(),
            event_publisher=_NoopPublisher(),
            shutdown=_NoopShutdown(),
            lifecycle=_FixedLifecycle(
                HostedApplicationLifecycleSnapshot(
                    state=HostedApplicationLifecycleState.READY,
                    accepting_new_work=True,
                    shutdown_requested=False,
                    last_transition_at=moment,
                )
            ),
        )


def test_process_identity_rejects_naive_started_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        HostedApplicationProcessIdentity(
            process_id=1,
            started_at=datetime(2026, 7, 14, 12, 0),
        )
