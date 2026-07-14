# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from tests.unit.hosting._helpers import SampleComponent, build_sample_context

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
