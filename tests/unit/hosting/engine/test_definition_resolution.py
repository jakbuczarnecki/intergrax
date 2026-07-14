# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import re

import pytest

from intergrax.hosting import (
    HostedApplicationComponentRegistration,
    HostedApplicationProfile,
    resolve_hosted_application_definition,
)
from intergrax.hosting.contracts.policies import ComponentFailureAction
from intergrax.hosting.errors import HostedApplicationConfigurationError, HostedApplicationDefinitionError
from tests.unit.hosting.engine._fakes import FakeComponent, component_registration
from tests.unit.hosting.test_hosted_application_profile_core import sample_application_factory

pytestmark = pytest.mark.unit

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_empty_profile_definition() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
    )
    definition = resolve_hosted_application_definition(profile)
    assert definition.component_start_order == ()
    assert definition.pre_runtime_component_ids == ()


def test_deterministic_definition_digest() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
    )
    first = resolve_hosted_application_definition(profile)
    second = resolve_hosted_application_definition(profile)
    assert first.definition_digest == second.definition_digest
    assert _DIGEST_PATTERN.match(first.definition_digest)


def test_missing_dependency_rejected() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        components=(
            HostedApplicationComponentRegistration(
                component=FakeComponent("worker"),
                dependencies=("missing",),
            ),
        ),
    )
    with pytest.raises(HostedApplicationDefinitionError, match="missing component dependency"):
        resolve_hosted_application_definition(profile)


def test_disabled_dependency_rejected() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        components=(
            HostedApplicationComponentRegistration(
                component=FakeComponent("cache"),
                component_id="cache",
                enabled=False,
            ),
            HostedApplicationComponentRegistration(
                component=FakeComponent("worker"),
                component_id="worker",
                dependencies=("cache",),
            ),
        ),
    )
    with pytest.raises(HostedApplicationDefinitionError, match="depends on disabled"):
        resolve_hosted_application_definition(profile)


def test_cycle_rejected_with_stable_path() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        components=(
            component_registration(FakeComponent("a"), dependencies=("b",)),
            component_registration(FakeComponent("b"), dependencies=("a",)),
        ),
    )
    with pytest.raises(HostedApplicationDefinitionError, match="cycle detected"):
        resolve_hosted_application_definition(profile)


def test_pre_runtime_required_closure() -> None:
    optional = FakeComponent("optional")
    required = FakeComponent("required")
    dependency = FakeComponent("dependency")
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        components=(
            component_registration(optional, required=False),
            component_registration(required, required=True, dependencies=("dependency",)),
            component_registration(dependency, required=False),
        ),
    )
    definition = resolve_hosted_application_definition(profile)
    assert set(definition.pre_runtime_component_ids) == {"dependency", "required"}
    assert definition.post_runtime_component_ids == ("optional",)


def test_unsupported_component_failure_action_rejected() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
        components=(
            component_registration(
                FakeComponent("worker"),
                failure_action=ComponentFailureAction.RESTART_COMPONENT,
            ),
        ),
    )
    with pytest.raises(HostedApplicationConfigurationError, match="not supported in W2"):
        resolve_hosted_application_definition(profile)


def test_public_view_excludes_runtime_refs() -> None:
    profile = HostedApplicationProfile(
        application_id="test_app",
        application_factory=sample_application_factory,
    )
    definition = resolve_hosted_application_definition(profile)
    payload = definition.public_view().model_dump(mode="json")
    assert "application_factory" not in payload
    assert "handler" not in str(payload)
