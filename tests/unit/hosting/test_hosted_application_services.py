# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.hosting.services import (
    HostedApplicationServiceRegistry,
    HostedApplicationServiceRegistryDuplicateError,
    HostedApplicationServiceRegistryStateError,
)
from tests.unit.hosting._helpers import SampleComponent

pytestmark = pytest.mark.unit


def test_typed_register_require_optional() -> None:
    registry = HostedApplicationServiceRegistry()
    component = SampleComponent()
    registry.register(SampleComponent, component)
    assert registry.contains(SampleComponent) is True
    assert registry.optional(SampleComponent) is component
    assert registry.require(SampleComponent) is component


def test_duplicate_registration_rejected() -> None:
    registry = HostedApplicationServiceRegistry()
    registry.register(SampleComponent, SampleComponent())
    with pytest.raises(HostedApplicationServiceRegistryDuplicateError):
        registry.register(SampleComponent, SampleComponent())


def test_sealed_registry_rejects_registration() -> None:
    registry = HostedApplicationServiceRegistry()
    registry.seal()
    with pytest.raises(HostedApplicationServiceRegistryStateError):
        registry.register(SampleComponent, SampleComponent())


def test_closed_registry_rejects_registration_and_resolution() -> None:
    registry = HostedApplicationServiceRegistry()
    registry.register(SampleComponent, SampleComponent())
    registry.close()
    with pytest.raises(HostedApplicationServiceRegistryStateError):
        registry.require(SampleComponent)
    with pytest.raises(HostedApplicationServiceRegistryStateError):
        registry.register(SampleComponent, SampleComponent())


def test_service_objects_absent_from_diagnostics() -> None:
    registry = HostedApplicationServiceRegistry()
    registry.register(SampleComponent, SampleComponent())
    diagnostics = registry.diagnostic_view()
    assert diagnostics == ("tests.unit.hosting._helpers.SampleComponent",)
    assert repr(SampleComponent()) not in diagnostics
