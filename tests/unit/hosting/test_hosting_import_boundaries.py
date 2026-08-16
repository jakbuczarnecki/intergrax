# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

import pytest

import intergrax.hosting as hosting

pytestmark = pytest.mark.unit

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_EXPECTED_PUBLIC_NAMES = (
    "HOSTED_APPLICATION_PROFILE_SPEC_VERSION",
    "HostedApplicationIdentity",
    "HostedApplicationProfile",
    "HostedApplicationProfilePublicView",
    "HostedApplicationContext",
    "HostedApplicationContextPublicView",
    "HostedApplicationPaths",
    "HostedApplicationProcessIdentity",
    "HostedApplicationLifecycleState",
    "HostedApplicationLifecycleSnapshot",
    "HostedApplicationLifecycleSnapshotProvider",
    "HostedApplicationServiceRegistry",
    "HostedApplicationHook",
    "HostedApplicationHooks",
    "HostedApplicationHookPoint",
    "HostedApplicationHookMode",
    "HostedApplicationComponent",
    "HostedApplicationComponentRegistration",
    "HostedApplicationComponentHealth",
    "HostedApplicationComponentState",
    "LifecyclePolicy",
    "ShutdownPolicy",
    "RestartPolicy",
    "ComponentFailurePolicy",
    "HookFailurePolicy",
    "InstancePolicy",
    "HostedApplicationEvent",
    "HostedApplicationEventType",
    "HostedApplicationEventSubscription",
    "HostedApplicationClock",
    "HostedApplicationLogger",
    "HostedApplicationEventPublisher",
    "HostedApplicationShutdownCoordinator",
    "HostedApplicationShutdownRequestSnapshot",
    "HostedApplicationEngine",
    "HostedApplicationRuntime",
    "HostedApplicationDefinition",
    "resolve_hosted_application_definition",
    "ObservabilityHostedApplicationEventPublisher",
)


def test_public_package_exports_documented_names() -> None:
    for name in _EXPECTED_PUBLIC_NAMES:
        assert hasattr(hosting, name), name
        assert name in hosting.__all__


def test_runtime_callables_absent_from_profile_schema() -> None:
    from intergrax.hosting import HostedApplicationProfile

    schema = HostedApplicationProfile.model_json_schema()
    properties = schema.get("properties", {})
    assert "application_factory" not in properties
    assert "handler" not in properties
    assert "component" not in properties


def test_contracts_import_boundary() -> None:
    forbidden_fragments = (
        "fastapi",
        "uvicorn",
        "intergrax.runtime.nexus",
        "intergrax.runtime.task",
        "intergrax.agents",
        "local_workspace_application",
        "nexus_loop",
    )
    package = importlib.import_module("intergrax.hosting")
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        source = inspect.getsource(module)
        lowered = source.lower()
        for fragment in forbidden_fragments:
            assert fragment not in lowered, f"{module_info.name} imports forbidden fragment {fragment}"


def test_no_global_service_registry_exists() -> None:
    from intergrax.hosting import services as services_module

    source = inspect.getsource(services_module)
    assert "GLOBAL" not in source
    assert "singleton" not in source.lower()


def test_no_private_hosting_event_bus_exists() -> None:
    package = importlib.import_module("intergrax.hosting")
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        assert "HostingEventBus" not in inspect.getsource(module)
