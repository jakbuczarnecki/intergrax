# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_FRAGMENTS = (
    "fastapi",
    "uvicorn",
    "intergrax.runtime.nexus",
    "intergrax.runtime.task",
    "intergrax.agents",
    "local_workspace_application",
    "nexus_loop",
)

_RUNTIME_SPINE_ALLOWED = (
    "intergrax.hosting.eventing",
)


def test_engine_modules_do_not_import_forbidden_packages() -> None:
    package = importlib.import_module("intergrax.hosting.engine")
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        source = inspect.getsource(module).lower()
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source, f"{module_info.name} contains forbidden fragment {fragment}"
        if "intergrax.runtime.events" in source:
            assert module_info.name in _RUNTIME_SPINE_ALLOWED, (
                f"{module_info.name} must not import runtime event spine directly"
            )


def test_public_engine_exports_present() -> None:
    import intergrax.hosting as hosting

    for name in (
        "HostedApplicationEngine",
        "HostedApplicationRuntime",
        "HostedApplicationDefinition",
        "resolve_hosted_application_definition",
        "RuntimeSpineHostedApplicationEventPublisher",
    ):
        assert hasattr(hosting, name)
