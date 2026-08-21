# © Artur Czarnecki. All rights reserved.

"""AC-3 production factory contract guards."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from intergrax.utils import attribute_access

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PRODUCTION_FACTORY_MODULES = (
    ("legal_application.host.factory", "create_legal_backend_app"),
    ("research_application.host.factory", "create_research_backend_app"),
    ("local_workspace_application.host.factory", "create_local_workspace_backend_app"),
    ("governed_contractor_application.host.factory", "create_governed_contractor_backend_app"),
    ("dispute_sim_application.host.factory", "create_dispute_sim_backend_app"),
)


def test_production_factories_require_registry_projection_parameter() -> None:
    for module_path, func_name in _PRODUCTION_FACTORY_MODULES:
        module = importlib.import_module(module_path)
        factory = attribute_access.optional(module, func_name)
        signature = inspect.signature(factory)
        assert "registry_projection" in signature.parameters, module_path
        param = signature.parameters["registry_projection"]
        assert param.default is inspect.Parameter.empty, module_path


def test_production_factory_sources_do_not_request_manifest_development() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    forbidden = (
        "RegistryAssemblyMode.MANIFEST_DEVELOPMENT",
        "build_manifest_development_registry",
        "build_lab_registry",
        "build_poc_template_registry",
    )
    for module_path, _ in _PRODUCTION_FACTORY_MODULES:
        source_path = repo_root / "applications" / f"{module_path.replace('.', '/')}.py"
        source = source_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{module_path} contains forbidden {token}"
