# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import pytest

from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.new_agent import create_agent
from intergrax.scaffold.new_application import create_application

pytestmark = pytest.mark.gate


def test_agent_scaffold_emits_tracing_extension(tmp_path: Path) -> None:
    slug = "trace_agent"
    create_agent(name=slug, capabilities=[f"{slug}.basic"], root=tmp_path, force=True)
    agent_dir = tmp_path / "agents" / slug
    assert (agent_dir / "tracing" / "example_diag.py").is_file()
    assert (agent_dir / "tracing" / "registry.py").is_file()
    text = (agent_dir / "tracing" / "example_diag.py").read_text(encoding="utf-8")
    assert f'agent_diagnostic_schema_id("{slug}", "custom_check")' in text


def test_application_scaffold_emits_tracing_extension(tmp_path: Path) -> None:
    slug = "trace_app"
    create_agent(name=slug, capabilities=[f"{slug}.basic"], root=tmp_path, force=True)
    create_application(
        name=slug,
        agents=[slug],
        profile="lab",
        root=tmp_path,
        force=True,
        minimal=True,
    )
    names = ScaffoldApplicationNames.resolve(slug)
    tracing = tmp_path / "applications" / names.pkg / "tracing" / "example_diag.py"
    assert tracing.is_file()
    content = tracing.read_text(encoding="utf-8")
    assert f'application_diagnostic_schema_id("{names.short}", "host_lifecycle")' in content


def test_scaffolded_agent_tracing_registry_importable(tmp_path: Path) -> None:
    slug = "trace_reg"
    create_agent(name=slug, capabilities=[f"{slug}.basic"], root=tmp_path, force=True)
    agents_root = tmp_path / "agents"
    sys.path.insert(0, str(tmp_path))
    sys.path.insert(0, str(agents_root))
    try:
        registry = importlib.import_module(f"{slug}.tracing.registry")
        registry.register_tracing_schemas()
        diag = importlib.import_module(f"{slug}.tracing.example_diag")
        from intergrax.runtime.observability.extension_sdk import get_registered_diagnostic_payload

        assert (
            get_registered_diagnostic_payload(diag.CustomCheckDiagV1.schema_id())
            is diag.CustomCheckDiagV1
        )
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
        for mod in list(sys.modules):
            if mod == slug or mod.startswith(f"{slug}."):
                del sys.modules[mod]
