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


def test_agent_scaffold_emits_domain_signal_extension(tmp_path: Path) -> None:
    slug = "signal_agent"
    create_agent(name=slug, capabilities=[f"{slug}.basic"], root=tmp_path, force=True)
    agent_dir = tmp_path / "agents" / slug
    assert (agent_dir / "signals" / "example_signal.py").is_file()
    assert (agent_dir / "signals" / "emit.py").is_file()
    text = (agent_dir / "signals" / "example_signal.py").read_text(encoding="utf-8")
    assert f'agent_signal_schema_id("{slug}", "milestone_reached")' in text


def test_application_scaffold_emits_domain_signal_extension(tmp_path: Path) -> None:
    slug = "signal_app"
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
    signals = tmp_path / "applications" / names.pkg / "signals" / "example_signal.py"
    assert signals.is_file()
    content = signals.read_text(encoding="utf-8")
    assert f'application_signal_schema_id("{names.short}", "host_ready")' in content


def test_scaffolded_agent_signal_emit_importable(tmp_path: Path) -> None:
    slug = "signal_reg"
    create_agent(name=slug, capabilities=[f"{slug}.basic"], root=tmp_path, force=True)
    agents_root = tmp_path / "agents"
    sys.path.insert(0, str(tmp_path))
    sys.path.insert(0, str(agents_root))
    try:
        registry = importlib.import_module(f"{slug}.signals.registry")
        registry.register_signal_schemas()
        emit_mod = importlib.import_module(f"{slug}.signals.emit")
        from intergrax.runtime.events.emit_context import EmitContext
        from intergrax.runtime.events.event_bus import RuntimeEventBus
        from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
        from intergrax.runtime.events.runtime_event import RuntimeEventType

        clear_event_kind_registry()
        registry.register_signal_schemas()
        bus = RuntimeEventBus(record_history=True)
        ctx = EmitContext(task_id="t1", run_id="r1", tenant_id="tenant-a", bus=bus)
        event = emit_mod.emit_milestone_reached(ctx, milestone="boot")
        assert event.event_type == RuntimeEventType.DOMAIN_SIGNAL
        assert event.event_kind == f"agents.{slug}.milestone_reached"
    finally:
        clear_event_kind_registry()
        sys.path.pop(0)
        sys.path.pop(0)
        for mod in list(sys.modules):
            if mod == slug or mod.startswith(f"{slug}."):
                del sys.modules[mod]
