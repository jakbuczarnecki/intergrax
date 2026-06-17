# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain signal scaffold templates for agent and application extension SDK (OBS-EVOL-9.8)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def write_agent_signal_scaffold(*, target: Path, slug: str, force: bool) -> None:
    _write(target / "signals" / "__init__.py", _agent_signals_init(slug), force=force)
    _write(target / "signals" / "example_signal.py", _agent_example_signal(slug), force=force)
    _write(target / "signals" / "registry.py", _agent_signal_registry_py(slug), force=force)
    _write(target / "signals" / "emit.py", _agent_signal_emit_py(slug), force=force)
    _write(target / "tests" / f"test_{slug}_signals.py", _agent_signal_test(slug), force=force)


def write_application_signal_scaffold(
    *,
    target: Path,
    pkg: str,
    short: str,
    force: bool,
) -> None:
    _write(target / "signals" / "__init__.py", _application_signals_init(pkg), force=force)
    _write(target / "signals" / "example_signal.py", _application_example_signal(short), force=force)
    _write(target / "signals" / "registry.py", _application_signal_registry_py(pkg, short), force=force)
    _write(target / "signals" / "emit.py", _application_signal_emit_py(pkg, short), force=force)


def _write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _agent_signals_init(slug: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Agent domain runtime signals for {slug}."""

        from {slug}.signals.registry import register_signal_schemas

        __all__ = ["register_signal_schemas"]
        '''
    )


def _agent_example_signal(slug: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from pydantic import Field

        from intergrax.runtime.events.payloads.base import RuntimeEventPayload
        from intergrax.runtime.observability.extension_sdk import agent_signal_schema_id


        class MilestoneReachedPayloadV1(RuntimeEventPayload):
            """Example operator-visible domain signal — replace with product semantics."""

            schema_id = agent_signal_schema_id("{slug}", "milestone_reached")
            milestone: str = Field(min_length=1)
            detail: str = ""

            def redact(self) -> MilestoneReachedPayloadV1:
                return self
        '''
    )


def _agent_signal_registry_py(slug: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.runtime.observability.extension_sdk import (
            PayloadSchemaRegistry,
            agent_signal_event_kind,
        )
        from {slug}.signals.example_signal import MilestoneReachedPayloadV1


        def register_signal_schemas() -> None:
            """Register agent domain signal kinds with the Harness event kind registry."""
            PayloadSchemaRegistry.register_runtime_extension(
                MilestoneReachedPayloadV1,
                event_kind=agent_signal_event_kind("{slug}", "milestone_reached"),
            )
        '''
    )


def _agent_signal_emit_py(slug: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.runtime.events.emit_context import EmitContext
        from intergrax.runtime.events.runtime_event import RuntimeEvent
        from intergrax.runtime.events.signals import emit_domain_signal
        from intergrax.runtime.observability.extension_sdk import agent_signal_event_kind
        from {slug}.signals.example_signal import MilestoneReachedPayloadV1


        def emit_milestone_reached(
            ctx: EmitContext,
            *,
            milestone: str,
            detail: str = "",
        ) -> RuntimeEvent:
            """Emit a typed domain signal for operator-visible agent milestones."""
            return emit_domain_signal(
                ctx,
                kind=agent_signal_event_kind("{slug}", "milestone_reached"),
                payload=MilestoneReachedPayloadV1(milestone=milestone, detail=detail),
            )
        '''
    )


def _agent_signal_test(slug: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        import pytest

        from intergrax.runtime.events.emit_context import EmitContext
        from intergrax.runtime.events.event_bus import RuntimeEventBus
        from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
        from intergrax.runtime.events.runtime_event import RuntimeEventType
        from intergrax.runtime.observability.extension_sdk import agent_signal_event_kind
        from {slug}.signals.emit import emit_milestone_reached
        from {slug}.signals.registry import register_signal_schemas

        pytestmark = pytest.mark.gate


        @pytest.fixture(autouse=True)
        def _register_agent_signal_kinds() -> None:
            clear_event_kind_registry()
            register_signal_schemas()
            yield
            clear_event_kind_registry()


        def test_agent_signal_emits_domain_signal() -> None:
            bus = RuntimeEventBus(record_history=True)
            ctx = EmitContext(task_id="task-1", run_id="run-1", tenant_id="tenant-a", bus=bus)
            event = emit_milestone_reached(ctx, milestone="scaffold", detail="smoke")
            kind = agent_signal_event_kind("{slug}", "milestone_reached")
            assert event.event_type == RuntimeEventType.DOMAIN_SIGNAL
            assert event.event_kind == kind
            assert bus.history[-1].event_id == event.event_id
        '''
    )


def _application_signals_init(pkg: str) -> str:
    apps_root = "applications"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Application domain runtime signals for {pkg}."""

        from {apps_root}.{pkg}.signals.registry import register_signal_schemas

        __all__ = ["register_signal_schemas"]
        '''
    )


def _application_example_signal(short: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from pydantic import Field

        from intergrax.runtime.events.payloads.base import RuntimeEventPayload
        from intergrax.runtime.observability.extension_sdk import application_signal_schema_id


        class HostReadyPayloadV1(RuntimeEventPayload):
            """Example operator-visible host signal — replace with product semantics."""

            schema_id = application_signal_schema_id("{short}", "host_ready")
            phase: str = Field(min_length=1)

            def redact(self) -> HostReadyPayloadV1:
                return self
        '''
    )


def _application_signal_registry_py(pkg: str, short: str) -> str:
    apps_root = "applications"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.runtime.observability.extension_sdk import (
            PayloadSchemaRegistry,
            application_signal_event_kind,
        )
        from {apps_root}.{pkg}.signals.example_signal import HostReadyPayloadV1


        def register_signal_schemas() -> None:
            """Register application domain signal kinds with the Harness event kind registry."""
            PayloadSchemaRegistry.register_runtime_extension(
                HostReadyPayloadV1,
                event_kind=application_signal_event_kind("{short}", "host_ready"),
            )
        '''
    )


def _application_signal_emit_py(pkg: str, short: str) -> str:
    apps_root = "applications"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.runtime.events.emit_context import EmitContext
        from intergrax.runtime.events.runtime_event import RuntimeEvent
        from intergrax.runtime.events.signals import emit_domain_signal
        from intergrax.runtime.observability.extension_sdk import application_signal_event_kind
        from {apps_root}.{pkg}.signals.example_signal import HostReadyPayloadV1


        def emit_host_ready(ctx: EmitContext, *, phase: str) -> RuntimeEvent:
            """Emit a typed domain signal when the host reaches a lifecycle milestone."""
            return emit_domain_signal(
                ctx,
                kind=application_signal_event_kind("{short}", "host_ready"),
                payload=HostReadyPayloadV1(phase=phase),
            )
        '''
    )
