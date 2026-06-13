# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent
from intergrax.runtime.sandbox.sandbox_runtime import requires_sandbox_tool
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.providers.codecraft.bundle import register_codecraft_tools
from intergrax.tools.providers.codecraft.contracts import CodeCraftRunToolInput
from intergrax.tools.providers.codecraft.service import codecraft_run
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.codecraft.trace import (
    CODECRAFT_STEP_DISPOSED,
    CODECRAFT_STEP_EXEC,
    CODECRAFT_STEP_SESSION_OPENED,
    CODECRAFT_STEP_STATIC_GATE,
    CodeCraftTraceEmitter,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sandbox_session(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id="tenant-1",
        task_id="task-1",
        allowed_operations=frozenset(
            {"echo", "write_file", "read_file", "list_files", "run_python", "run_script", "browser_fetch"}
        ),
    )


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_codecraft_run_denied_without_profile() -> None:
    out = codecraft_run(
        ToolWiringContext(),
        CodeCraftRunToolInput(code="print(1)"),
    )
    assert out.result.success is False
    assert out.result.error == "codecraft_profile_missing"


def test_codecraft_run_denied_when_mode_disabled() -> None:
    profile = CodeCraftProfile(mode="disabled")
    ctx = ToolWiringContext(extras={"codecraft_profile": profile})
    out = codecraft_run(ctx, CodeCraftRunToolInput(code="print(1)"))
    assert out.result.success is False
    assert out.result.error == "codecraft_mode_disabled"


def test_codecraft_run_static_gate_blocks_forbidden_import() -> None:
    profile = CodeCraftProfile(mode="autonomous")
    ctx = ToolWiringContext(extras={"codecraft_profile": profile})
    out = codecraft_run(
        ctx,
        CodeCraftRunToolInput(code="import os\nprint('nope')\n"),
    )
    assert out.result.success is False
    assert "forbidden_import" in out.result.static_gate.rule_ids
    assert out.trace_event_count >= 3


def test_codecraft_run_denied_without_sandbox_in_autonomous_mode() -> None:
    profile = CodeCraftProfile(mode="autonomous")
    ctx = ToolWiringContext(extras={"codecraft_profile": profile})
    out = codecraft_run(ctx, CodeCraftRunToolInput(code="print('ok')\n"))
    assert out.result.success is False
    assert out.result.error == "sandbox_session_not_configured"


def test_codecraft_run_executes_in_sandbox(sandbox_session: SandboxSession) -> None:
    profile = CodeCraftProfile(mode="autonomous", forbidden_imports=["os", "subprocess"])
    ctx = ToolWiringContext(sandbox_session=sandbox_session, extras={"codecraft_profile": profile})
    out = codecraft_run(
        ctx,
        CodeCraftRunToolInput(code="print('crafted')\n", tenant_id="tenant-1", task_id="task-1"),
    )
    assert out.result.success is True
    assert "crafted" in out.result.stdout
    assert out.result.sandbox_session_id == sandbox_session.session_id
    assert out.result.verdict == "promote"
    assert out.trace_event_count >= 4


def test_codecraft_run_cloud_tier_uses_sandbox_resolver_fallback(sandbox_session: SandboxSession) -> None:
    profile = CodeCraftProfile(
        mode="autonomous",
        isolation_tier="cloud",
        forbidden_imports=["os", "subprocess"],
    )
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        extras={"codecraft_profile": profile},
    )
    out = codecraft_run(
        ctx,
        CodeCraftRunToolInput(code="print('cloud-fallback')\n", tenant_id="tenant-1", task_id="task-1"),
    )
    assert out.result.success is True
    assert "cloud-fallback" in out.result.stdout


def test_codecraft_dry_run_skips_exec(sandbox_session: SandboxSession) -> None:
    profile = CodeCraftProfile(mode="dry_run")
    ctx = ToolWiringContext(sandbox_session=sandbox_session, extras={"codecraft_profile": profile})
    out = codecraft_run(ctx, CodeCraftRunToolInput(code="print('skip me')\n"))
    assert out.result.success is True
    assert out.result.stdout == ""


def test_codecraft_assist_only_returns_code() -> None:
    profile = CodeCraftProfile(mode="assist_only")
    ctx = ToolWiringContext(extras={"codecraft_profile": profile})
    code = "print('helper')\n"
    out = codecraft_run(ctx, CodeCraftRunToolInput(code=code))
    assert out.result.success is True
    assert out.result.structured_output.get("code") == code


def test_codecraft_tool_registered_in_catalog() -> None:
    register_default_tools()
    assert "codecraft.run" in list_catalog_tool_ids()
    bundle = get_bundle("codecraft")
    assert bundle is not None
    assert "codecraft.run" in bundle.tool_ids
    assert "codecraft.start" in bundle.tool_ids


def test_requires_sandbox_tool_includes_codecraft_run() -> None:
    assert requires_sandbox_tool("codecraft.run") is True


def test_build_registry_enables_codecraft_tool(sandbox_session: SandboxSession) -> None:
    register_default_tools()
    profile = CodeCraftProfile(mode="autonomous")
    ctx = ToolWiringContext(
        sandbox_session=sandbox_session,
        extras={"codecraft_profile": profile},
    )
    registry = build_registry_from_profile(ToolProfile(enabled=["codecraft.run"]), ctx=ctx)
    assert registry.has("codecraft.run")


def test_trace_emitter_uses_codecraft_component() -> None:
    emitter = CodeCraftTraceEmitter(run_id="run-1")
    evt = emitter.session_opened(
        craft_id="craft-1",
        mode="autonomous",
        tenant_id="t",
        task_id="task",
    )
    assert evt.component is TraceComponent.CODECRAFT
    assert evt.step == CODECRAFT_STEP_SESSION_OPENED
    gate_evt = emitter.static_gate(
        craft_id="craft-1",
        mode="autonomous",
        passed=False,
        rule_ids=("forbidden_import",),
        tenant_id="t",
        task_id="task",
    )
    assert gate_evt.step == CODECRAFT_STEP_STATIC_GATE
    exec_evt = emitter.exec_completed(
        craft_id="craft-1",
        mode="autonomous",
        sandbox_session_id="sbox-1",
        exit_code=0,
        success=True,
        tenant_id="t",
        task_id="task",
    )
    assert exec_evt.step == CODECRAFT_STEP_EXEC
    disposed_evt = emitter.disposed(
        craft_id="craft-1",
        mode="autonomous",
        tenant_id="t",
        task_id="task",
    )
    assert disposed_evt.step == CODECRAFT_STEP_DISPOSED
    assert len(emitter.events) == 4


def test_register_codecraft_tools_on_registry(sandbox_session: SandboxSession) -> None:
    ctx = ToolWiringContext(sandbox_session=sandbox_session)
    registry = ToolRegistry()
    register_codecraft_tools(registry, ctx)
    assert registry.has("codecraft.run")
