# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W1 — public authoring surfaces must not teach Nexus imports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.invocation_pattern import (
    ToolInvocationInvokerPort,
    ToolInvocationPattern,
    ToolInvocationPatternContext,
    ToolInvocationPatternResult,
    ToolInvocationPlannerPort,
)
from tests.qualification.harness_01.public_authoring_nexus_detector import (
    markdown_fenced_python_imports_nexus,
    python_source_imports_nexus,
    scan_all_public_authoring_surfaces,
    scan_public_authoring_path,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_PLUGIN_SRC = (
    _REPO_ROOT
    / "examples"
    / "platform_plugins"
    / "intergrax_reference_enterprise_plugin"
    / "src"
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gate,
    pytest.mark.usefixtures("reference_enterprise_plugin_installed"),
]


def test_harness_01_public_authoring_surfaces_do_not_reference_nexus_as_import_api() -> None:
    hits = scan_all_public_authoring_surfaces(_REPO_ROOT)
    assert hits == [], "public authoring surfaces must not import Nexus:\n" + "\n".join(
        f"{h.relative_path} [{h.kind}] {h.detail}" for h in hits
    )


def test_harness_01_public_authoring_negative_synthetic_nexus_import_fails() -> None:
    synthetic_md = """\
# Bad guide

```python
from intergrax.runtime.nexus.foo import Bar

class Bad:
    pass
```
"""
    assert markdown_fenced_python_imports_nexus(synthetic_md)
    assert python_source_imports_nexus("from intergrax.runtime.nexus.foo import Bar\n")


def test_harness_01_public_authoring_positive_public_abi_import_allowed() -> None:
    synthetic_md = """\
# Good guide

```python
from intergrax.tools.invocation_pattern import ToolInvocationPattern

class Good:
    pass
```

Public extensions must not import `intergrax.runtime.nexus.*`.
Nexus is an internal Execution Engine implementation detail.
"""
    assert markdown_fenced_python_imports_nexus(synthetic_md) == []
    assert not python_source_imports_nexus(
        "from intergrax.tools.invocation_pattern import ToolInvocationPattern\n"
    )


def test_harness_01_reference_enterprise_plugin_is_nexus_free() -> None:
    hits: list[str] = []
    for path in _REFERENCE_PLUGIN_SRC.rglob("*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        file_hits = scan_public_authoring_path(_REPO_ROOT, rel)
        if file_hits:
            hits.extend(f"{h.relative_path}: {h.detail}" for h in file_hits)
    assert hits == []


def test_harness_01_reference_tool_invocation_pattern_public_abi_executable() -> None:
    from intergrax_reference_enterprise_plugin.invocation_pattern import (
        ReferenceEnterpriseSinglePassPattern,
    )

    pattern = ReferenceEnterpriseSinglePassPattern()
    assert isinstance(pattern, ToolInvocationPattern)

    class _Planner:
        def plan_tools(
            self,
            input_data: str | list[object],
            context: object | None = None,
            *,
            run_id: str,
            allowed_tool_ids: Sequence[str] | None = None,
            tool_choice: object | None = None,
        ) -> ToolPlanDecision:
            _ = input_data, context, run_id, allowed_tool_ids, tool_choice
            return ToolPlanDecision(final_answer=None, tool_plan=None, messages=[])

    class _Invoker:
        def invoke_tool(
            self,
            *,
            agent_id: str,
            request: ToolExecutionRequest,
        ) -> ToolExecutionResult:
            _ = agent_id, request
            raise AssertionError("empty pattern must not invoke tools")

    planner: ToolInvocationPlannerPort = _Planner()
    invoker: ToolInvocationInvokerPort = _Invoker()
    result = pattern.execute(
        context=ToolInvocationPatternContext(run_id="w1-proof"),
        invoker=invoker,
        planner=planner,
        plan=None,
        allowed_tool_ids=None,
        max_iterations=1,
        planner_input="noop",
    )
    assert isinstance(result, ToolInvocationPatternResult)
    assert result.pattern_id == "reference_enterprise_single_pass"
    assert result.stop_reason == "empty_tool_calls"
    _ = ToolCallPlan


def test_harness_01_new_agent_scaffold_generates_nexus_free_authoring_package(
    tmp_path: Path,
) -> None:
    """Public golden path: new-agent output must AST-scan Nexus-free and run smoke."""
    import importlib
    import os
    import subprocess
    import sys

    from intergrax.scaffold.new_agent import create_agent
    from tests.qualification.harness_01.public_authoring_nexus_detector import (
        scan_generated_agent_package,
    )

    slug = "w1c1_nexus_free"
    root = tmp_path / "repo"
    root.mkdir()
    (root / "agents").mkdir()
    target = create_agent(
        name=slug,
        capabilities=[f"{slug}.basic"],
        root=root,
        pattern="reflex",
    )
    assert target.is_dir()

    hits = scan_generated_agent_package(target)
    assert hits == [], "generated new-agent package must not import Nexus:\n" + "\n".join(
        f"{h.relative_path} [{h.kind}] {h.detail}" for h in hits
    )

    agent_py = (target / f"{slug}_agent.py").read_text(encoding="utf-8")
    assert "def build_context" not in agent_py
    assert "RuntimeConfig" not in agent_py
    assert "SessionManager" not in agent_py
    assert "RuntimeContext" not in agent_py
    assert "RuntimeRequest" not in agent_py

    agents_root = root / "agents"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(agents_root))
    try:
        module = importlib.import_module(f"{slug}.{slug}_agent")
        class_name = "".join(part.capitalize() for part in slug.split("_")) + "Agent"
        assert getattr(module, class_name) is not None
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
        for mod in list(sys.modules):
            if mod == slug or mod.startswith(f"{slug}."):
                del sys.modules[mod]

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(agents_root), str(root), env.get("PYTHONPATH", "")])
    completed = subprocess.run(
        ["uv", "run", "pytest", str(target / "tests"), "-q"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, (
        "generated agent tests failed:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_harness_01_new_agent_scaffold_inventory_includes_generator() -> None:
    from tests.qualification.harness_01.public_authoring_nexus_detector import (
        PUBLIC_EXTENSION_SCAFFOLD_RELATIVE_PATHS,
        iter_public_authoring_relative_paths,
    )

    assert "intergrax/scaffold/new_agent.py" in PUBLIC_EXTENSION_SCAFFOLD_RELATIVE_PATHS
    assert "intergrax/scaffold/new_agent.py" in iter_public_authoring_relative_paths(_REPO_ROOT)
