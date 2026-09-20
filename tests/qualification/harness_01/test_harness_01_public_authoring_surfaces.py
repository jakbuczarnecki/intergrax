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
