# © Artur Czarnecki. All rights reserved.

"""SCENARIO-PLATFORM-5/6A architecture gates for canonical scaffold."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    execute_scenario_task,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    INVESTIGATOR_CAPABILITY,
)
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    prepare_incident_execution_runtime,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    execute_resolved_skeleton,
    STANDALONE_SCENARIO_TENANT_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_runtime_bundle,
)
from scripts.proof.scenario_architecture_conformance import (
    assert_scenario_application_architecture,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _scenario_root() -> Path:
    return _repo_root() / "platform_proofs" / "scenarios" / "ai_incident_investigation"


_FORBIDDEN_TOOL_CALL_ID_REPAIR_SYMBOLS = frozenset(
    {
        "_IncidentToolCallIdAdapter",
    }
)


def _collect_ast_violations(path: Path, forbidden: frozenset[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    rel = path.relative_to(_repo_root()).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            violations.append(f"{rel}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            violations.append(f"{rel}:{node.lineno} references .{node.attr}")
    return violations


def test_application_must_satisfy_universal_architecture_rules() -> None:
    assert_scenario_application_architecture(
        repo_root=_repo_root(),
        scenario_slug="ai_incident_investigation",
    )


def test_application_must_not_define_tool_call_id_repair_wrapper() -> None:
    violations: list[str] = []
    app_dir = _scenario_root() / "application"
    for path in sorted(app_dir.rglob("*.py")):
        violations.extend(
            _collect_ast_violations(path, _FORBIDDEN_TOOL_CALL_ID_REPAIR_SYMBOLS)
        )
    assert violations == []


@pytest.mark.asyncio
async def test_platform_execution_persists_terminal_runtime_event() -> None:
    bundle = build_runtime_bundle()
    composition = bundle.runtime_composition
    platform = composition.platform
    assert platform.has_runtime_event_store

    prepare_incident_execution_runtime(composition)
    platform_result = await execute_scenario_task(
        platform,
        ScenarioExecutionRequest(
            tenant_id=STANDALONE_SCENARIO_TENANT_ID,
            message="Investigate Line 4 target attainment degradation",
            capability=INVESTIGATOR_CAPABILITY,
        ),
    )
    assert platform_result.task_result.state.value == "completed"

    store = platform.observability.runtime_event_store
    assert store is not None
    events = store.list_for_task(platform_result.task_id, tenant_id=STANDALONE_SCENARIO_TENANT_ID)
    assert events
    assert any(event.event_type == RuntimeEventType.TASK_COMPLETED for event in events)


@pytest.mark.asyncio
async def test_tenant_invariant_standalone() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.execution_tenant_id == STANDALONE_SCENARIO_TENANT_ID
    assert bundle.runtime_composition.platform.tenant_id == STANDALONE_SCENARIO_TENANT_ID
