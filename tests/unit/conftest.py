# © Artur Czarnecki. All rights reserved.

"""CI smoke scope — lightweight representative unit tests for merge/PR feedback."""

from __future__ import annotations

import pytest

# Whole directories: typed contracts only (pure pydantic / schema).
CI_SMOKE_DIR_PREFIXES: tuple[str, ...] = (
    "tests/unit/contracts/",
    "tests/contract/core/plugins/",
)

# High-signal single modules across harness layers (deterministic, no catalog bootstrap).
CI_SMOKE_FILES: frozenset[str] = frozenset(
    {
        "tests/unit/agents/authoring/test_step_outcome.py",
        "tests/unit/agents/authoring/test_decisions.py",
        "tests/unit/agents/authoring/test_step_loop.py",
        "tests/unit/agents/authoring/test_shared_context_view.py",
        "tests/unit/runtime/nexus/tools/test_capability_payload_tool_plan.py",
        "tests/unit/runtime/nexus/tools/test_tool_gateway.py",
        "tests/unit/runtime/nexus/context/test_context_budget.py",
        "tests/unit/runtime/nexus/context/test_context_engine.py",
        "tests/unit/runtime/nexus/test_orchestration_capabilities.py",
        "tests/unit/runtime/critic/test_critic_contracts.py",
        "tests/unit/prompts/test_prompt_registry_resolver.py",
        "tests/unit/tools/registry/test_wiring.py",
        "tests/unit/llm_adapters/test_adapter_response_contract.py",
        "tests/unit/applications/test_manifest.py",
        "tests/unit/applications/test_manifest_conformance.py",
        "tests/unit/applications/test_prompt_wiring.py",
        "tests/unit/applications/test_capability_graph_wiring.py",
        "tests/unit/applications/test_registry_wiring.py",
        "tests/unit/applications/test_package_wiring.py",
        "tests/unit/applications/test_scaffold_acceptance.py",
        "tests/unit/applications/test_application_dependency_model.py",
        "tests/unit/core/plugins/test_catalog_bootstrap.py",
        "tests/unit/knowledge/contracts/test_document_conformance.py",
    }
)


def _rel_test_path(nodeid: str) -> str:
    return nodeid.replace("\\", "/").split("::", 1)[0]


def _in_ci_smoke_scope(nodeid: str) -> bool:
    rel = _rel_test_path(nodeid)
    if rel in CI_SMOKE_FILES:
        return True
    return any(rel.startswith(prefix) for prefix in CI_SMOKE_DIR_PREFIXES)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("no_ci"):
            continue
        if _in_ci_smoke_scope(item.nodeid):
            item.add_marker(pytest.mark.ci_smoke)
