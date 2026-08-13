# © Artur Czarnecki. All rights reserved.

"""Shared runtime-graph semantics used by legacy and AP-7 consumers."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from intergrax.agent_distribution import runtime_graph_service
from intergrax.applications._shared import application_runtime_graph
from intergrax.runtime_graph_semantics import (
    format_agent_dependency_cycle,
    is_agent_distribution,
    is_application_distribution,
    normalize_distribution_name,
    parse_dependency_name,
)


def test_shared_dependency_name_normalization_used_by_both_consumers() -> None:
    dep = '  "requests>=2.32; python_version >= \\"3.12\\""  '
    assert parse_dependency_name(dep) == "requests"
    assert normalize_distribution_name("Intergrax_AI") == "intergrax-ai"
    assert application_runtime_graph.normalize_distribution_name is normalize_distribution_name
    assert runtime_graph_service.normalize_distribution_name is normalize_distribution_name
    assert application_runtime_graph.parse_dependency_name is parse_dependency_name
    assert runtime_graph_service.parse_dependency_name is parse_dependency_name


def test_shared_agent_distribution_classification_used_by_both() -> None:
    assert is_agent_distribution("intergrax-local-search-agent")
    assert is_agent_distribution("intergrax-assistant-agent")
    assert not is_agent_distribution("intergrax-demo-application")
    assert application_runtime_graph.agent_dir_from_distribution(
        "intergrax-local-search-agent"
    ) == "local_search"
    assert runtime_graph_service.is_agent_distribution("intergrax-local-indexer-agent")


def test_shared_application_tier_violation_classification_used_by_both() -> None:
    assert is_application_distribution("intergrax-demo-application")
    assert not is_application_distribution("intergrax-local-search-agent")
    assert runtime_graph_service.is_application_distribution(
        "intergrax-demo-application"
    )


def test_cycle_formatting_remains_equivalent() -> None:
    message = format_agent_dependency_cycle(
        ["intergrax-a-agent", "intergrax-b-agent"],
        "intergrax-a-agent",
    )
    assert message.startswith("AGENT_DEPENDENCY_CYCLE:")
    assert "intergrax-a-agent" in message
    assert "intergrax-b-agent" in message


def test_neutral_module_has_no_forbidden_imports() -> None:
    repo = Path(__file__).resolve().parents[2]
    source = (repo / "intergrax" / "runtime_graph_semantics.py").read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.append(node.module)
        for module in modules:
            top = module.split(".", 1)[0]
            assert top not in {"applications", "agents"}
            assert not module.startswith("intergrax.agent_distribution")


def test_ap7_runtime_graph_service_has_no_duplicate_taxonomy_regex() -> None:
    source = inspect.getsource(runtime_graph_service)
    assert "_AGENT_DIST_RE" not in source
    assert "_APPLICATION_DIST_RE" not in source
    assert "def _normalize_distribution_name" not in source
    assert "def _parse_dependency_name" not in source
