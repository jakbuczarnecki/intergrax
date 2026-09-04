# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.architecture.capability_graph import build_catalog_capability_graph

REPO_ROOT = Path(__file__).resolve().parents[4]
CAPABILITY_GRAPH_MODULE = REPO_ROOT / "intergrax" / "runtime" / "architecture" / "capability_graph.py"

_FORBIDDEN_LIFECYCLE_FRAGMENTS = (
    "AgentRegistry",
    "InstallationService",
    "BindingService",
    "ActivationService",
    "RuntimeRevisionService",
)


def test_capability_graph_module_has_no_lifecycle_authority_imports() -> None:
    text = CAPABILITY_GRAPH_MODULE.read_text(encoding="utf-8")
    violations = [
        f"contains forbidden lifecycle fragment {fragment!r}"
        for fragment in _FORBIDDEN_LIFECYCLE_FRAGMENTS
        if fragment in text
    ]
    assert not violations, "\n".join(violations)


def test_capability_graph_builder_does_not_instantiate_agents() -> None:
    graph = build_catalog_capability_graph()
    assert graph.nodes
    assert graph.edges
