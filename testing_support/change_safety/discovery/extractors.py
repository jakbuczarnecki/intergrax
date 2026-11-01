# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from testing_support.change_safety.discovery.git_paths import git_ls_files

_REPO = Path(__file__).resolve().parents[3]

DISCOVERY_ROOTS: tuple[str, ...] = (
    "intergrax/",
    "agents/",
    "applications/",
    "testing_support/change_safety/",
)


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    category: str


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    rel: str


def _module_id(path: str) -> str:
    rel = path.replace("\\", "/")
    if not rel.endswith(".py"):
        return f"path:{rel}"
    body = rel[:-3].replace("/", ".")
    return f"module:{body}"


def _extract_import_edges(path: Path) -> tuple[GraphEdge, ...]:
    rel = path.relative_to(_REPO).as_posix()
    source = _module_id(rel)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError:
        return ()
    edges: list[GraphEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(GraphEdge(source, f"module:{alias.name}", "IMPORTS"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            edges.append(GraphEdge(source, f"module:{node.module}", "IMPORTS"))
    return tuple(edges)


def build_inventory_nodes() -> tuple[GraphNode, ...]:
    return _build_inventory_nodes_cached()


@lru_cache(maxsize=4)
def _build_inventory_nodes_cached() -> tuple[GraphNode, ...]:
    nodes: list[GraphNode] = []
    for root in DISCOVERY_ROOTS:
        prefix = root.rstrip("/")
        for rel in git_ls_files(prefix):
            if rel.endswith(".py"):
                nodes.append(GraphNode(_module_id(rel), "module"))
    return tuple(sorted(nodes, key=lambda n: n.node_id))


def build_import_edges() -> tuple[GraphEdge, ...]:
    return _build_import_edges_cached()


@lru_cache(maxsize=4)
def _build_import_edges_cached() -> tuple[GraphEdge, ...]:
    edges: list[GraphEdge] = []
    for root in DISCOVERY_ROOTS:
        prefix = root.rstrip("/")
        for rel in git_ls_files(prefix):
            if rel.endswith(".py"):
                edges.extend(_extract_import_edges(_REPO / rel))
    return tuple(sorted(edges, key=lambda e: (e.source_id, e.target_id, e.rel)))
