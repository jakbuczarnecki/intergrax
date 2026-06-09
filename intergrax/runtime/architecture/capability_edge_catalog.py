# © Artur Czarnecki. All rights reserved.

"""Exported capability edge catalog (IDEAL-20.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdgeType,
    CapabilityNodeType,
    _ALLOWED_EDGES,
)


def build_edge_catalog() -> dict[str, Any]:
    edges: list[dict[str, str]] = []
    for edge_type, pairs in _ALLOWED_EDGES.items():
        for source, target in sorted(pairs):
            edges.append(
                {
                    "edge_type": edge_type.value,
                    "source_type": source.value,
                    "target_type": target.value,
                }
            )
    return {"schema_version": "1.0.0", "edges": edges}


def catalog_path(repo_root: Path) -> Path:
    return repo_root / "intergrax" / "runtime" / "architecture" / "capability_edge_catalog.json"


def write_catalog(repo_root: Path) -> Path:
    path = catalog_path(repo_root)
    path.write_text(json.dumps(build_edge_catalog(), indent=2) + "\n", encoding="utf-8")
    return path


def load_catalog(repo_root: Path) -> dict[str, Any]:
    return json.loads(catalog_path(repo_root).read_text(encoding="utf-8"))
