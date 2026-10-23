# © Artur Czarnecki. All rights reserved.

"""AST scan for intergrax.runtime imports in contract-layer Python sources."""

from __future__ import annotations

import ast
from pathlib import Path


def collect_contract_file_runtime_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime"):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime"):
                hits.append(node.module)
    return tuple(hits)
