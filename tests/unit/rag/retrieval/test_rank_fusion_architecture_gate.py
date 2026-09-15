# © Artur Czarnecki. All rights reserved.

"""Architecture gate — rank fusion must stay platform-owned."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_IMPORT_ROOTS = (
    "platform_proofs",
    "platform_proofs.scenarios.verified_product_identification",
)


def test_rank_fusion_has_no_platform_proofs_imports() -> None:
    package_root = (
        Path(__file__).resolve().parents[4] / "intergrax" / "rag" / "retrieval" / "fusion"
    )
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name):
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _is_forbidden(node.module):
                    violations.append(f"{path}: from {node.module}")
    assert violations == []


def test_fusion_retriever_accepts_strategy_port() -> None:
    from intergrax.rag.retrieval.fusion import RankFusionStrategyPort
    from intergrax.rag.retrievers.providers.fusion_retriever import FusionRetriever

    assert "fusion_strategy" in FusionRetriever.__init__.__code__.co_varnames
    annotations = FusionRetriever.__init__.__annotations__
    assert "fusion_strategy" in annotations
    assert "RankFusionStrategyPort" in str(annotations["fusion_strategy"])


def _is_forbidden(module_name: str) -> bool:
    return any(
        module_name == root or module_name.startswith(f"{root}.")
        for root in FORBIDDEN_IMPORT_ROOTS
    )
