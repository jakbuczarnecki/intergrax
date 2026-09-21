# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D-R3-R1 — canonical WebSearch executor contract ownership gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from intergrax.tools.providers.websearch.executor_contract import WebSearchQueryExecutor
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.websearch.schemas.web_search_result import WebSearchResult

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_CONTRACT = _REPO_ROOT / "intergrax/tools/providers/websearch/executor_contract.py"
_RESEARCH_COMPOSITION = _REPO_ROOT / "applications/research_application/host/host_runtime_composition.py"
_TOOL_WIRING_CTX = _REPO_ROOT / "intergrax/tools/registry/wiring.py"
_APP_TOOL_WIRING = _REPO_ROOT / "intergrax/applications/_shared/tool_wiring.py"
_ENV_WIRING = _REPO_ROOT / "intergrax/applications/_shared/environment_wiring.py"
_WEBSEARCH_SERVICE = _REPO_ROOT / "intergrax/tools/providers/websearch/service.py"

_FORBIDDEN_RESEARCH_PROTOCOL = "ResearchWebSearchExecutor"
_WEAK_EXECUTOR_ANN = re.compile(
    r"websearch_executor\s*:\s*(Any|object)(\s*\|\s*None)?",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _annotation_for_param(func: ast.FunctionDef, name: str) -> str | None:
    for arg in (*func.args.args, *func.args.kwonlyargs):
        if arg.arg == name and arg.annotation is not None:
            return ast.unparse(arg.annotation)
    return None


def _websearch_executor_field_annotation(path: Path) -> str | None:
    tree = ast.parse(_read(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ToolWiringContext":
            for child in node.body:
                if (
                    isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and child.target.id == "websearch_executor"
                    and child.annotation is not None
                ):
                    return ast.unparse(child.annotation)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "build_application_tool_wiring",
            "wire_application_environment",
        }:
            ann = _annotation_for_param(node, "websearch_executor")
            if ann is not None:
                return ann
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_search_via_executor":
            if node.args.args and node.args.args[0].annotation is not None:
                return ast.unparse(node.args.args[0].annotation)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ResearchHostRuntimeComposition":
            for child in node.body:
                if (
                    isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and child.target.id == "websearch_executor"
                    and child.annotation is not None
                ):
                    return ast.unparse(child.annotation)
    return None


def _protocol_classes_with_search_sync() -> list[str]:
    names: list[str] = []
    paths: list[Path] = []
    for root in (_REPO_ROOT / "intergrax", _REPO_ROOT / "applications"):
        if root.is_dir():
            paths.extend(root.rglob("*.py"))
    for path in paths:
        if "node_modules" in path.parts or ".venv" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            is_protocol = any(
                (isinstance(base, ast.Name) and base.id == "Protocol")
                or (isinstance(base, ast.Attribute) and base.attr == "Protocol")
                for base in node.bases
            )
            if not is_protocol:
                continue
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "search_sync":
                    rel = path.relative_to(_REPO_ROOT).as_posix()
                    names.append(f"{rel}:{node.name}")
    return sorted(names)


class _ExternalWebSearchExecutor:
    def search_sync(
        self,
        query: str,
        top_k: int | None = None,
        locale: str | None = None,
        region: str | None = None,
        language: str | None = None,
        safe_search: bool | None = None,
    ) -> list[WebSearchResult]:
        _ = query, top_k, locale, region, language, safe_search
        return []


def test_no_research_local_executor_protocol_symbol() -> None:
    source = _read(_RESEARCH_COMPOSITION)
    assert _FORBIDDEN_RESEARCH_PROTOCOL not in source
    assert "WebSearchQueryExecutor" in source


def test_single_canonical_search_sync_protocol() -> None:
    matches = _protocol_classes_with_search_sync()
    assert matches == [
        "intergrax/tools/providers/websearch/executor_contract.py:WebSearchQueryExecutor",
    ]


def test_tool_wiring_context_websearch_executor_typed() -> None:
    ann = _websearch_executor_field_annotation(_TOOL_WIRING_CTX)
    assert ann is not None
    assert "WebSearchQueryExecutor" in ann
    assert "Any" not in ann and "object" not in ann


def test_build_application_tool_wiring_websearch_executor_typed() -> None:
    ann = _websearch_executor_field_annotation(_APP_TOOL_WIRING)
    assert ann is not None
    assert "WebSearchQueryExecutor" in ann
    assert "Any" not in ann


def test_wire_application_environment_websearch_executor_typed() -> None:
    ann = _websearch_executor_field_annotation(_ENV_WIRING)
    assert ann is not None
    assert "WebSearchQueryExecutor" in ann
    assert "Any" not in ann


def test_websearch_service_search_via_executor_typed() -> None:
    ann = _websearch_executor_field_annotation(_WEBSEARCH_SERVICE)
    assert ann is not None
    assert "WebSearchQueryExecutor" in ann
    assert "Any" not in ann


def test_executor_path_modules_avoid_weak_websearch_executor_annotations() -> None:
    violations: list[str] = []
    for path in (
        _TOOL_WIRING_CTX,
        _APP_TOOL_WIRING,
        _ENV_WIRING,
        _WEBSEARCH_SERVICE,
        _RESEARCH_COMPOSITION,
    ):
        for match in _WEAK_EXECUTOR_ANN.finditer(_read(path)):
            violations.append(f"{path.relative_to(_REPO_ROOT)}: {match.group(0)}")
    assert not violations


def test_research_composition_does_not_import_concrete_websearch_executor() -> None:
    source = _read(_RESEARCH_COMPOSITION)
    assert "intergrax.websearch.service.websearch_executor" not in source
    assert "WebSearchExecutor" not in source


def test_canonical_contract_module_has_no_concrete_executor_import() -> None:
    source = _read(_CANONICAL_CONTRACT)
    assert "websearch_executor" not in source
    assert "google" not in source.lower()
    assert "bing" not in source.lower()


def test_external_executor_structurally_satisfies_port() -> None:
    executor = _ExternalWebSearchExecutor()
    assert isinstance(executor, WebSearchQueryExecutor)
    ctx = ToolWiringContext(websearch_executor=executor)
    assert ctx.websearch_executor is executor
