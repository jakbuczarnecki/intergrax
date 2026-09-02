# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q4 static anti-forcing gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JOB_PATH = _REPO_ROOT / "agents" / "model_routing_qualifier" / "steps" / "model_routing_job.py"


def _function_body_names(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    raise AssertionError(f"{function_name} not found in {path}")


def test_run_model_routing_job_has_no_post_decision_profile_override() -> None:
    source = _JOB_PATH.read_text(encoding="utf-8")
    assert "force_selected_profile" not in source
    assert "override_selected_profile" not in source


def test_run_model_routing_job_has_no_post_generation_output_swap() -> None:
    names = _function_body_names(_JOB_PATH, "run_model_routing_job")
    assert "replace_model_output" not in names
    assert "swap_response" not in names


def test_routing_modules_do_not_import_diagnostics() -> None:
    paths = (
        _REPO_ROOT / "intergrax" / "llm_adapters" / "routing" / "evaluator.py",
        _REPO_ROOT / "intergrax" / "llm_adapters" / "registry" / "model_router.py",
        _REPO_ROOT / "intergrax" / "applications" / "_shared" / "llm_resolver.py",
    )
    forbidden = ("functional_diagnostics_q4", "functional_diagnostic_analyzer")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for token in forbidden:
                    assert token not in node.module
