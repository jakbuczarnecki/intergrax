# © Artur Czarnecki. All rights reserved.

"""Module content evidence for TRACE-X-P6 classification (path + markers, not filename heuristics)."""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=512)
def _read_module_text(repo_relative_path: str) -> str:
    path = _REPO_ROOT / repo_relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def restart_resume_markers_in_module(repo_relative_path: str, markers: tuple[str, ...]) -> frozenset[str]:
    text = _read_module_text(repo_relative_path)
    return frozenset(marker for marker in markers if marker in text)


def terminal_evidence(repo_relative_path: str) -> frozenset[str]:
    text = _read_module_text(repo_relative_path)
    if not text:
        return frozenset({"missing_module"})
    evidence: set[str] = set()
    if re.search(r"\bclass\s+ExecutionTerminalService\b", text):
        evidence.add("defines_execution_terminal_service")
    if re.search(r"\bcommit_terminal_outcome\s*\(", text):
        evidence.add("calls_commit_terminal_outcome")
    elif "commit_terminal_outcome" in text:
        evidence.add("references_commit_terminal_outcome")
    if re.search(r"\brecord_cancellation\s*\(", text):
        evidence.add("calls_record_cancellation")
    elif "record_cancellation" in text:
        evidence.add("references_record_cancellation")
    if "reconcile_task_state_with_terminal" in text:
        evidence.add("reconcile_task_state_with_terminal")
    if "ExecutionTerminalOutcome" in text:
        evidence.add("execution_terminal_outcome_type")
    if "ExecutionTerminalRecord" in text:
        evidence.add("execution_terminal_record_type")
    if "ExecutionTerminalConflictError" in text:
        evidence.add("execution_terminal_conflict_type")
    if "ExecutionTerminalService(" in text:
        evidence.add("constructs_execution_terminal_service")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        evidence.add("syntax_error")
        return frozenset(evidence)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ExecutionTerminalService":
            evidence.add("ast_class_execution_terminal_service")
    return frozenset(evidence)
