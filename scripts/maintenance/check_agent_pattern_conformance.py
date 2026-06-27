# © Artur Czarnecki. All rights reserved.

"""Verify AgentContract.cognitive_pattern matches agent class (ACP-13)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from intergrax.agents.authoring.patterns import PATTERN_AGENT_BY_ID
from intergrax.contracts.agent_run_enums import CognitivePattern

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_ROOT = REPO_ROOT / "agents"


def _class_pattern_from_ast(tree: ast.AST) -> CognitivePattern | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "cognitive_pattern":
                    if isinstance(stmt.value, ast.Attribute) and isinstance(stmt.value.value, ast.Name):
                        if stmt.value.value.id == "CognitivePattern":
                            member = stmt.value.attr
                            try:
                                return CognitivePattern[member]
                            except KeyError:
                                return None
    return None


def main() -> int:
    errors: list[str] = []
    for contract_path in sorted(AGENTS_ROOT.glob("*/contract.py")):
        source = contract_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        declared: CognitivePattern | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_PATTERN":
                        if isinstance(node.value, ast.Attribute):
                            member = node.value.attr
                            try:
                                declared = CognitivePattern[member]
                            except KeyError:
                                pass
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id != "AgentContract":
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "cognitive_pattern" and isinstance(keyword.value, ast.Attribute):
                        member = keyword.value.attr
                        try:
                            declared = CognitivePattern[member]
                        except KeyError:
                            pass

        if declared is None:
            continue

        agent_py = contract_path.parent / f"{contract_path.parent.name}_agent.py"
        if not agent_py.exists():
            continue
        class_pattern = _class_pattern_from_ast(ast.parse(agent_py.read_text(encoding="utf-8")))
        if class_pattern is None:
            base_names = {
                base.id
                for node in ast.walk(ast.parse(agent_py.read_text(encoding="utf-8")))
                if isinstance(node, ast.ClassDef)
                for base in node.bases
                if isinstance(base, ast.Name)
            }
            for pattern, cls in PATTERN_AGENT_BY_ID.items():
                if cls.__name__ in base_names:
                    class_pattern = pattern
                    break

        if class_pattern != declared:
            errors.append(
                f"{agent_py}: contract declares {declared!r} but class pattern is {class_pattern!r}"
            )

    if errors:
        print("check_agent_pattern_conformance: FAIL", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("check_agent_pattern_conformance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
