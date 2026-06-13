# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeGenerationAdapter — separate LLM profile for craft codegen (ECC-2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CodeGenerationAdapter(Protocol):
    """Generate or patch candidate code for a craft goal."""

    def generate(self, *, goal: str, constraints: str = "", language: str = "python") -> str: ...

    def patch(
        self,
        *,
        goal: str,
        code: str,
        diagnostics: str,
        language: str = "python",
    ) -> str: ...


class TemplateCodeGenerationAdapter:
    """Deterministic adapter for gate tests — no network."""

    def generate(self, *, goal: str, constraints: str = "", language: str = "python") -> str:
        _ = constraints, language
        safe_goal = goal.replace('"', "'").strip() or "task"
        return f'print("{safe_goal}")\n'

    def patch(
        self,
        *,
        goal: str,
        code: str,
        diagnostics: str,
        language: str = "python",
    ) -> str:
        _ = goal, diagnostics, language
        if "SyntaxError" in diagnostics or "syntax" in diagnostics.lower():
            return 'print("patched")\n'
        return code
