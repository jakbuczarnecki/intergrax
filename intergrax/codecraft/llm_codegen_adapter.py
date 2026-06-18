# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LLM-backed CodeGenerationAdapter (ECC-MAINT-02)."""

from __future__ import annotations

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


class LLMCodeGenerationAdapter:
    """Wrap a dedicated LLM adapter for craft codegen with trace-visible identity."""

    def __init__(self, llm: LLMAdapter, *, profile_ref: str | None = None) -> None:
        self._llm = llm
        ref_suffix = f"@{profile_ref}" if profile_ref else ""
        self.model_id = f"{llm._provider_slug()}/{llm.model}{ref_suffix}"

    def generate(self, *, goal: str, constraints: str = "", language: str = "python") -> str:
        messages = [
            ChatMessage(
                role="system",
                content=f"You generate minimal {language} code. Output executable code only.",
            ),
            ChatMessage(
                role="user",
                content=f"Goal: {goal}\nConstraints: {constraints or 'none'}",
            ),
        ]
        response = self._llm.generate_messages(messages, temperature=0.2)
        return (response.content or "").strip()

    def patch(
        self,
        *,
        goal: str,
        code: str,
        diagnostics: str,
        language: str = "python",
    ) -> str:
        messages = [
            ChatMessage(
                role="system",
                content=f"Fix {language} code based on diagnostics. Output code only.",
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Goal: {goal}\nDiagnostics:\n{diagnostics}\n\nCurrent code:\n{code}"
                ),
            ),
        ]
        response = self._llm.generate_messages(messages, temperature=0.2)
        return (response.content or "").strip()
