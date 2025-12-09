# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Dict, Any, Optional

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.websearch.service.websearch_executor import WebSearchExecutor
from intergrax.websearch.context.websearch_context_builder import WebSearchContextBuilder
from intergrax.llm_adapters import LLMAdapter

from intergrax.llm.messages import ChatMessage


class WebSearchAnswerer:
    """
    High-level helper that:
      1) runs web search via WebSearchExecutor,
      2) builds LLM-ready context/messages from web documents,
      3) calls any LLMAdapter to generate a final answer.

    This class is model-agnostic: it only requires an LLMAdapter implementation.
    """

    def __init__(
        self,
        executor: WebSearchExecutor,
        adapter: LLMAdapter,
        context_builder: Optional[WebSearchContextBuilder] = None,
        answer_language: str = GLOBAL_SETTINGS.default_language,
        system_prompt_override: Optional[str] = None,
    ) -> None:
        """
        Parameters:
          executor              : WebSearchExecutor instance (providers, rate limits, etc.).
          adapter               : LLMAdapter instance (OpenAI, Gemini, Ollama, etc.).
          context_builder       : WebSearchContextBuilder for building messages from web docs.
          answer_language       : language of the final answer (for prompt instruction).
          system_prompt_override: optional global system prompt for all calls (can be overridden per call).
        """
        self.executor = executor
        self.adapter = adapter
        self.context_builder = context_builder or WebSearchContextBuilder()
        self.answer_language = answer_language
        self.system_prompt_override = system_prompt_override

    async def answer_async(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full async flow:
          1) Web search,
          2) Build messages from sources,
          3) Call LLMAdapter,
          4) Return answer + sources.

        Per-call system_prompt_override, if provided, has priority over the
        instance-level system_prompt_override set in __init__.

        Returns:
          {
            "answer": str,
            "messages": List[ChatMessage],
            "web_docs": List[Dict[str, Any]]
          }
        """
        # 1) Run web search (serialized dicts ready for context building)
        web_docs = await self.executor.search_async(
            query=question,
            top_k=top_k,
            serialize=True,
        )

        # 2) Decide which system prompt to use
        effective_system_prompt = system_prompt_override or self.system_prompt_override

        # 3) Build LLM-ready messages (system + user)
        msg_dicts = self.context_builder.build_messages_from_serialized(
            user_question=question,
            web_docs=web_docs,
            answer_language=self.answer_language,
            system_prompt_override=effective_system_prompt,
        )

        messages: List[ChatMessage] = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in msg_dicts
        ]

        # 4) Call LLM via adapter
        answer_text = self.adapter.generate_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return {
            "answer": answer_text,
            "messages": messages,
            "web_docs": web_docs,
        }

    def answer_sync(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous convenience wrapper for non-async environments.

        NOTE:
          Do not use this inside environments with a running event loop
          (e.g. Jupyter). Prefer 'answer_async' there.
        """
        import asyncio

        return asyncio.run(
            self.answer_async(
                question=question,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt_override=system_prompt_override,
            )
        )
