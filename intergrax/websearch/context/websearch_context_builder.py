# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import Sequence, List, Optional

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm.messages import ChatMessage
from intergrax.websearch.schemas.web_search_result import WebSearchResult


class WebSearchContextBuilder:
    """
    Builds LLM-ready textual context and chat messages from web search results.

    It can work with:
      - raw WebDocument objects (from the pipeline / executor),
      - serialized dicts (as returned by WebSearchExecutor with serialize=True).

    Typical flow:
      1) Run websearch → obtain list[WebDocument] or list[dict]
      2) Build context string from top-N documents
      3) Build chat messages (system + user) for any chat-style LLM
    """

    def __init__(
        self,
        max_docs: int = 4,
        max_chars_per_doc: int = 1500,
        include_snippet: bool = True,
        include_url: bool = True,
        source_label_prefix: str = "Source",
    ) -> None:
        """
        Parameters:
          max_docs           : maximum number of documents to include in context
          max_chars_per_doc  : maximum number of characters from each document text
          include_snippet    : include search snippet in context header
          include_url        : include document URL in context header
          source_label_prefix: prefix for source labels, e.g. "Source" -> [Source 1]
        """
        self.max_docs = max_docs
        self.max_chars_per_doc = max_chars_per_doc
        self.include_snippet = include_snippet
        self.include_url = include_url
        self.source_label_prefix = source_label_prefix


    def _build_system_prompt(self) -> str:
        """
        Builds a strict system prompt enforcing:
          - use only web sources,
          - no hallucinations,
          - single, concise answer,
          - explicit handling of missing information.
        """
        return (
            "You are a careful research assistant.\n"
            "You MUST base your answer ONLY on the information contained in the 'Web sources' section.\n"
            "Follow these rules strictly:\n"
            "1) Do NOT use any outside knowledge or assumptions beyond the provided sources.\n"
            "2) If the sources do NOT contain enough information to answer something, explicitly say that it is not specified.\n"
            "3) Do NOT invent or guess facts (e.g. prices, popularity, performance, features) that are not clearly stated.\n"
            "4) Do NOT repeat the same information multiple times.\n"
            "5) Provide ONE coherent answer, not a list of alternative stories.\n"
            "6) When you refer to a specific statement from the sources, cite it using [Source N].\n"
        )

    def _build_user_prompt(
        self,
        user_question: str,
        context: str,
        answer_language: str,
    ) -> str:
        """
        Builds the user-facing prompt that wraps:
          - web sources,
          - the question,
          - concrete tasks.
        """
        return (
            f"Web sources:\n{context}\n\n"
            f"User question:\n{user_question}\n\n"
            "Tasks:\n"
            "1) Read all web sources carefully.\n"
            "2) Extract ONLY the information that is explicitly stated in the sources and relevant to the question.\n"
            "3) Answer the question in a single, coherent response in language: "
            f"{answer_language}.\n"
            "4) Be concise and avoid repetition. Prefer 1-3 short paragraphs or a short bullet list.\n"
            "5) If some aspect of the question is NOT covered by the sources, clearly say that the sources do not provide this information.\n"
            f"6) Add [{self.source_label_prefix} N] markers next to statements that come from specific sources.\n"
        )

    def _build_context_from_results(
        self,
        web_results: Sequence[WebSearchResult],
    ) -> str:
        """
        Builds a textual context string from typed WebSearchResult objects.
        """
        sections: List[str] = []
        for idx, doc in enumerate(web_results[: self.max_docs], start=1):
            title = (doc.title or "").strip() or "(no title)"
            url = (doc.url or "").strip()
            snippet = (doc.snippet or "").strip() if doc.snippet else ""
            text = (doc.text or "").strip()

            if self.max_chars_per_doc and len(text) > self.max_chars_per_doc:
                text = text[: self.max_chars_per_doc]

            header_lines: List[str] = []
            header_lines.append(f"[{self.source_label_prefix} {idx}]")
            header_lines.append(f"Title: {title}")
            if self.include_url and url:
                header_lines.append(f"URL: {url}")
            if self.include_snippet and snippet:
                header_lines.append(f"Snippet: {snippet}")
            header_lines.append("")

            section = "\n".join(header_lines + [text])
            sections.append(section)

        return "\n\n---\n\n".join(sections)
    

    # -------------------------------------------------------------------------
    # Messages for LLM – STRICT "sources-only" mode
    # -------------------------------------------------------------------------    

    def build_messages_from_results(
        self,
        user_question: str,
        web_results: Sequence[WebSearchResult],
        answer_language: str = GLOBAL_SETTINGS.default_language,
        system_prompt_override: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        Builds a typical pair of chat messages (system + user) for chat-style LLMs
        from typed web search results.

        Returns:
        [
            ChatMessage(role="system", content="..."),
            ChatMessage(role="user", content="...")
        ]
        """
        context = self._build_context_from_results(web_results)

        system_prompt = system_prompt_override or self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            user_question=user_question,
            context=context,
            answer_language=answer_language,
        )

        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

