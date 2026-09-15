# © Artur Czarnecki. All rights reserved.

"""LLM-backed session memory extraction strategy (MEM-ENT-4)."""

from __future__ import annotations

import json
from typing import List, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm.messages import ChatMessage
from intergrax.memory.session_summary_schema import SessionSummarySchema
from intergrax.memory.strategies.errors import MemoryStrategyProviderError
from intergrax.memory.strategies.models import (
    MemoryCandidate,
    MemoryExtractionRequest,
    MemoryExtractionResult,
)
from intergrax.memory.user_profile_memory import MemoryImportance, MemoryKind


class LlmMemoryExtractionStrategy:
    strategy_id = "builtin.memory.extraction.llm_session"

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    async def extract(self, request: MemoryExtractionRequest) -> MemoryExtractionResult:
        if not request.messages:
            return MemoryExtractionResult(candidates=())

        prompt_text = self._build_prompt(request)
        try:
            llm_output = self._call_llm(prompt_text, temperature=request.temperature, run_id=request.run_id)
        except Exception as exc:
            raise MemoryStrategyProviderError(f"LLM extraction failed: {exc}") from exc

        parsed = self._parse_llm_output(llm_output)
        if parsed is None:
            return MemoryExtractionResult(candidates=())

        candidates = self._candidates_from_parsed(request, parsed)
        return MemoryExtractionResult(candidates=tuple(candidates))

    def _build_prompt(self, request: MemoryExtractionRequest) -> str:
        conversation_lines: List[str] = []
        for message in request.messages:
            content = (message.content or "").strip()
            if not content:
                continue
            conversation_lines.append(f"{message.role}: {content}")

        conversation_block = "\n".join(conversation_lines) or "(empty session)"
        language = request.language

        return f"""
You are an AI specializing in extracting long-term user profile memory
from a single chat session.

Your task:
- Read the conversation below.
- Decide which pieces of information are:
    * USER_FACT  - stable facts or long-term goals about the user
                   (e.g. background, skills, health constraints, recurring goals).
    * PREFERENCE - stable or recurring preferences about communication,
                   workflow, tools, style, or constraints (e.g. "no emojis in code").
    * SESSION_SUMMARY - a short global summary of what happened in this
                   specific session, useful as context for future work.

Important distinctions:
- USER_FACT:
    - Should be true regardless of a single session.
    - Describes who the user is, what they does, what they care about,
      which constraints and long-term goals they have.
    - Example: "The user is a senior software engineer working with Python and .NET."
- PREFERENCE:
    - Describes how the user wants the assistant to respond or work.
    - Example: "The user prefers very technical, concise answers in English,
      without emojis and with code examples."
- SESSION_SUMMARY:
    - High-level recap of the current session only.
    - Should mention the main problem(s), decisions, and next steps.
    - Avoid very fine-grained step-by-step details.

Rules:
- Focus ONLY on information that will be useful across many future sessions.
- Ignore transient, local details that are unlikely to matter later
  (e.g. one-off examples, small talk, debug attempts that will not repeat).
- Ignore meta-information about the model itself (e.g. that you are an AI).
- Do NOT invent facts that are not clearly supported by the conversation.
- If you are not sure whether something is true or stable, either:
    * skip it, or
    * mark it with LOWER importance (LOW).
- Avoid time-sensitive language such as "today", "recently", "this week".
- Use the target language: {language}.
- Do NOT include sensitive data that should not be stored long-term
  (e.g. passwords, tokens, very private health details).
- Keep the content concise but precise.

Output format:
Return a single JSON object with the following structure:

{{
  "facts": [
    {{
      "title": "short label",
      "content": "concise description of the fact or goal",
      "importance": "LOW | MEDIUM | HIGH | CRITICAL",
      "tags": ["user", "goal"]
    }}
  ],
  "preferences": [
    {{
      "title": "short label",
      "content": "concise description of the preference",
      "importance": "LOW | MEDIUM | HIGH | CRITICAL",
      "tags": ["communication", "tone"]
    }}
  ],
  "session_summary": {{
    "title": "global summary",
    "content": "short paragraph describing the session",
    "importance": "LOW | MEDIUM | HIGH | CRITICAL",
    "tags": ["session_summary"]
  }}
}}

Constraints:
- Return ONLY valid JSON (no comments, no trailing commas, no markdown).
- If you do not want to provide a session summary, set "session_summary": null.
- Respect the importance scale: use HIGH or CRITICAL only for items that
  are clearly central for future collaboration and appear strongly in the
  conversation.

Session identifier (for your reasoning only, do not repeat it verbatim):
- session_id: {request.session_id}

Conversation:
{conversation_block}
"""

    def _call_llm(
        self,
        prompt_text: str,
        *,
        temperature: float | None,
        run_id: str | None,
    ) -> str:
        messages: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content=(
                    "You extract structured long-term memory from a single "
                    "chat session and output pure JSON."
                ),
            ),
            ChatMessage(
                role="user",
                content=prompt_text,
            ),
        ]
        response = self._llm.generate_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=None,
            run_id=run_id,
        )
        return response.content

    def _parse_llm_output(self, text: str) -> Optional[dict[str, object]]:
        if not text:
            return None

        stripped = text.strip()
        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, dict):
                return loaded
            return None
        except json.JSONDecodeError:
            pass

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    return loaded
            except json.JSONDecodeError:
                return None

        return None

    def _candidates_from_parsed(
        self,
        request: MemoryExtractionRequest,
        parsed: dict[str, object],
    ) -> List[MemoryCandidate]:
        candidates: List[MemoryCandidate] = []
        facts = parsed.get("facts") or []
        preferences = parsed.get("preferences") or []
        summary = parsed.get("session_summary")

        if isinstance(facts, list):
            for item in facts[: request.max_facts]:
                candidate = self._candidate_from_item(
                    item=item,
                    request=request,
                    expected_kind=MemoryKind.USER_FACT,
                    default_importance=request.default_fact_importance,
                )
                if candidate is not None:
                    candidates.append(candidate)

        if isinstance(preferences, list):
            for item in preferences[: request.max_preferences]:
                candidate = self._candidate_from_item(
                    item=item,
                    request=request,
                    expected_kind=MemoryKind.PREFERENCE,
                    default_importance=request.default_preference_importance,
                )
                if candidate is not None:
                    candidates.append(candidate)

        if request.include_session_summary and summary:
            structured = self._structured_summary_from_parsed(summary, session_id=request.session_id)
            summary_item: dict[str, object]
            if isinstance(summary, dict):
                summary_item = {
                    "title": structured.title,
                    "content": structured.to_storage_text(),
                    "importance": summary.get("importance") or "MEDIUM",
                    "tags": summary.get("tags") or ["session_summary"],
                }
            else:
                summary_item = {
                    "title": structured.title,
                    "content": structured.to_storage_text(),
                    "importance": "MEDIUM",
                    "tags": ["session_summary"],
                }
            summary_candidate = self._candidate_from_item(
                item=summary_item,
                request=request,
                expected_kind=MemoryKind.SESSION_SUMMARY,
                default_importance=request.default_summary_importance,
                structured_summary=structured,
            )
            if summary_candidate is not None:
                candidates.append(summary_candidate)
                episodic = MemoryCandidate(
                    content=structured.narrative or structured.to_storage_text(),
                    session_id=request.session_id,
                    kind=MemoryKind.EPISODIC_EVENT,
                    title=structured.title or "Session episodic recap",
                    importance=request.default_summary_importance,
                    tags=("session_consolidation",),
                    source="session_consolidation",
                    structured_summary=None,
                    include_episodic_from_summary=False,
                )
                candidates.append(episodic)

        return candidates

    def _structured_summary_from_parsed(
        self,
        summary: object,
        *,
        session_id: str,
    ) -> SessionSummarySchema:
        if not isinstance(summary, dict):
            return SessionSummarySchema(session_id=session_id, narrative=str(summary))
        facts_raw = summary.get("facts") or []
        tasks_raw = summary.get("open_tasks") or []
        decisions_raw = summary.get("decisions") or []
        return SessionSummarySchema(
            title=str(summary.get("title") or ""),
            narrative=str(summary.get("content") or summary.get("narrative") or ""),
            facts=[str(item) for item in facts_raw] if isinstance(facts_raw, list) else [],
            open_tasks=[str(item) for item in tasks_raw] if isinstance(tasks_raw, list) else [],
            decisions=[str(item) for item in decisions_raw] if isinstance(decisions_raw, list) else [],
            session_id=session_id,
        )

    def _candidate_from_item(
        self,
        item: object,
        request: MemoryExtractionRequest,
        expected_kind: MemoryKind,
        default_importance: MemoryImportance,
        structured_summary: SessionSummarySchema | None = None,
    ) -> MemoryCandidate | None:
        if not isinstance(item, dict):
            return None

        content = str(item.get("content") or "").strip()
        if not content:
            return None

        title = str(item.get("title") or "").strip() or None
        importance_str = str(item.get("importance") or "").strip().upper()
        tags_raw = item.get("tags") or []
        tags = tuple(str(tag) for tag in tags_raw) if isinstance(tags_raw, list) else ()

        importance = self._map_importance(importance_str, default_importance)

        return MemoryCandidate(
            content=content,
            kind=expected_kind,
            session_id=request.session_id,
            title=title,
            importance=importance,
            tags=tags,
            source="session_consolidation",
            structured_summary=structured_summary,
            include_episodic_from_summary=False,
        )

    def _map_importance(self, value: str, default: MemoryImportance) -> MemoryImportance:
        if not value:
            return default
        upper = value.upper()
        for level in MemoryImportance:
            if level.name == upper:
                return level
        return default
