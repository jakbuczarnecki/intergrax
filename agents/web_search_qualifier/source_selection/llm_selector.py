# © Artur Czarnecki. All rights reserved.

"""LLM-backed source selection collaborator for SourceSelectionEngine."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from web_search_qualifier.source_selection.matching import match_url_from_response
from web_search_qualifier.web_search import WebSearchCandidate


@dataclass(frozen=True, slots=True)
class LLMSourceSelectionResult:
    selected_url: str | None
    raw_response: str


@dataclass(frozen=True, slots=True)
class LLMSourceSelector:
    adapter: LLMAdapter
    system_prompt: str

    def select(
        self,
        *,
        run_id: str,
        task_message: str,
        candidates: tuple[WebSearchCandidate, ...],
        highlight_top_rank: bool = False,
    ) -> LLMSourceSelectionResult:
        if not candidates:
            return LLMSourceSelectionResult(selected_url=None, raw_response="")
        response = self.adapter.generate_messages(
            [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        f"Task: {task_message}\n\nCandidates:\n"
                        f"{_format_candidates(candidates, highlight_top_rank=highlight_top_rank)}"
                    ),
                ),
            ],
            temperature=0.0,
            run_id=run_id,
        )
        raw = response.content.strip()
        selected = match_url_from_response(raw, candidates)
        return LLMSourceSelectionResult(selected_url=selected, raw_response=raw)


def _format_candidates(
    candidates: tuple[WebSearchCandidate, ...],
    *,
    highlight_top_rank: bool,
) -> str:
    lines: list[str] = []
    for candidate in candidates:
        rank_note = ""
        if highlight_top_rank and candidate.rank == 1:
            rank_note = " [highest-ranked candidate]"
        lines.append(
            f"{candidate.rank}. {candidate.title}{rank_note}\n"
            f"URL: {candidate.url}\n"
            f"Snippet: {candidate.snippet[:300]}",
        )
    return "\n\n".join(lines)


__all__ = ["LLMSourceSelectionResult", "LLMSourceSelector"]
