# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.websearch.schemas.web_search_result import WebSearchResult
from intergrax.websearch.service.websearch_config import WebSearchConfig, WebSearchStrategyType


@dataclass
class WebSearchContextResult:
    context_text: str
    debug_info: Dict[str, Any]


class WebSearchContextGenerator(Protocol):
    async def generate(
        self,
        web_docs: List[WebSearchResult],
        *,
        user_query: Optional[str],
    ) -> WebSearchContextResult:
        ...


def create_websearch_context_generator(cfg: WebSearchConfig) -> WebSearchContextGenerator:
    if cfg.strategy == WebSearchStrategyType.SERP_ONLY:
        return SerpOnlyContextGenerator(cfg)
    if cfg.strategy == WebSearchStrategyType.URL_CONTEXT_TOPK:
        return UrlContextTopKContextGenerator(cfg)
    if cfg.strategy == WebSearchStrategyType.CHUNK_RERANK:
        return ChunkRerankContextGenerator(cfg)
    if cfg.strategy == WebSearchStrategyType.MAP_REDUCE:
        return MapReduceContextGenerator(cfg)
    return SerpOnlyContextGenerator(cfg)


class SerpOnlyContextGenerator:
    def __init__(self, cfg: WebSearchConfig) -> None:
        self._cfg = cfg

    async def generate(
        self,
        web_docs: List[WebSearchResult],
        *,
        user_query: Optional[str],
    ) -> WebSearchContextResult:
        debug: Dict[str, Any] = {
            "strategy": "SERP_ONLY",
            "num_docs": len(web_docs),
        }

        used = web_docs[: self._cfg.max_docs]
        debug["used_docs"] = len(used)
        debug["top_urls"] = [d.url for d in used[:3] if (d.url or "").strip()]

        lines: List[str] = [
            "WEB SOURCES (SERP)\n"
            "- Treat as external context.\n"
            "- Never fabricate facts not supported by sources.\n"
        ]

        for idx, doc in enumerate(used, start=1):
            title = (doc.title or "").strip() or "(no title)"
            url = (doc.url or "").strip()
            snippet = (doc.snippet or "").strip()
            lines.append(f"\n[{idx}] {title}\nURL: {url}\nSnippet: {snippet}")

        return WebSearchContextResult(context_text="\n".join(lines), debug_info=debug)


class UrlContextTopKContextGenerator:
    def __init__(self, cfg: WebSearchConfig) -> None:
        self._cfg = cfg

    async def generate(
        self,
        web_docs: List[WebSearchResult],
        *,
        user_query: Optional[str],
    ) -> WebSearchContextResult:
        debug: Dict[str, Any] = {
            "strategy": "URL_CONTEXT_TOPK",
            "num_docs": len(web_docs),
            "budget_total_tokens": self._cfg.token_budget_total,
            "budget_per_doc_tokens": self._cfg.token_budget_per_doc,
        }

        used = web_docs[: self._cfg.max_docs]
        debug["used_docs"] = len(used)
        debug["top_urls"] = [d.url for d in used[:3] if (d.url or "").strip()]

        remaining = self._cfg.token_budget_total
        injected = 0
        usage: List[Dict[str, Any]] = []

        lines: List[str] = [
            "WEB SOURCES (GROUNDED EXCERPTS)\n"
            "- Prefer facts supported by these excerpts.\n"
            "- Cite URLs when stating concrete facts.\n"
        ]

        for idx, doc in enumerate(used, start=1):
            if remaining <= 0:
                break

            title = (doc.title or "").strip() or "(no title)"
            url = (doc.url or "").strip()

            raw = (doc.text or "").strip()
            if not raw:
                raw = (doc.snippet or "").strip()
            if not raw:
                continue

            allowed = self._cfg.token_budget_per_doc
            if allowed > remaining:
                allowed = remaining

            excerpt, used_tokens = _truncate_to_token_budget(raw, allowed)
            if not excerpt:
                continue

            lines.append(f"\n[{idx}] {title}\nURL: {url}\nEXCERPT:\n{excerpt}")
            remaining -= used_tokens
            injected += 1
            usage.append({"idx": idx, "url": url, "used_tokens": used_tokens, "remaining": remaining})

        debug["injected_docs"] = injected
        debug["budget_remaining_tokens"] = remaining
        debug["usage"] = usage

        return WebSearchContextResult(context_text="\n".join(lines), debug_info=debug)


class ChunkRerankContextGenerator:
    def __init__(self, cfg: WebSearchConfig) -> None:
        self._cfg = cfg

    async def generate(
        self,
        web_docs: List[WebSearchResult],
        *,
        user_query: Optional[str],
    ) -> WebSearchContextResult:
        return WebSearchContextResult(
            context_text=(
                "WEB SOURCES (CHUNK_RERANK)\n"
                "NOTE: Not implemented yet. Use URL_CONTEXT_TOPK for grounded excerpts.\n"
            ),
            debug_info={"strategy": "CHUNK_RERANK", "executed": False},
        )


class MapReduceContextGenerator:
    def __init__(self, cfg: WebSearchConfig) -> None:
        self._cfg = cfg

    async def generate(
        self,
        web_docs: List[WebSearchResult],
        *,
        user_query: Optional[str],
    ) -> WebSearchContextResult:
        if self._cfg.llm.map_adapter is None or self._cfg.llm.reduce_adapter is None:
            raise ValueError("MAP_REDUCE requires WebSearchConfig.llm.map_adapter and llm.reduce_adapter.")

        q = (user_query or "").strip()
        if not q:
            raise ValueError("MAP_REDUCE requires non-empty user_query.")

        debug: Dict[str, Any] = {
            "strategy": "MAP_REDUCE",
            "num_docs": len(web_docs),
        }

        used = web_docs[: self._cfg.max_docs]
        debug["used_docs"] = len(used)
        debug["top_urls"] = [d.url for d in used[:3] if (d.url or "").strip()]

        # -------------------------
        # MAP: per-URL fact cards
        # -------------------------
        fact_cards: List[str] = []
        map_usage: List[Dict[str, Any]] = []

        map_system = (
            "You extract grounded facts from a web page excerpt.\n"
            "Rules:\n"
            "- Use ONLY the provided PAGE_EXCERPT.\n"
            "- If there is no answer-relevant evidence, output exactly: NO_EVIDENCE\n"
            "- Otherwise output 5-10 bullet points.\n"
            "- Each bullet must be a single factual claim.\n"
            "- Keep bullets short and information-dense.\n"
            "- Do NOT add commentary, disclaimers, or additional sections.\n"
            "- Do NOT hallucinate.\n"
            "- Each bullet MUST end with: (Source: <URL>)\n"
        )

        for idx, doc in enumerate(used, start=1):
            title = (doc.title or "").strip() or "(no title)"
            url = (doc.url or "").strip()

            raw = (doc.text or "").strip()
            if not raw:
                raw = (doc.snippet or "").strip()
            if not raw:
                continue

            excerpt, excerpt_tokens = _truncate_to_token_budget(raw, self._cfg.token_budget_per_doc)
            if not excerpt:
                continue

            map_user = (
                f"QUESTION:\n{q}\n\n"
                f"PAGE_TITLE:\n{title}\n\n"
                f"SOURCE_URL:\n{url}\n\n"
                f"PAGE_EXCERPT:\n{excerpt}\n"
            )

            map_messages = [
                ChatMessage(role="system", content=map_system),
                ChatMessage(role="user", content=map_user),
            ]

            # If you don't have temperature/max_tokens fields in cfg.llm yet,
            # call generate_messages(map_messages) without kwargs.
            map_text = self._cfg.llm.map_adapter.generate_messages(
                map_messages,
            )
            map_text = (map_text or "").strip()
            if not map_text or map_text == "NO_EVIDENCE":
                continue

            fact_cards.append(f"[{idx}] {map_text}")
            map_usage.append(
                {
                    "idx": idx,
                    "url": url,
                    "excerpt_tokens": excerpt_tokens,
                }
            )

        debug["map_cards_count"] = len(fact_cards)
        debug["map_usage"] = map_usage

        if not fact_cards:
            debug["executed"] = True
            debug["reduce_skipped_reason"] = "no_fact_cards"
            return WebSearchContextResult(
                context_text=(
                    "WEB SOURCES (MAP_REDUCE)\n"
                    "No answer-relevant evidence extracted from the fetched pages.\n"
                ),
                debug_info=debug,
            )

        # -------------------------
        # REDUCE: synthesis
        # -------------------------
        reduce_system = (
            "You synthesize grounded context from multiple FACT CARDS.\n"
            "Rules:\n"
            "- Use ONLY facts present in FACT_CARDS.\n"
            "- Preserve citations by keeping the [index] markers.\n"
            "- If facts conflict, mention the conflict.\n"
            "- Output format:\n"
            "GROUNDED_CONTEXT:\n"
            "- bullet list of consolidated facts with [index] citations\n"
            "SOURCES:\n"
            "- list of unique Source URLs found in fact cards\n"
        )

        reduce_user = (
            f"QUESTION:\n{q}\n\n"
            "FACT_CARDS:\n"
            + "\n\n".join(fact_cards)
        )

        reduce_messages = [
            ChatMessage(role="system", content=reduce_system),
            ChatMessage(role="user", content=reduce_user),
        ]

        reduce_text = self._cfg.llm.reduce_adapter.generate_messages(
            reduce_messages
        )
        reduce_text = (reduce_text or "").strip()

        debug["executed"] = True
        debug["reduce_chars"] = len(reduce_text)

        context_text = (
            "WEB SOURCES (MAP_REDUCE GROUNDED)\n"
            "- The following context was synthesized from fetched pages.\n"
            "- Cite sources using [index] markers and URLs.\n\n"
            + reduce_text
        )

        return WebSearchContextResult(context_text=context_text, debug_info=debug)



def _approx_tokens(s: str) -> int:
    t = (s or "").strip()
    if not t:
        return 0
    return max(1, int(len(t) / 4))


def _truncate_to_token_budget(s: str, budget_tokens: int) -> Tuple[str, int]:
    t = (s or "").strip()
    if not t or budget_tokens <= 0:
        return "", 0

    approx = _approx_tokens(t)
    if approx <= budget_tokens:
        return t, approx

    char_budget = max(80, int(budget_tokens * 4))
    truncated = t[:char_budget]
    if len(truncated) < len(t):
        truncated = truncated.rstrip() + "..."
    return truncated, min(budget_tokens, _approx_tokens(truncated))
