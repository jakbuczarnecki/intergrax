# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
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


def create_websearch_context_generator(
    cfg: WebSearchConfig,
    *,
    prompt_registry: Optional[YamlPromptRegistry] = None,
) -> WebSearchContextGenerator:
    
    registry = prompt_registry or YamlPromptRegistry.create_default(load=True)

    if cfg.strategy == WebSearchStrategyType.SERP_ONLY:
        return SerpOnlyContextGenerator(cfg, registry)
    if cfg.strategy == WebSearchStrategyType.URL_CONTEXT_TOPK:
        return UrlContextTopKContextGenerator(cfg, registry)
    if cfg.strategy == WebSearchStrategyType.CHUNK_RERANK:
        return ChunkRerankContextGenerator(cfg, registry)
    if cfg.strategy == WebSearchStrategyType.MAP_REDUCE:
        return MapReduceContextGenerator(cfg, registry)

    return SerpOnlyContextGenerator(cfg, registry)



class SerpOnlyContextGenerator:
    def __init__(
            self, 
            cfg: WebSearchConfig, 
            prompt_registry: YamlPromptRegistry,
    ) -> None:
        self._cfg = cfg
        self._prompt_registry = prompt_registry

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

        localized = self._prompt_registry.resolve_localized(
            prompt_id="websearch_serp_context"
        )

        lines: List[str] = [localized.system]

        for idx, doc in enumerate(used, start=1):
            title = (doc.title or "").strip() or "(no title)"
            url = (doc.url or "").strip()
            snippet = (doc.snippet or "").strip()
            lines.append(f"\n[{idx}] {title}\nURL: {url}\nSnippet: {snippet}")

        return WebSearchContextResult(context_text="\n".join(lines), debug_info=debug)


class UrlContextTopKContextGenerator:
    def __init__(
        self,
        cfg: WebSearchConfig,
        prompt_registry: YamlPromptRegistry,
    ) -> None:
        self._cfg = cfg
        self._prompt_registry = prompt_registry

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

        lines: List[str] = []
        localized = self._prompt_registry.resolve_localized(
            prompt_id="websearch_grounded_context"
        )

        lines: List[str] = [localized.system]

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
    def __init__(
        self,
        cfg: WebSearchConfig,
        prompt_registry: YamlPromptRegistry,
    ) -> None:
        self._cfg = cfg
        self._prompt_registry = prompt_registry

    async def generate(
        self,
        web_docs: List[WebSearchResult],
        *,
        user_query: Optional[str],
    ) -> WebSearchContextResult:
        localized = self._prompt_registry.resolve_localized(
            prompt_id="websearch_chunk_rerank_notice"
        )

        return WebSearchContextResult(
            context_text=localized.system,
            debug_info={"strategy": "CHUNK_RERANK", "executed": False},
        )


class MapReduceContextGenerator:
    def __init__(
        self,
        cfg: WebSearchConfig,
        prompt_registry: YamlPromptRegistry,
    ) -> None:
        self._cfg = cfg
        self._prompt_registry = prompt_registry

    async def generate(
        self,
        web_docs: List[WebSearchResult],
        *,
        user_query: Optional[str],
    ) -> WebSearchContextResult:

        if (
            self._cfg.llm.map_adapter is None
            or self._cfg.llm.reduce_adapter is None
        ):
            raise ValueError(
                "MAP_REDUCE requires WebSearchConfig.llm.map_adapter "
                "and llm.reduce_adapter."
            )

        q = (user_query or "").strip()
        if not q:
            raise ValueError("MAP_REDUCE requires non-empty user_query.")

        debug: Dict[str, Any] = {
            "strategy": "MAP_REDUCE",
            "num_docs": len(web_docs),
        }

        used = web_docs[: self._cfg.max_docs]
        debug["used_docs"] = len(used)
        debug["top_urls"] = [
            d.url for d in used[:3] if (d.url or "").strip()
        ]

        # ==========================================================
        # MAP PHASE
        # ==========================================================

        fact_cards: List[str] = []
        map_usage: List[Dict[str, Any]] = []

        # --- system prompt from registry ---
        map_system = self._prompt_registry.resolve_localized(
            prompt_id="websearch_map_system"
        ).system

        for idx, doc in enumerate(used, start=1):
            title = (doc.title or "").strip() or "(no title)"
            url = (doc.url or "").strip()

            raw = (doc.text or "").strip()
            if not raw:
                raw = (doc.snippet or "").strip()
            if not raw:
                continue

            excerpt, excerpt_tokens = _truncate_to_token_budget(
                raw,
                self._cfg.token_budget_per_doc,
            )
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

            map_text = self._cfg.llm.map_adapter.generate_messages(
                map_messages,
                run_id=self._cfg.run_id,
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

        # --- no evidence branch from registry ---
        if not fact_cards:
            debug["executed"] = True
            debug["reduce_skipped_reason"] = "no_fact_cards"

            no_evidence_text = self._prompt_registry.resolve_localized(
                prompt_id="websearch_mapreduce_no_evidence"
            ).system

            return WebSearchContextResult(
                context_text=no_evidence_text,
                debug_info=debug,
            )

        # ==========================================================
        # REDUCE PHASE
        # ==========================================================

        reduce_system = self._prompt_registry.resolve_localized(
            prompt_id="websearch_reduce_system"
        ).system

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
            reduce_messages,
            run_id=self._cfg.run_id,
        )

        reduce_text = (reduce_text or "").strip()

        debug["executed"] = True
        debug["reduce_chars"] = len(reduce_text)

        # --- final header from registry ---
        header = self._prompt_registry.resolve_localized(
            prompt_id="websearch_mapreduce_final_header"
        ).system

        context_text = f"{header}\n\n{reduce_text}"

        return WebSearchContextResult(
            context_text=context_text,
            debug_info=debug,
        )




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
