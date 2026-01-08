# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

from intergrax.rag.rag_retriever import RagRetriever
from intergrax.rag.re_ranker import ReRanker

from operator import itemgetter

from intergrax.logging import IntergraxLogging

logger = IntergraxLogging.get_logger(__name__, component="rag")

# How to use:
# 1) You already have:
# store, embed_manager, retriever = intergraxRagRetriever(...), reranker = intergraxReRanker(...) (optional)
# llm: e.g. from langchain_ollama import ChatOllama
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.1:latest", temperature=0.2)

# 2) Hooks (optional)
# def before(payload):
#     # e.g. force an additional filter
#     if payload.get("where") is None:
#         payload["where"] = {}
#     return payload

# def after_prompt(prompt_text, payload):
#     # e.g. append a note for the model
#     return prompt_text + "\n\nNote: Be precise and cite context segments."

# def after_llm(answer_text, payload):
#     # e.g. post-processing
#     return answer_text.strip()

# 3) Configuration
# cfg = intergraxChainConfig(
#     top_k=12,
#     min_score=0.18,
#     use_rerank=True,
#     rerank_k=6,
#     max_context_chars=10_000,
#     on_before_build_prompt=before,
#     on_after_build_prompt=after_prompt,
#     on_after_llm=after_llm,
# )

# 4) Build the chain
# qa_chain = intergraxLangChainQAChain(
#     retriever=retriever,
#     llm=llm,
#     reranker=reranker,   # or None
#     config=cfg,
# )

# 5) Call
# res = qa_chain.invoke("What are intergrax virtual fairs?")
# print(res["answer"])


# ----------------------------
# Helper types
# ----------------------------
BeforeBuildPromptHook = Callable[[Dict[str, Any]], Dict[str, Any]]
AfterBuildPromptHook  = Callable[[str, Dict[str, Any]], str]
AfterLLMHook          = Callable[[str, Dict[str, Any]], str]

PromptBuilder = Union[
    ChatPromptTemplate,
    Callable[[str, str, List[Dict[str, Any]]], str],  # (question, context, hits) -> prompt_text
]


@dataclass
class ChainConfig:
    # Retrieval
    top_k: int = 10
    min_score: float = 0.15
    where: Optional[Dict[str, Any]] = None

    # Re-rank
    use_rerank: bool = False
    rerank_k: int = 5

    # Context
    max_context_chars: int = 12000

    # Prompt
    prompt_builder: Optional[PromptBuilder] = None  # none => default

    # Citations
    meta_source_keys: Tuple[str, ...] = ("source_file", "source", "file_name")
    meta_page_keys: Tuple[str, ...] = ("page", "page_number", "page_index")

    # Hooks (optional)
    on_before_build_prompt: Optional[BeforeBuildPromptHook] = None
    on_after_build_prompt: Optional[AfterBuildPromptHook]   = None
    on_after_llm: Optional[AfterLLMHook]                    = None

    # Return extended data
    return_traces: bool = True


def _default_prompt_builder(question: str, context: str, hits: List[Dict[str, Any]]) -> str:
    """Simple, strict QA prompt using only context."""
    return (
        "You are a careful, factual assistant.\n"
        "Answer ONLY using the information from CONTEXT below.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Use the language of the question.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "FINAL ANSWER:"
    )


def _build_context(hits: List[Dict[str, Any]], max_chars: int) -> Tuple[str, List[Dict[str, Any]]]:
    parts: List[str] = []
    used: List[Dict[str, Any]] = []
    total = 0
    for h in hits:
        txt = (h.get("content") or "").strip()
        if not txt:
            continue
        need = len(txt)
        if total + need > max_chars:
            remain = max(max_chars - total, 0)
            if remain <= 0:
                break
            txt = txt[:remain]
            need = len(txt)
        parts.append(txt)
        used.append(h)
        total += need
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(parts), used


def _format_citations(
    hits: List[Dict[str, Any]],
    meta_source_keys: Tuple[str, ...],
    meta_page_keys: Tuple[str, ...],
) -> str:
    lines: List[str] = []
    for i, h in enumerate(hits, start=1):
        meta = h.get("metadata", {}) or {}
        src = next((meta.get(k) for k in meta_source_keys if meta.get(k)), "unknown")
        page = next((meta.get(k) for k in meta_page_keys if meta.get(k) is not None), None)
        score = h.get("similarity_score")
        chunk_info = ""
        if page is None:
            ci = meta.get("chunk_index")
            ct = meta.get("chunk_total")
            if ci is not None and ct:
                chunk_info = f" | chunk {ci+1}/{ct}"
        page_txt = f" (page {page})" if page is not None else chunk_info
        sc = f" | score={score:.3f}" if isinstance(score, (int, float)) else ""
        lines.append(f"[{i}] {src}{page_txt}{sc}")
    return "\n".join(lines)


class LangChainQAChain:
    """
    Builds a flexible QA chain (RAG → [rerank] → prompt → LLM) LangChain-style,
    with hooks modifying data at stages:
      - on_before_build_prompt(payload): Dict -> Dict
      - on_after_build_prompt(prompt_text, payload): str
      - on_after_llm(answer_text, payload): str

    Input to .invoke / .astream:
      {"question": str, "where": Optional[dict]}

    Output:
      {
        "answer": str,
        "sources": List[dict],
        "prompt": str,
        "raw_hits": List[dict],
        "used_hits": List[dict]
      }
    """

    def __init__(
        self,
        *,
        retriever: RagRetriever,
        llm,  # any LangChain LLM (e.g., ChatOllama, ChatOpenAI, etc.)
        reranker: Optional[ReRanker] = None,
        config: Optional[ChainConfig] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.cfg = config or ChainConfig()

        # Default prompt builder
        if self.cfg.prompt_builder is None:
            self.cfg.prompt_builder = _default_prompt_builder

        # Build Runnable pipeline
        self._chain = self._build_chain()

    # ----------------------
    # Public API
    # ----------------------

    def runnable(self):
        """Returns a LangChain Runnable (you can use .invoke / .astream)."""
        return self._chain

    def invoke(self, question: str, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._chain.invoke({"question": question, "where": where})

    async def ainvoke(self, question: str, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._chain.ainvoke({"question": question, "where": where})

    # ----------------------
    # Chain construction
    # ----------------------

    def _build_chain(self) -> RunnableSequence:
        """
        Creates a sequence:
          1) prepare_payload (RunnableLambda)
          2) retrieve_stage     (RunnableLambda)
          3) rerank_stage       (RunnableLambda, optional)
          4) build_context      (RunnableLambda)
          5) build_prompt       (RunnableLambda)  + hooks before/after
          6) llm                (self.llm)
          7) parse              (StrOutputParser)
          8) package_result     (RunnableLambda)
        """
        # 1) input -> payload
        def prepare_payload(inp: Dict[str, Any]) -> Dict[str, Any]:
            payload = {
                "question": inp.get("question") or "",
                "where": inp.get("where") if inp.get("where") is not None else self.cfg.where,
                "top_k": self.cfg.top_k,
                "min_score": self.cfg.min_score,
            }            
            logger.debug(f"[intergraxChain] Q='{payload['question']}' | top_k={payload['top_k']} | min_score={payload['min_score']}")
            return payload

        # 2) retrieve
        def retrieve_stage(payload: Dict[str, Any]) -> Dict[str, Any]:
            hits = self.retriever.retrieve(
                question=payload["question"],
                top_k=payload["top_k"],
                score_threshold=payload["min_score"],
                where=payload["where"],
            )
            payload["raw_hits"] = hits            
            logger.debug(f"[intergraxChain] Retrieved {len(hits)} hits")
            return payload

        # 3) rerank (optional)
        def rerank_stage(payload: Dict[str, Any]) -> Dict[str, Any]:
            hits = payload.get("raw_hits", [])
            if self.cfg.use_rerank and self.reranker and hits:                
                logger.debug(f"[intergraxChain] Reranking to top {self.cfg.rerank_k}")
                payload["raw_hits"] = self.reranker.rerank_candidates(
                    query=payload["question"],
                    candidates=hits,
                    rerank_k=self.cfg.rerank_k,
                )
            return payload

        # 4) build context
        def build_context_stage(payload: Dict[str, Any]) -> Dict[str, Any]:
            hits = payload.get("raw_hits", [])
            context, used = _build_context(hits, self.cfg.max_context_chars)
            payload["context"] = context
            payload["used_hits"] = used
            return payload

        # 5) build prompt (with hooks)
        def build_prompt_stage(payload: Dict[str, Any]) -> Dict[str, Any]:
            # hook: on_before_build_prompt (e.g., enrich payload)
            if callable(self.cfg.on_before_build_prompt):
                try:
                    payload = self.cfg.on_before_build_prompt(payload) or payload
                except Exception as e:
                    print(f"[intergraxChain] on_before_build_prompt error: {e}")

            question = payload["question"]
            context  = payload.get("context", "")
            used     = payload.get("used_hits", [])

            # builder can be ChatPromptTemplate or callable
            prompt_builder = self.cfg.prompt_builder
            if isinstance(prompt_builder, ChatPromptTemplate):
                # ChatPromptTemplate -> text (single-turn)
                prompt_text = prompt_builder.format(question=question, context=context)
            else:
                # callable(question, context, hits) -> str
                prompt_text = prompt_builder(question, context, used)

            # hook: on_after_build_prompt (e.g., add headers, instructions, system msg)
            if callable(self.cfg.on_after_build_prompt):
                try:
                    prompt_text = self.cfg.on_after_build_prompt(prompt_text, payload) or prompt_text
                except Exception as e:
                    print(f"[intergraxChain] on_after_build_prompt error: {e}")

            payload["prompt_text"] = prompt_text
            return payload

        # 6) LLM call → returns pure text (use StrOutputParser)
        def prompt_to_llm(payload: Dict[str, Any]) -> str:
            return payload["prompt_text"]

        # 7) Post-LLM hook and packaging
        def package_result(answer_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            # hook: on_after_llm
            if callable(self.cfg.on_after_llm):
                try:
                    answer_text = self.cfg.on_after_llm(answer_text, payload) or answer_text
                except Exception as e:
                    print(f"[intergraxChain] on_after_llm error: {e}")

            used_hits = payload.get("used_hits", [])
            citations = _format_citations(
                hits=used_hits,
                meta_source_keys=self.cfg.meta_source_keys,
                meta_page_keys=self.cfg.meta_page_keys,
            )
            final_answer = answer_text
            if citations:
                final_answer = f"{answer_text}\n\nCitations:\n{citations}"

            out = {
                "answer": final_answer,
                "sources": used_hits,
                "prompt": payload.get("prompt_text", ""),
                "raw_hits": payload.get("raw_hits", []) if self.cfg.return_traces else None,
                "used_hits": used_hits if self.cfg.return_traces else None,
            }
            return out

        # Build Runnable
        chain: RunnableSequence = (
            RunnableLambda(prepare_payload)
            | RunnableLambda(retrieve_stage)
            | RunnableLambda(rerank_stage)
            | RunnableLambda(build_context_stage)
            | RunnableLambda(build_prompt_stage)
            # ↓ KEY CHANGE: payload = full input dictionary at this stage
            | RunnableMap({
                "llm_out": itemgetter("prompt_text") | self.llm,   # LLM receives only the string
                "payload": RunnableLambda(lambda x: x),            # preserve all other data
            })
            | RunnableMap({
                "answer_text": itemgetter("llm_out") | StrOutputParser(),
                "payload": itemgetter("payload"),
            })
            | RunnableLambda(lambda d: package_result(d["answer_text"], d["payload"]))
        )
        return chain
