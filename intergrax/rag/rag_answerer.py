# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol, Iterable, Literal, Type

from intergrax.llm.conversational_memory import IntergraxConversationalMemory, ChatMessage
from intergrax.llm_adapters import LLMAdapter

# Pydantic optionally (no hard runtime dependency)
try:
    from pydantic import BaseModel  # type: ignore
except Exception:
    class BaseModel:  # fallback
        pass


# =========================
# Configuration & data models
# =========================

@dataclass
class IntergraxAnswererConfig:
    # Retrieval / ranking
    top_k: int = 12
    min_score: Optional[float] = None
    re_rank_k: int = 0
    max_context_chars: int = 12000

    # LLM
    temperature: float = 0.2
    max_answer_tokens: Optional[int] = None

    # Citations
    meta_source_keys: Tuple[str, ...] = (
        "source_id", "source_path", "source_file", "source", "file_name", "path", "filename", "title"
    )
    meta_page_keys: Tuple[str, ...] = ("page", "page_number", "page_index", "page_no")

    # Prompts
    system_instructions: str = (
        "You are a factual, precise assistant. Base your answer ONLY on the provided context if present.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Answer in user's language, concise but complete. If useful, cite sources like [source|page]."
    )
    system_context_template: str = "Context for this session:\n{context}"
    user_question_template: str = "Question:\n{question}"
    user_instruction_template: str = "{instruction}"

    summary_prompt_template: str = (
        "Summarize the answer below into no more than 6 bullet points. Keep it factual and do not introduce new content.\n\nANSWER:\n{answer}\n"
    )


@dataclass
class AnswerSource:
    source: str
    page: Optional[str]
    score: Optional[float]
    preview: str


# =========================
# Main Answerer (messages & roles)
# =========================

class IntergraxRagAnswerer:
    def __init__(
        self,
        retriever: Any,
        llm: LLMAdapter,
        reranker: Optional[Any] = None,
        config: Optional[IntergraxAnswererConfig] = None,
        verbose: bool = False,
        *,
        memory: Optional[IntergraxConversationalMemory] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.cfg = config or IntergraxAnswererConfig()
        self.verbose = verbose
        self.memory = memory

    # ---------- Public API ----------

    def run(
        self,
        question: str,
        *,
        where: Optional[dict] = None,
        stream: bool = False,
        summarize: bool = False,
        user_instruction: Optional[str] = None,
        output_model: Optional[Type[BaseModel]] = None,   # ← like in intergraxToolsAgent: additional structured output next to text
    ) -> Dict[str, Any]:
        """
        Behavior identical to intergraxToolsAgent.run:
        - 'answer': textual LLM answer (string),
        - 'output_structure': Pydantic instance (if output_model is provided and the adapter supports generate_structured),
        - no 'structured' field.
        """
        if self.memory is not None:
            self.memory.add_message(role="user", content=question)

        tk = self.cfg.top_k
        ms = self.cfg.min_score

        # 1) Retrieval
        if self.verbose:
            print(f"[intergraxRagAnswerer] Retrieve: top_k={tk}, min_score={ms}")
        t0 = time.perf_counter()
        # ⬇️ remove unsupported score_threshold argument (we apply the threshold locally)
        hits_raw = self.retriever.retrieve(question=question, top_k=tk, where=where)
        t_retrieve = time.perf_counter() - t0
        hits = [self._normalize_hit(h) for h in hits_raw]

        # Local similarity threshold filter (if configured)
        if ms is not None:
            try:
                thr = float(ms)
                hits = [h for h in hits if (h.get("similarity_score") or 0.0) >= thr]
            except Exception:
                # if parsing min_score fails, skip the filter
                pass

        if not hits:
            msg = "No sufficiently relevant context fragments were found to answer the question."
            return {
                "answer": msg,
                "output_structure": None,
                "sources": [],
                "summary": None,
                "context": "",
                "messages": [],
                "stats": {
                    "hits_in": 0,
                    "context_chars": 0,
                    "retrieve_s": round(t_retrieve, 4),
                    "rerank_s": 0.0,
                    "llm_s": 0.0,
                },
            }

        # 2) (Optional) Re-rank
        t_rerank = 0.0
        if self.reranker and (self.cfg.re_rank_k or 0) > 0:
            if self.verbose:
                print(f"[intergraxRagAnswerer] Re-rank to top {self.cfg.re_rank_k}")
            t1 = time.perf_counter()
            rr_hits = None
            try:
                if hasattr(self.reranker, "__call__"):
                    rr_hits = self.reranker(question=question, candidates=hits, top_k=self.cfg.re_rank_k)
                elif hasattr(self.reranker, "rerank_candidates"):
                    rr_hits = self.reranker.rerank_candidates(question, hits, rerank_k=self.cfg.re_rank_k)
                else:
                    rr_hits = hits
            except TypeError:
                rr_hits = self.reranker(question, hits, self.cfg.re_rank_k)
            hits = [self._normalize_hit(h) for h in rr_hits]
            t_rerank = time.perf_counter() - t1

        # 3) Build context
        context_text, used_hits = self._build_context(hits, self.cfg.max_context_chars, per_chunk_cap=4000)
        sources = self._make_citations(used_hits)

        # 4) Build messages (system + user)
        if self.memory is not None:
            messages = self._build_messages_memory_aware(context_text)
        else:
            messages = self._build_messages(question, context_text, user_instruction=user_instruction)

        if self.verbose:
            print(f"[intergraxRagAnswerer] Sending message to LLM: {messages}")

        # 5) LLM — always generate TEXT (answer)
        t_llm = 0.0
        if stream:
            if self.verbose:
                print("[intergraxRagAnswerer] Streaming answer...")
            t2 = time.perf_counter()
            parts: List[str] = []
            for piece in self.llm.stream_messages(
                messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_answer_tokens,
            ):
                parts.append(piece or "")
            answer = "".join(parts)
            t_llm = time.perf_counter() - t2
        else:
            if self.verbose:
                print("[intergraxRagAnswerer] Generating answer...")
            t2 = time.perf_counter()
            answer = self.llm.generate_messages(
                messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_answer_tokens,
            )
            t_llm = time.perf_counter() - t2

        # 6) (Optional) output_structure — like in intergraxToolsAgent
        output_structure_obj: Optional[BaseModel] = None
        if (output_model is not None) and (not stream) and hasattr(self.llm, "generate_structured"):
            try:
                # Reuse the same messages (which include the context) to force a schema matching output_model
                output_structure_obj = self.llm.generate_structured(
                    messages,
                    output_model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_answer_tokens,
                )
                # (Optional) write to memory as a JSON log
                if self.memory is not None:
                    if hasattr(output_structure_obj, "model_dump"):
                        payload = output_structure_obj.model_dump()
                    elif hasattr(output_structure_obj, "dict"):
                        payload = output_structure_obj.dict()
                    else:
                        payload = dict(output_structure_obj)
                    self.memory.add_message(role="assistant", content=json.dumps(payload, ensure_ascii=False))
            except Exception:
                # if anything fails, keep the text path only
                output_structure_obj = None

        # 7) (Optional) textual summary
        summary = None
        if summarize:
            summary_msgs = [
                ChatMessage(role="system", content="You summarize answers without adding facts."),
                ChatMessage(role="user", content=self.cfg.summary_prompt_template.format(answer=answer)),
            ]
            try:
                summary = self.llm.generate_messages(summary_msgs, temperature=0.0, max_tokens=512)
            except Exception:
                summary = None

        # 8) Memory — store TEXT answer (+ optional summary)
        if self.memory is not None:
            llm_content = answer
            if summary:
                llm_content += "\n\n" + summary
            self.memory.add_message(role="assistant", content=llm_content)

        # 9) Telemetry + return (same shape as intergraxToolsAgent)
        return {
            "answer": answer,                   # ← TEXT
            "output_structure": output_structure_obj,  # ← Pydantic instance or None
            "sources": sources,
            "summary": summary,
            "context": context_text,
            "messages": messages,
            "stats": {
                "hits_in": len(hits),
                "context_chars": len(context_text),
                "retrieve_s": round(t_retrieve, 4),
                "rerank_s": round(t_rerank, 4),
                "llm_s": round(t_llm, 4),
            },
        }

    # ---------- Helpers ----------

    def _build_messages_memory_aware(
        self,
        context_text: str,
    ) -> List[ChatMessage]:
        """
        Builds messages using conversation memory:
        - takes the full history from memory,
        - if the history lacks a system prompt, inserts it at the BEGINNING (instructions only),
        - injects the CURRENT context as a one-off 'system' message directly BEFORE the last 'user' message (if present), otherwise at the end.
        """
        assert self.memory is not None, "memory is required"

        history = self.memory.get_all()  # already contains the current user question
        msgs: List[ChatMessage] = list(history)

        has_system = any(m.role == "system" for m in msgs)
        if not has_system:
            msgs.insert(0, ChatMessage(role="system", content=self.cfg.system_instructions))

        if context_text:
            ctx_msg = ChatMessage(
                role="system",
                content=self.cfg.system_context_template.format(context=context_text)
            )
            if len(msgs) >= 1 and msgs[-1].role == "user":
                msgs = msgs[:-1] + [ctx_msg, msgs[-1]]
            else:
                msgs.append(ctx_msg)

        return msgs

    def _build_messages(
        self,
        question: str,
        context_text: str,
        *,
        user_instruction: Optional[str] = None
    ) -> List[ChatMessage]:
        # 1) SYSTEM
        if context_text:
            system_content = (
                f"{self.cfg.system_instructions}\n\n"
                f"{self.cfg.system_context_template.format(context=context_text)}"
            )
        else:
            system_content = self.cfg.system_instructions

        messages: List[ChatMessage] = [ChatMessage(role="system", content=system_content)]

        # 2) USER
        user_parts = [self.cfg.user_question_template.format(question=question)]
        if user_instruction:
            user_parts.append(self.cfg.user_instruction_template.format(instruction=user_instruction))

        messages.append(ChatMessage(role="user", content="\n\n".join(p.strip() for p in user_parts if p and p.strip())))
        return messages

    def _sanitize_text(self, s: str) -> str:
        return "".join(ch for ch in s if ch.isprintable() or ch in "\n\t ").strip()

    def _build_context(self, hits: List[Dict[str, Any]], max_chars: int, per_chunk_cap: int = 4000) -> Tuple[str, List[Dict[str, Any]]]:
        parts: List[str] = []
        used_hits: List[Dict[str, Any]] = []
        total = 0
        for h in hits:
            txt = self._sanitize_text((h.get("content") or "").strip())
            if not txt:
                continue
            if per_chunk_cap and len(txt) > per_chunk_cap:
                txt = txt[:per_chunk_cap]
            need = len(txt)
            if total + need > max_chars:
                remain = max(max_chars - total, 0)
                if remain == 0:
                    break
                txt = txt[:remain]
                need = len(txt)
            parts.append(txt)
            used_hits.append(h)
            total += need
            if total >= max_chars:
                break
        return "\n\n---\n\n".join(parts), used_hits

    def _make_citations(self, hits: List[Dict[str, Any]]) -> List[AnswerSource]:
        out: List[AnswerSource] = []
        for h in hits:
            meta = h.get("metadata", {}) or {}
            # source
            source = None
            for k in self.cfg.meta_source_keys:
                v = meta.get(k)
                if v:
                    source = v
                    break
            source = source or "unknown"
            # page
            page = None
            for k in self.cfg.meta_page_keys:
                if meta.get(k) is not None:
                    page = str(meta.get(k))
                    break
            # preview
            pv = (h.get("content") or "").strip()
            if len(pv) > 300:
                cut = pv[:300]
                last_space = cut.rfind(" ")
                pv = (cut if last_space < 0 else cut[:last_space]) + "..."
            out.append(AnswerSource(source=source, page=page, score=h.get("similarity_score"), preview=pv))
        return out

    def _normalize_hit(self, h: Any) -> Dict[str, Any]:
        # 1) dict
        if isinstance(h, dict):
            content = h.get("content") or h.get("text") or h.get("page_content") or ""
            meta = h.get("metadata") or {}
            score = h.get("similarity_score") or h.get("score")
            return {"content": content, "metadata": meta, "similarity_score": score}

        # 2) LangChain Document
        try:
            from langchain_core.documents import Document
            if isinstance(h, Document):
                return {
                    "content": getattr(h, "page_content", "") or "",
                    "metadata": dict(getattr(h, "metadata", {}) or {}),
                    "similarity_score": None,
                }
        except Exception:
            pass

        # 3) DocHit (your internal shape: .doc, .score)
        if hasattr(h, "doc") and hasattr(h, "score"):
            doc = getattr(h, "doc")
            meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or ""
            return {
                "content": content,
                "metadata": dict(meta or {}),
                "similarity_score": float(getattr(h, "score", 0.0)),
            }

        # 4) tuple (Document, score)
        if isinstance(h, (tuple, list)) and len(h) == 2:
            doc, score = h
            meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or ""
            return {
                "content": content or "",
                "metadata": dict(meta or {}),
                "similarity_score": float(score),
            }

        # fallback
        return {"content": str(h), "metadata": {}, "similarity_score": None}
