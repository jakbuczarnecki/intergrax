# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations


import logging
from typing import List, Dict, Optional, Tuple

from intergrax.llm.messages import ChatMessage
from intergrax.rag.dual_retriever import DualRetriever
from intergrax.rag.rag_answerer import AnswerSource, RagAnswerer

logger = logging.getLogger("intergrax.windowed_answerer")


class WindowedAnswerer:
    """
    Windowed (map→reduce) layer on top of the base Answerer.
    """
    def __init__(
            self, 
            answerer: RagAnswerer, 
            retriever: DualRetriever, 
            *, 
            verbose: bool = False):
        self.answerer = answerer
        self.retriever = retriever
        self.verbose = verbose
        self.log = logger.getChild("window")
        if self.verbose:
            self.log.setLevel(logging.INFO)

    def _build_context_local(
        self,
        hits: List[Dict],
        max_chars: int,
        per_chunk_cap: int = 4000
    ) -> Tuple[str, List[Dict]]:
        def sanitize(s: str) -> str:
            return "".join(ch for ch in s if ch.isprintable() or ch in "\n\t ").strip()

        parts, used, total = [], [], 0
        for h in hits:
            txt = sanitize(h.get("text") or h.get("content") or "")
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
            used.append(h)
            total += need
            if total >= max_chars:
                break
        return "\n\n---\n\n".join(parts), used

    def _build_messages_for_context(self, question: str, context_text: str):
        """
        Build messages with memory-awareness, without duplicating the system prompt.
        If the answerer has `_build_messages_memory_aware` and a memory store, use it.
        Otherwise, fall back to `_build_messages`.
        """        
        use_memory = self.answerer.memory is not None                
        
        if use_memory :            
            history = list(self.answerer.memory.get_all())
            if not history or history[-1].role != "user" or (history[-1].content or "") != (question or ""):
                history.append(ChatMessage(role="user", content=question))

            # Reuse RagAnswerer logic: insert system if missing + inject context before last user.
            msgs: List[ChatMessage] = list(history)
            has_system = any(m.role == "system" for m in msgs)
            if not has_system:
                msgs.insert(0, ChatMessage(role="system", content=self.answerer.cfg.system_instructions))

            if context_text:
                ctx_msg = ChatMessage(
                    role="system",
                    content=self.answerer.cfg.system_context_template.format(context=context_text),
                )
                if msgs and msgs[-1].role == "user":
                    msgs = msgs[:-1] + [ctx_msg, msgs[-1]]
                else:
                    msgs.append(ctx_msg)

            return msgs
        else:
            # No-memory variant — standard message construction
            return self.answerer._build_messages(
                question=question,
                context_text=context_text,
                user_instruction=None
            )

    def ask_windowed(
        self,
        question: str,
        *,
        top_k_total: int = 60,
        window_size: int = 12,
        summarize_each: bool = False,
        summarize_final: bool = False,
        map_max_tokens: Optional[int] = None,
        reduce_max_tokens: Optional[int] = None,
        source_preview_len: int = 64,
        run_id: Optional[str] = None,
    ):
        if self.verbose:
            self.log.info("[Windowed] Asking: '%s' (top_k_total=%d, window=%d)", question, top_k_total, window_size)

        def _safe_tokens(x: Optional[int], default: int) -> int:
            try:
                v = int(x) if x is not None else int(default)
                return v if v > 0 else int(default)
            except Exception:
                return int(default)

        # # If we have memory, record the user's question ONCE (to keep history consistent).   
        # if self.answerer.memory is not None:                         
        #     self.answerer.memory.add_message(ChatMessage(role="user", content=question))

        # 1) Broad retrieval
        raw_hits = self.retriever.retrieve(question, top_k=top_k_total)

        if self.verbose:
            self.log.info("[Windowed] Retrieved %d candidates", len(raw_hits))

        base_tokens = _safe_tokens(self.answerer.cfg.max_answer_tokens, 1024)

        map_tokens = _safe_tokens(map_max_tokens, base_tokens)
        reduce_tokens = _safe_tokens(reduce_max_tokens, base_tokens)

        if not raw_hits:
            msg = "No sufficiently relevant context was found to answer."
            # If we have memory, append an informational assistant reply.
            if self.answerer.memory is not None:   
                self.answerer.memory.add_message(ChatMessage(role="assistant", content=msg))
            return {
                "answer": msg,
                "sources": [],
                "summary": None,
                "stats": {"windows": 0, "top_k_total": top_k_total, "window_size": window_size},
            }

        # 2) Windows
        windows = [raw_hits[i:i + window_size] for i in range(0, len(raw_hits), window_size)]

        if self.verbose:
            self.log.info("[Windowed] Processing %d windows", len(windows))

        partial_answers = []
        collected_sources: List[AnswerSource] = []

        for wi, w in enumerate(windows, 1):
            ctx_text, used_hits = self._build_context_local(
                w,
                max_chars=self.answerer.cfg.max_context_chars
            )
            self.log.info("[Windowed] Window %d/%d: %d hits", wi, len(windows), len(used_hits))

            # 2a) Build MESSAGES (memory-aware if available)
            msgs = self._build_messages_for_context(question=question, context_text=ctx_text)

            # 2b) LLM → partial answer for this window
            ans = self.answerer.llm.generate_messages(
                msgs,
                temperature=self.answerer.cfg.temperature,
                max_tokens=map_tokens,
                run_id=run_id
            )

            # 2c) (Optional) summarize per-window partial
            if summarize_each:
                sum_msgs = [
                    ChatMessage(role="system", content=self.answerer.cfg.summary_system_instruction),
                    ChatMessage(role="user", content=self.answerer.cfg.summary_prompt_template.format(answer=ans)),
                ]
                ans = self.answerer.llm.generate_messages(
                    sum_msgs, 
                    temperature=self.answerer.cfg.temperature,
                    max_tokens=map_tokens,
                    run_id=run_id
                )

            partial_answers.append(ans)

            # 2d) Collect sources
            collected_sources.extend(
                self.answerer._make_citations([self.answerer._normalize_hit(h) for h in used_hits])
            )

        # 3) Reduce — synthesize final answer from partials
        synthesis_ctx = "\n\n".join(
            [f"WINDOW {i}\n{txt}".strip() for i, txt in enumerate(partial_answers, start=1)]
        )
        msgs_reduce = self._build_messages_for_context(question=question, context_text=synthesis_ctx)
        final_answer = self.answerer.llm.generate_messages(
            msgs_reduce,
            temperature=self.answerer.cfg.temperature,
            max_tokens=reduce_tokens,
            run_id=run_id,
        )

        final_summary = None
        if summarize_final:
            sum_msgs = [
                ChatMessage(role="system", content=self.answerer.cfg.summary_system_instruction),
                ChatMessage(role="user", content=self.answerer.cfg.summary_prompt_template.format(answer=final_answer)),
            ]
            final_summary = self.answerer.llm.generate_messages(
                sum_msgs, 
                temperature=self.answerer.cfg.temperature,
                max_tokens=reduce_tokens,
                run_id=run_id,
            )

        # 4) Deduplicate sources
        seen, dedup_sources = set(), []
        for s in collected_sources:
            key = (s.source, s.page, s.score, (s.preview or "")[:source_preview_len])
            if key in seen:
                continue
            seen.add(key)
            dedup_sources.append(s)

        # 5) If we have memory, append the final answer (and optional summary)        
        if self.answerer.memory is not None:
            content_to_save = final_answer
            if final_summary:
                content_to_save += "\n\n" + final_summary            
            self.answerer.memory.add_message(ChatMessage(role="assistant", content=content_to_save))

        self.log.info("[Windowed] Done (%d windows)", len(windows))
        return {
            "answer": final_answer,
            "sources": dedup_sources,
            "summary": final_summary,
            "stats": {"windows": len(windows), "top_k_total": top_k_total, "window_size": window_size},
        }
