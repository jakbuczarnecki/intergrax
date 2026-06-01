# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""LLM-based entity/edge extraction for GraphRAG (injected adapter — no fixed provider)."""

from __future__ import annotations

import json
import re
import uuid
from typing import List, Sequence

from langchain_core.documents import Document

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore


class LlmGraphIndexer:
    def __init__(self, store: GraphStore, llm: LLMAdapter) -> None:
        self._store = store
        self._llm = llm

    def index_documents(self, documents: Sequence[Document], *, chunk_ids: Sequence[str] | None = None) -> int:
        ids = list(chunk_ids) if chunk_ids else [str(uuid.uuid4()) for _ in documents]
        total = 0
        for doc, chunk_id in zip(documents, ids):
            total += self._index_one(doc, chunk_id)
        return total

    def _index_one(self, doc: Document, chunk_id: str) -> int:
        text = (doc.page_content or "").strip()
        if len(text) < 20:
            return 0

        prompt = (
            "Extract entities and relations from the text. "
            'Return JSON: {"entities":[{"label":"..."}],"relations":[{"source":"...","target":"...","relation":"..."}]}\n\n'
            f"TEXT:\n{text[:4000]}"
        )
        try:
            response = self._llm.generate_messages(
                [ChatMessage(role="user", content=prompt)],
                run_id="rag-graph-index",
            )
            payload = _parse_json_block(response.content or "")
        except Exception:
            return 0

        entities = payload.get("entities") or []
        relations = payload.get("relations") or []
        if not entities:
            return 0

        node_ids: List[str] = []
        label_to_id: dict[str, str] = {}
        for item in entities[:15]:
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            node_id = f"ent:{label.lower().replace(' ', '_')}"
            label_to_id[label.lower()] = node_id
            self._store.upsert_node(GraphNode(id=node_id, label=label, node_type="entity"))
            self._store.link_chunk(node_id, chunk_id)
            node_ids.append(node_id)

        for rel in relations[:20]:
            src_label = str(rel.get("source", "")).strip().lower()
            tgt_label = str(rel.get("target", "")).strip().lower()
            if src_label not in label_to_id or tgt_label not in label_to_id:
                continue
            self._store.upsert_edge(
                GraphEdge(
                    source_id=label_to_id[src_label],
                    target_id=label_to_id[tgt_label],
                    relation=str(rel.get("relation", "related_to")),
                )
            )
        return len(node_ids)


def _parse_json_block(content: str) -> dict:
    content = content.strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
