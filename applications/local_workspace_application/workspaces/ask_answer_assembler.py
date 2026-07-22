# © Artur Czarnecki. All rights reserved.

"""LKW AskAnswerAssembler — grounded answer from verified search evidence (MVP-2)."""

from __future__ import annotations

import json
import re
from typing import Any

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from local_workspace_application.serving.workspace_schemas import WorkspaceSearchHitV1
from local_workspace_application.workspaces.ask_models import (
    AskAnswerAssemblyError,
    AskAnswerAssemblyResult,
    AskAnswerAssemblyStatus,
    AskCitation,
    AskCitationLocation,
)

_EVIDENCE_ID_RE = re.compile(r"^E([1-9]\d*)$")
_APPROVED_CONTEXT_FIELDS = (
    "document_id",
    "source_id",
    "workspace_id",
    "file_name",
    "source_path",
    "snippet",
    "score",
)

_SYSTEM_PROMPT = """You answer questions using only the indexed evidence provided by the system.
Rules:
1. Answer only from the supplied evidence items (E1, E2, ...).
2. Do not use external knowledge.
3. Do not invent missing facts.
4. If the evidence does not adequately support an answer, return insufficient_evidence.
5. Return the evidence indexes that support material claims.
6. Never invent document paths, document IDs, or source identifiers.
7. Respond with a single JSON object only (no markdown fences) using exactly:
{"status":"completed"|"insufficient_evidence","answer":string|null,"used_evidence_ids":["E1",...]}
When status is insufficient_evidence, answer must be null and used_evidence_ids must be [].
When status is completed, answer must be a non-empty string and used_evidence_ids must contain at least one valid evidence index from the supplied list.
"""


def evidence_index(position: int) -> str:
    """Stable evidence index for 1-based position (E1, E2, ...)."""
    if position < 1:
        raise ValueError("evidence position must be >= 1")
    return f"E{position}"


def index_verified_evidence(
    evidence: list[WorkspaceSearchHitV1],
) -> dict[str, WorkspaceSearchHitV1]:
    """Map stable indexes to verified hits in deterministic input order."""
    return {evidence_index(i): hit for i, hit in enumerate(evidence, start=1)}


def build_indexed_model_context(
    evidence: list[WorkspaceSearchHitV1],
) -> list[dict[str, Any]]:
    """Deterministic model context from approved verified-hit fields only."""
    items: list[dict[str, Any]] = []
    for i, hit in enumerate(evidence, start=1):
        item: dict[str, Any] = {"evidence_id": evidence_index(i)}
        for field in _APPROVED_CONTEXT_FIELDS:
            item[field] = getattr(hit, field)
        location = _approved_location(hit)
        if location is not None:
            item["location"] = location.model_dump(exclude_none=True)
        items.append(item)
    return items


def project_ask_citations(
    *,
    used_evidence_ids: list[str],
    indexed_evidence: dict[str, WorkspaceSearchHitV1],
) -> list[AskCitation]:
    """Project citations from used evidence IDs → verified hits. Model never creates citations."""
    citations: list[AskCitation] = []
    seen: set[str] = set()
    for evidence_id in used_evidence_ids:
        if evidence_id in seen:
            continue
        hit = indexed_evidence.get(evidence_id)
        if hit is None:
            raise AskAnswerAssemblyError(
                "unknown_evidence_reference",
                f"unknown evidence reference: {evidence_id}",
            )
        seen.add(evidence_id)
        citations.append(
            AskCitation(
                evidence_id=evidence_id,
                document_id=hit.document_id,
                source_id=hit.source_id,
                workspace_id=hit.workspace_id,
                source_path=hit.source_path,
                file_name=hit.file_name,
                excerpt=hit.snippet,
                score=hit.score,
                chunk_id=_approved_chunk_id(hit),
                location=_approved_location(hit),
            )
        )
    return citations


def _approved_chunk_id(hit: WorkspaceSearchHitV1) -> str | None:
    raw = hit.metadata.get("chunk_id") if isinstance(hit.metadata, dict) else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _approved_location(hit: WorkspaceSearchHitV1) -> AskCitationLocation | None:
    if not isinstance(hit.metadata, dict):
        return None
    page = hit.metadata.get("page")
    if isinstance(page, bool) or not isinstance(page, int):
        return None
    return AskCitationLocation(page=page)


class AskAnswerAssembler:
    """Bounded grounded-answer assembly owned by the LKW application layer."""

    def __init__(self, llm_adapter: LLMAdapter) -> None:
        self._llm = llm_adapter
        self.model_call_count = 0
        self.last_model_context: list[dict[str, Any]] | None = None

    def assemble(
        self,
        *,
        question: str,
        evidence: list[WorkspaceSearchHitV1],
    ) -> AskAnswerAssemblyResult:
        if not evidence:
            self.last_model_context = []
            return AskAnswerAssemblyResult(
                status=AskAnswerAssemblyStatus.INSUFFICIENT_EVIDENCE,
                answer=None,
                used_evidence_ids=[],
            )

        indexed = index_verified_evidence(evidence)
        context = build_indexed_model_context(evidence)
        self.last_model_context = context

        user_payload = {
            "question": question,
            "evidence": context,
        }
        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
            ),
        ]
        self.model_call_count += 1
        try:
            response = self._llm.generate_messages(
                messages,
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as exc:
            raise AskAnswerAssemblyError(
                "model_invocation_failed",
                f"model invocation failed: {exc.__class__.__name__}",
            ) from exc

        content = str(getattr(response, "content", "") or "").strip()
        parsed = self._parse_model_content(content)
        return self._validate_assembly_result(parsed, indexed_evidence=indexed)

    def _parse_model_content(self, content: str) -> AskAnswerAssemblyResult:
        if not content:
            raise AskAnswerAssemblyError("model_output_empty", "model returned empty content")
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AskAnswerAssemblyError(
                "model_output_invalid",
                "model output is not valid JSON",
            ) from exc
        if not isinstance(payload, dict):
            raise AskAnswerAssemblyError(
                "model_output_invalid",
                "model output must be a JSON object",
            )
        try:
            return AskAnswerAssemblyResult.model_validate(
                {
                    "status": payload.get("status"),
                    "answer": payload.get("answer"),
                    "used_evidence_ids": payload.get("used_evidence_ids") or [],
                }
            )
        except Exception as exc:
            raise AskAnswerAssemblyError(
                "model_output_invalid",
                "model output failed typed validation",
            ) from exc

    def _validate_assembly_result(
        self,
        result: AskAnswerAssemblyResult,
        *,
        indexed_evidence: dict[str, WorkspaceSearchHitV1],
    ) -> AskAnswerAssemblyResult:
        if result.status == AskAnswerAssemblyStatus.INSUFFICIENT_EVIDENCE:
            return AskAnswerAssemblyResult(
                status=AskAnswerAssemblyStatus.INSUFFICIENT_EVIDENCE,
                answer=None,
                used_evidence_ids=[],
            )

        answer = (result.answer or "").strip()
        if not answer:
            raise AskAnswerAssemblyError(
                "completed_without_answer",
                "completed assembly requires a non-empty answer",
            )

        used_ids = [str(item).strip() for item in result.used_evidence_ids if str(item).strip()]
        if not used_ids:
            return AskAnswerAssemblyResult(
                status=AskAnswerAssemblyStatus.INSUFFICIENT_EVIDENCE,
                answer=None,
                used_evidence_ids=[],
            )

        for evidence_id in used_ids:
            if not _EVIDENCE_ID_RE.fullmatch(evidence_id):
                raise AskAnswerAssemblyError(
                    "unknown_evidence_reference",
                    f"unknown evidence reference: {evidence_id}",
                )
            if evidence_id not in indexed_evidence:
                raise AskAnswerAssemblyError(
                    "unknown_evidence_reference",
                    f"unknown evidence reference: {evidence_id}",
                )

        # Projection validates mapping; duplicate IDs collapse.
        citations = project_ask_citations(
            used_evidence_ids=used_ids,
            indexed_evidence=indexed_evidence,
        )
        if not citations:
            raise AskAnswerAssemblyError(
                "completed_without_citation",
                "completed assembly requires at least one verified citation",
            )

        return AskAnswerAssemblyResult(
            status=AskAnswerAssemblyStatus.COMPLETED,
            answer=answer,
            used_evidence_ids=used_ids,
        )
