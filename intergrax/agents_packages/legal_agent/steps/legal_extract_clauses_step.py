# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from intergrax.agents_packages.legal_agent.steps.legal_base_step import LegalBaseStep
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.context.context_builder import RetrievedChunk
from intergrax.agents_packages.legal_agent.legal_agent_state import LegalAgentState


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LegalClause(BaseModel):
    clause_type: str
    text: str
    risk_level: str


class LegalClausesExtractionResult(BaseModel):
    clauses: List[LegalClause] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class LegalExtractClausesStep(LegalBaseStep):

    async def run_step(
        self,
        state: RuntimeState,
        agent_state: LegalAgentState,
    ) -> None:

        # ------------------------------------------------------------------
        # 1. Resolve ingestion service
        # ------------------------------------------------------------------
        service = state.context.ingestion_service
        if service is None:
            raise RuntimeError("AttachmentIngestionService is not configured.")

        req = state.request

        # ------------------------------------------------------------------
        # 2. Ingestion (optional)
        # ------------------------------------------------------------------
        if req.attachments:
            await service.ingest_attachments_for_session(
                attachments=req.attachments,
                session_id=state.request.session_id,
                user_id=state.request.user_id,
                tenant_id=state.tenant_id,
                workspace_id=state.request.workspace_id,
            )

        # ------------------------------------------------------------------
        # 3. Retrieval
        # ------------------------------------------------------------------
        query = req.message or "Analyze legal document and extract clauses."

        search_result = await service.search_session_attachments(
            query=query,
            session_id=state.request.session_id,
            user_id=state.request.user_id,
            tenant_id=state.tenant_id,
            workspace_id=state.request.workspace_id,
            top_k=12,
        )

        hits: List[RetrievedChunk] = search_result.get("hits", [])

        if not hits:
            agent_state.clauses = []
            return

        # ------------------------------------------------------------------
        # 4. LLM processing (per chunk)
        # ------------------------------------------------------------------
        all_clauses: List[LegalClause] = []

        llm = state.context.config.llm_adapter

        for chunk in hits:
            result = await llm.generate_structured(
                prompt=self._build_prompt(chunk.text),
                output_model=LegalClausesExtractionResult,
            )

            if not isinstance(result, LegalClausesExtractionResult):
                raise TypeError("Invalid LLM response type.")

            if result.clauses:
                all_clauses.extend(result.clauses)

        # ------------------------------------------------------------------
        # 5. Save
        # ------------------------------------------------------------------
        agent_state.clauses = all_clauses

        state.trace_event(
            component="LEGAL_AGENT",
            step="LegalExtractClausesStep",
            message=f"Extracted {len(all_clauses)} clauses from {len(hits)} chunks.",
        )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        return f"""
You are a legal analysis system.

Extract all legal clauses from the text below.

For each clause:
- identify clause_type
- extract full clause text
- assign risk_level: low, medium, high

Return structured JSON only.

TEXT:
{text}
"""