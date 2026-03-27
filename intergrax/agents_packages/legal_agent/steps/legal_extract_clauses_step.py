# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
from typing import List

from pydantic import BaseModel, Field

from intergrax.agents_packages.legal_agent.prompts.legal_agent_llm_prompts import (
    DEFAULT_RAG_QUERY_FOR_CLAUSE_EXTRACTION,
    EXTRACT_CLAUSES_SYSTEM,
    extract_clauses_chunk_user,
)
from intergrax.agents_packages.legal_agent.steps.base.legal_base_step import LegalBaseStep
from intergrax.agents_packages.legal_agent.domain.legal_agent_state import Clause, LegalAgentState
from intergrax.agents_packages.legal_agent.tracing.legal_extract_clauses_step_diag_v1 import (
    LegalExtractClausesStepDiagV1,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.context_builder import RetrievedChunk
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


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
        query = req.message or DEFAULT_RAG_QUERY_FOR_CLAUSE_EXTRACTION

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
            agent_state.clause_extraction_retrieval_outcome = "no_hits"
            agent_state.clauses = []
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalExtractClausesStep",
                message="No retrieval hits; clauses cleared.",
                level=TraceLevel.INFO,
                payload=LegalExtractClausesStepDiagV1(
                    step_name="LegalExtractClausesStep",
                    outcome="no_hits",
                    retrieval_chunks_count=0,
                    clauses_extracted_count=0,
                    llm_calls_count=0,
                    attachments_ingested=bool(req.attachments),
                    pre_flagged_sensitive_clauses_count=0,
                ),
            )
            return

        agent_state.clause_extraction_retrieval_outcome = "hits"
        state.used_rag = True

        # ------------------------------------------------------------------
        # 4. LLM processing (per chunk)
        # ------------------------------------------------------------------
        all_clauses: List[Clause] = []

        llm = state.context.config.llm_adapter

        for chunk in hits:
            messages = [
                ChatMessage(role="system", content=EXTRACT_CLAUSES_SYSTEM),
                ChatMessage(
                    role="user",
                    content=extract_clauses_chunk_user(chunk_text=chunk.text),
                ),
            ]
            result = llm.generate_structured(
                    messages,
                    LegalClausesExtractionResult,
                    run_id=state.run_id,
                )

            if not isinstance(result, LegalClausesExtractionResult):
                raise TypeError("Invalid LLM response type.")

            if result.clauses:
                for lc in result.clauses:
                    all_clauses.append(
                        Clause(
                            id=uuid.uuid4().hex,
                            text=lc.text,
                            category=lc.clause_type,
                            is_sensitive=lc.risk_level.lower() == "high",
                        )
                    )

        # ------------------------------------------------------------------
        # 5. Save
        # ------------------------------------------------------------------
        agent_state.clauses = all_clauses

        sensitive_n = sum(1 for c in all_clauses if c.is_sensitive)
        state.trace_event(
            component=TraceComponent.STEP,
            step="LegalExtractClausesStep",
            message=f"Extracted {len(all_clauses)} clauses from {len(hits)} chunks.",
            level=TraceLevel.INFO,
            payload=LegalExtractClausesStepDiagV1(
                step_name="LegalExtractClausesStep",
                outcome="extracted",
                retrieval_chunks_count=len(hits),
                clauses_extracted_count=len(all_clauses),
                llm_calls_count=len(hits),
                attachments_ingested=bool(req.attachments),
                pre_flagged_sensitive_clauses_count=sensitive_n,
            ),
        )