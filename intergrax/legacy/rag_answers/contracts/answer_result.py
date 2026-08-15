# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.knowledge.contracts import KnowledgeDocument

from intergrax.legacy.rag_answers.pipeline.pipeline_trace import PipelineTrace


@dataclass(slots=True)
class AnswerResult:
    """
    Result returned by AnswerEngine.
    """

    answer: str    

    context_documents: tuple[KnowledgeDocument, ...]

    pipeline_trace: Optional[PipelineTrace] = None