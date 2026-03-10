# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TypedDict, Dict, Any, Union

from langchain_core.documents import Document


class RerankerNormalizationMode(str, Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"


class RerankerField(str, Enum):
    CONTENT = "content"
    TEXT = "text"
    PAGE_CONTENT = "page_content"
    METADATA = "metadata"

    ORIGINAL_SCORE = "similarity_score"

    RERANK_SCORE = "rerank_score"
    FUSION_SCORE = "fusion_score"
    RERANK_RANK = "rank_reranked"


class RerankerHit(TypedDict, total=False):

    id: str
    content: str
    text: str
    page_content: str

    metadata: Dict[str, Any]

    similarity_score: float

    rerank_score: float
    fusion_score: float

    rank_reranked: int


@dataclass(slots=True)
class RerankerCandidate:
    """
    Normalized candidate passed to rerankers.
    """

    id: Optional[str]
    text: str
    metadata: Dict[str, Any]
    original_score: Optional[float]


@dataclass(slots=True)
class RerankerResult:
    """
    Result of reranking.
    """

    candidate: RerankerCandidate
    rerank_score: float
    fusion_score: Optional[float]
    rank: int


Candidates = Union[List[RerankerCandidate], List[Document]]