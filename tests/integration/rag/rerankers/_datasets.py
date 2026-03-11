# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.rerankers.contracts.reranker_types import RerankerCandidate

def candidates() -> list[RerankerCandidate]:

    return [
        RerankerCandidate(
            id="1",
            text="Paris is the capital of France",
            metadata={},
            original_score=0.1,
        ),
        RerankerCandidate(
            id="2",
            text="Bananas are yellow fruits",
            metadata={},
            original_score=0.1,
        ),
        RerankerCandidate(
            id="3",
            text="Berlin is the capital of Germany",
            metadata={},
            original_score=0.1,
        ),
    ]
