# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.retrieval.retrieval_request import RetrievalRequest

pytestmark = pytest.mark.unit


def test_resolved_prefetch_uses_profile_when_final_top_k_set() -> None:
    req = RetrievalRequest(query="q", final_top_k=5)
    assert req.resolved_final_k(profile_final=8) == 5
    assert req.resolved_prefetch_k(profile_prefetch=20, final_k=5) == 20


def test_resolved_prefetch_bumps_when_below_final() -> None:
    req = RetrievalRequest(query="q", final_top_k=15, prefetch_k=5)
    assert req.resolved_prefetch_k(profile_prefetch=20, final_k=15) == 20
