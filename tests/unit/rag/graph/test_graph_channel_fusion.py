# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.retrieval.graph_channel_fusion import (
    GraphChannelHit,
    build_keyword_hits,
    fuse_graph_channels,
    lexical_score,
)


@pytest.mark.gate
def test_lexical_score_matches_query_tokens() -> None:
    assert lexical_score("Nimbus Analytics", "Nimbus Analytics platform") > 0.5


@pytest.mark.gate
def test_fuse_graph_channels_merges_vector_keyword_graph() -> None:
    result = fuse_graph_channels(
        vector_hits=[GraphChannelHit("doc-a", 0.9, "vector")],
        keyword_hits=[GraphChannelHit("doc-b", 0.8, "keyword")],
        graph_hits=[GraphChannelHit("doc-c", 0.7, "graph")],
        top_k=3,
    )
    assert set(result.merged_document_ids) == {"doc-a", "doc-b", "doc-c"}
    assert result.channel_contributions["vector"] == ["doc-a"]
    assert result.channel_contributions["keyword"] == ["doc-b"]
    assert result.channel_contributions["graph"] == ["doc-c"]


@pytest.mark.gate
def test_build_keyword_hits_skips_zero_overlap() -> None:
    hits = build_keyword_hits(
        query_text="alpha beta",
        candidates=[("doc-x", "gamma delta")],
    )
    assert hits == []
