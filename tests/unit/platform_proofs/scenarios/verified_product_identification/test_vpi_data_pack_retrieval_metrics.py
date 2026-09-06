"""Unit tests for single-relevant retrieval metrics."""

from __future__ import annotations

import math

import pytest

from platform_proofs.scenarios.verified_product_identification.arena.evaluation.metrics import (
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.retrieval_metrics import (
    evaluate_single_relevant_ranking,
    locate_single_relevant_rank,
    single_relevant_mrr_at_k,
    single_relevant_ndcg_at_k,
    single_relevant_recall_at_k,
)

pytestmark = pytest.mark.unit

EXPECTED_ID = "offer-expected"
RANKED_AT_4 = ("offer-a", "offer-b", "offer-c", EXPECTED_ID, "offer-e")


def _ranked_with_expected_at(one_based_rank: int) -> tuple[str, ...]:
    filler = tuple(f"offer-filler-{index}" for index in range(one_based_rank - 1))
    return filler + (EXPECTED_ID,) + tuple(
        f"offer-tail-{index}" for index in range(12 - one_based_rank)
    )


def test_rank_1_metrics() -> None:
    metrics = evaluate_single_relevant_ranking((EXPECTED_ID, "offer-b"), EXPECTED_ID)
    assert metrics.recall_at_1 == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr_at_10 == 1.0
    assert metrics.ndcg_at_10 == 1.0
    assert metrics.relevant_rank == 1


def test_rank_2_metrics() -> None:
    metrics = evaluate_single_relevant_ranking(_ranked_with_expected_at(2), EXPECTED_ID)
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr_at_10 == 0.5
    assert metrics.ndcg_at_10 == pytest.approx(1.0 / math.log2(3))
    assert metrics.relevant_rank == 2


def test_rank_4_metrics() -> None:
    metrics = evaluate_single_relevant_ranking(RANKED_AT_4, EXPECTED_ID)
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr_at_10 == 0.25
    assert metrics.ndcg_at_10 == pytest.approx(1.0 / math.log2(5))
    assert metrics.relevant_rank == 4


def test_rank_10_metrics() -> None:
    metrics = evaluate_single_relevant_ranking(_ranked_with_expected_at(10), EXPECTED_ID)
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr_at_10 == 0.1
    assert metrics.ndcg_at_10 == pytest.approx(1.0 / math.log2(11))
    assert metrics.relevant_rank == 10


def test_rank_11_metrics() -> None:
    metrics = evaluate_single_relevant_ranking(_ranked_with_expected_at(11), EXPECTED_ID)
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_10 == 0.0
    assert metrics.mrr_at_10 == 0.0
    assert metrics.ndcg_at_10 == 0.0
    assert metrics.relevant_rank == 11


def test_absent_expected_metrics() -> None:
    metrics = evaluate_single_relevant_ranking(("offer-a", "offer-b"), EXPECTED_ID)
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_10 == 0.0
    assert metrics.mrr_at_10 == 0.0
    assert metrics.ndcg_at_10 == 0.0
    assert metrics.relevant_rank is None


def test_empty_ranked_list() -> None:
    metrics = evaluate_single_relevant_ranking((), EXPECTED_ID)
    assert metrics.recall_at_10 == 0.0
    assert metrics.relevant_rank is None


def test_duplicate_expected_uses_first_occurrence() -> None:
    ranked = (EXPECTED_ID, "offer-b", EXPECTED_ID, "offer-c")
    rank = locate_single_relevant_rank(ranked, EXPECTED_ID)
    assert rank.zero_based_rank == 0
    assert single_relevant_recall_at_k(rank, 1) == 1.0
    assert single_relevant_mrr_at_k(rank, 10) == 1.0
    assert single_relevant_ndcg_at_k(rank, 10) == 1.0


def test_single_relevant_metrics_match_arena_identity_transform() -> None:
    ranked_offer_ids = RANKED_AT_4
    metrics = evaluate_single_relevant_ranking(ranked_offer_ids, EXPECTED_ID)
    ranked_item_ids: list[int] = []
    irrelevant_counter = 1
    for offer_id in ranked_offer_ids:
        if offer_id == EXPECTED_ID:
            ranked_item_ids.append(0)
        else:
            ranked_item_ids.append(irrelevant_counter)
            irrelevant_counter += 1
    assert metrics.recall_at_1 == recall_at_k((0,), ranked_item_ids, 1)
    assert metrics.recall_at_5 == recall_at_k((0,), ranked_item_ids, 5)
    assert metrics.recall_at_10 == recall_at_k((0,), ranked_item_ids, 10)
    assert metrics.mrr_at_10 == mrr_at_k((0,), ranked_item_ids, 10)
    assert metrics.ndcg_at_10 == ndcg_at_k((0,), ranked_item_ids, 10)


def test_retrieval_metrics_module_has_no_provider_imports() -> None:
    import ast
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[5]
        / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/application/retrieval_metrics.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden_fragments = (
        "integrations",
        "composition",
        "stores",
        "postgresql",
        "qdrant",
    )
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(fragment in node.module for fragment in forbidden_fragments):
                violations.append(node.module)
    assert violations == []
