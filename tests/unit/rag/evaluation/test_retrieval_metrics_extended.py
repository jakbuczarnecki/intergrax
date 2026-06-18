# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b", "c"], {"b", "d"}, k=2) == 0.5


def test_ndcg_at_k_perfect_ranking() -> None:
    relevant = {"a", "b"}
    assert ndcg_at_k(["a", "b", "c"], relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_zero_when_no_relevant() -> None:
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


def test_recall_precision_consistency() -> None:
    retrieved = ["x", "y", "z"]
    relevant = {"y", "z"}
    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
