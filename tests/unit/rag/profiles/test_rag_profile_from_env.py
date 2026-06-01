# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

import pytest

from intergrax.rag.profiles.rag_profile import rag_profile_from_env

pytestmark = pytest.mark.unit


def test_rag_profile_from_env_reads_prefetch_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_RAG_PREFETCH_TOP_K", "25")
    monkeypatch.setenv("INTERGRAX_RAG_FINAL_TOP_K", "6")
    monkeypatch.setenv("INTERGRAX_RAG_METRICS_ENABLED", "true")
    profile = rag_profile_from_env()
    assert profile.prefetch_top_k == 25
    assert profile.final_top_k == 6
    assert profile.extras.get("metrics_enabled") == "true"
