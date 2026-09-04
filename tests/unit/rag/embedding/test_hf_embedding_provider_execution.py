# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Unit tests for HF embedding provider execution diagnostics."""

from __future__ import annotations

import pytest

from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider

pytestmark = pytest.mark.unit


def test_hf_execution_snapshot_reports_configured_state() -> None:
    provider = HFEmbeddingProvider(
        model_name="test-model",
        device="cuda",
        batch_size=64,
    )

    snapshot = provider.execution_snapshot()

    assert snapshot.configured_device == "cuda"
    assert snapshot.resolved_device == "cuda"
    assert snapshot.provider_batch_size == 64
    assert snapshot.max_length is None
    assert snapshot.evidence_source == "HFEmbeddingProvider.execution_snapshot"
