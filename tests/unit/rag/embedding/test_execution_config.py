# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Unit tests for provider-neutral embedding execution config."""

from __future__ import annotations

import pytest

from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig

pytestmark = pytest.mark.unit


def test_execution_config_carries_typed_values_only() -> None:
    config = EmbeddingProviderExecutionConfig(device="cuda", batch_size=64)

    assert config.device == "cuda"
    assert config.batch_size == 64
    assert not hasattr(config, "hf_init_kwargs")
