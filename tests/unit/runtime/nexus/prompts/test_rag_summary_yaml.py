# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.prompts.rag_prompt_builder import (
    DefaultRagPromptBuilder,
    RagPromptBundle,
)
from intergrax.runtime.nexus.context.context_builder import (
    RetrievedChunk,
    BuiltContext,
)
from intergrax.llm.messages import ChatMessage


pytestmark = pytest.mark.unit


def test_rag_prompt_builder_with_no_chunks_returns_empty_context() -> None:
    """
    If no retrieved chunks are present, builder should not inject any context messages.
    """

    # Arrange
    registry = YamlPromptRegistry.create_default(load=True)

    registry.load_all()

    builder = DefaultRagPromptBuilder(
        config=None,
        prompt_registry=registry,
    )

    built = BuiltContext(
        history_messages=[],
        retrieved_chunks=[],
        rag_used=True,
        rag_reason="test",
    )

    # Act
    bundle = builder.build_rag_prompt(built)

    # Assert
    assert isinstance(bundle, RagPromptBundle)
    assert bundle.context_messages == []


def test_rag_prompt_builder_injects_yaml_system_prompt_and_chunks() -> None:
    """
    Contract test for RAG prompt builder:

    - system instruction comes from YAML Prompt Registry
    - retrieved chunks are formatted and appended
    """

    # Arrange
    registry = YamlPromptRegistry.create_default(load=True)

    builder = DefaultRagPromptBuilder(
        config=None,
        prompt_registry=registry,
    )

    chunk = RetrievedChunk(
        id="chunk_1",
        text="This is a retrieved fragment.",
        metadata={"source_name": "test_doc"},
        score=.39,
    )

    built = BuiltContext(
        history_messages=[],
        retrieved_chunks=[chunk],
        rag_used=True,
        rag_reason="test",
    )

    # Act
    bundle = builder.build_rag_prompt(built)

    # Assert
    assert isinstance(bundle, RagPromptBundle)
    assert len(bundle.context_messages) == 1

    msg = bundle.context_messages[0]
    assert isinstance(msg, ChatMessage)
    assert msg.role == "user"

    # System part from YAML
    assert "retrieved" in msg.content.lower()

    # Chunk formatting
    assert "Source: test_doc" in msg.content
    assert "This is a retrieved fragment." in msg.content
