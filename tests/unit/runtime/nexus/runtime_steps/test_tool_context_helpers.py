# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.runtime.nexus.runtime_steps.tool_context_helpers import format_rag_context

pytestmark = pytest.mark.gate


@dataclass
class _DocChunk:
    text: str
    metadata: dict[str, str]


def test_format_rag_context_reads_dataclass_chunk() -> None:
    out = format_rag_context(
        [_DocChunk(text="Hello harness.", metadata={"source": "doc.md"})],
        max_chars=500,
    )
    assert "Hello harness" in out
    assert "doc.md" in out


def test_format_rag_context_reads_dict_chunk() -> None:
    out = format_rag_context(
        [{"content": "Dict body", "metadata": {"page": "2"}}],
        max_chars=500,
    )
    assert "Dict body" in out
    assert "p=2" in out
