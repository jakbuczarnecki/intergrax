# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_intergrax_rag_answers_import_raises() -> None:
    with pytest.raises(ImportError, match="RetrievalService"):
        import intergrax.rag.answers  # noqa: F401
