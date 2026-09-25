# © Artur Czarnecki. All rights reserved.

"""EBH-2G-R1 — rag-guard PR path triggers cover canonical RAG integration backends."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RAG_GUARD_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "rag-guard.yml"

# Integration provider trees consumed on production RAG ingest/retrieval/bootstrap paths.
_REQUIRED_INTEGRATION_PROVIDER_PATHS: tuple[str, ...] = (
    "intergrax/integrations/providers/vector_store/**",
    "intergrax/integrations/providers/embedding_provider/**",
    "intergrax/integrations/providers/document_parser/**",
    "intergrax/integrations/providers/rerank_provider/**",
    "intergrax/integrations/providers/graph_store/**",
)


def test_rag_guard_workflow_lists_required_integration_provider_paths() -> None:
    workflow_text = _RAG_GUARD_WORKFLOW.read_text(encoding="utf-8")
    for required in _REQUIRED_INTEGRATION_PROVIDER_PATHS:
        assert required in workflow_text, (
            f"rag-guard.yml must trigger on {required!r} (canonical RAG integration backend)"
        )
