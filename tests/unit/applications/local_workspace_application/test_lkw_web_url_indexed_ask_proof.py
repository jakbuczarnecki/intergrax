# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/run-lkw-web-url-indexed-ask-proof.py"
)


def _load_module() -> ModuleType:
    module_name = "run_lkw_web_url_indexed_ask_proof"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_assert_web_url_acceptance_normalizes_display_url() -> None:
    module = _load_module()
    source_id, operation_id = module.assert_web_url_acceptance(
        {
            "safe_display_url": "https://example.com",
            "source_id": "src-1",
            "operation_id": "op-1",
        }
    )
    assert source_id == "src-1"
    assert operation_id == "op-1"


def test_assert_indexed_hybrid_ask_run_requires_web_url_citation_marker() -> None:
    module = _load_module()
    with pytest.raises(module.ProofFailure, match="citation_excerpt_missing_marker"):
        module.assert_indexed_hybrid_ask_run(
            {
                "run_schema_version": 2,
                "query_mode": "indexed_only",
                "indexed_retrieval_status": "completed",
                "live_execution_status": "skipped",
                "status": "completed",
                "answer": "Example Domain",
                "citations": [
                    {
                        "source_id": "src-1",
                        "evidence_id": "idx:evidence-1",
                        "evidence_type": "indexed",
                        "file_name": "https://example.com",
                        "excerpt": "Wrong marker",
                    }
                ],
                "persisted_evidence": [{"evidence_type": "indexed", "evidence_id": "idx:evidence-1"}],
            },
            source_id="src-1",
        )
