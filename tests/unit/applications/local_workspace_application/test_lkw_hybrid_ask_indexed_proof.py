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
    / "applications/local_workspace_application/scripts/run-lkw-hybrid-ask-indexed-proof.py"
)


def _load_module() -> ModuleType:
    module_name = "run_lkw_hybrid_ask_indexed_proof"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_assert_indexed_hybrid_ask_run_accepts_valid_payload() -> None:
    module = _load_module()
    marker = "HYBRID-IDX-deadbeef"
    module.assert_indexed_hybrid_ask_run(
        {
            "run_schema_version": 2,
            "query_mode": "indexed_only",
            "indexed_retrieval_status": "completed",
            "live_execution_status": "skipped",
            "status": "completed",
            "answer": f"The escalation response window is 21 calendar days for {marker}.",
            "citations": [
                {
                    "evidence_id": "idx:evidence-1",
                    "evidence_type": "indexed",
                    "excerpt": f"Escalation response window: 21 calendar days. {marker}.",
                }
            ],
            "persisted_evidence": [{"evidence_type": "indexed", "evidence_id": "idx:evidence-1"}],
        },
        marker=marker,
        expected_answer_fragment="21 calendar days",
    )


def test_assert_indexed_hybrid_ask_run_accepts_retrieval_only_boundary() -> None:
    module = _load_module()
    module.assert_indexed_hybrid_ask_run(
        {
            "run_schema_version": 2,
            "query_mode": "indexed_only",
            "indexed_retrieval_status": "completed",
            "live_execution_status": "skipped",
            "status": "insufficient_evidence",
            "answer": None,
            "citations": [],
            "persisted_evidence": [],
        },
        marker="HYBRID-IDX-deadbeef",
        expected_answer_fragment="21 calendar days",
    )


def test_assert_indexed_hybrid_ask_run_rejects_live_branch() -> None:
    module = _load_module()
    with pytest.raises(module.ProofFailure, match="live_execution_not_skipped"):
        module.assert_indexed_hybrid_ask_run(
            {
                "run_schema_version": 2,
                "query_mode": "indexed_only",
                "indexed_retrieval_status": "completed",
                "live_execution_status": "completed",
                "status": "completed",
                "answer": "21 calendar days",
                "citations": [
                    {
                        "evidence_id": "idx:evidence-1",
                        "evidence_type": "indexed",
                        "excerpt": "21 calendar days",
                    }
                ],
                "persisted_evidence": [{"evidence_type": "indexed"}],
            },
            marker="HYBRID-IDX-deadbeef",
            expected_answer_fragment="21 calendar days",
        )
