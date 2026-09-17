# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15-R2: bounded guards against identity repair in Memory authority paths."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.gate

_REPO = Path(__file__).resolve().parents[3]
_MANAGER = _REPO / "intergrax" / "memory" / "user_profile_manager.py"
_LTM = _REPO / "intergrax" / "tools" / "providers" / "ltm" / "service.py"
_MEMORY_CTX = (
    _REPO / "intergrax" / "runtime" / "nexus" / "context" / "memory_context_invocation.py"
)


def test_user_profile_manager_forbids_identity_user_id_model_copy_repair() -> None:
    source = _MANAGER.read_text(encoding="utf-8")
    assert 'model_copy(update={"user_id"' not in source


def test_ltm_service_forbids_synthetic_request_identity_fallback() -> None:
    source = _LTM.read_text(encoding="utf-8")
    assert "RequestIdentity(" not in source


def test_ltm_service_forbids_manager_semantic_bypass() -> None:
    source = _LTM.read_text(encoding="utf-8")
    assert "add_memory_entry" not in source
    assert "search_longterm_memory" not in source
    assert "_keyword_hits" not in source


def test_runtime_ltm_recall_forbids_session_manager_semantic_bypass() -> None:
    source = _MEMORY_CTX.read_text(encoding="utf-8")
    assert "search_user_longterm_memory" not in source


def test_user_profile_manager_forbids_ltm_vector_projection_materialization() -> None:
    source = _MANAGER.read_text(encoding="utf-8")
    assert "UserProfileLtmVectorProjection" not in source


_SESSION_MANAGER = _REPO / "intergrax" / "runtime" / "nexus" / "session" / "session_manager.py"


def test_session_manager_forbids_user_profile_manager_ltm_search_bypass() -> None:
    source = _SESSION_MANAGER.read_text(encoding="utf-8")
    assert "search_longterm_memory" not in source
