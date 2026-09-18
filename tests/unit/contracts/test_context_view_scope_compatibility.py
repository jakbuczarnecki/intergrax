# © Artur Czarnecki. All rights reserved.

"""MP-5G-C1 — typed ContextView scope compatibility policy."""

from __future__ import annotations

import pytest

from intergrax.contracts.context_view import ContextViewCategory, ContextViewOperationScope, ContextViewScope
from intergrax.contracts.context_view_scope_compatibility import (
    DefaultContextViewScopeCompatibilityPolicy,
    memory_source_candidate_scope_compatible,
    source_candidate_scope_compatible,
)

pytestmark = pytest.mark.unit


def _scope(**overrides: object) -> ContextViewScope:
    payload = {"tenant_id": "tenant-a", "workspace_id": "ws-1"}
    payload.update(overrides)
    return ContextViewScope(**payload)


def test_memory_omits_unproven_work_item_and_operation_dimensions() -> None:
    request = _scope(
        work_item_id="wi-1",
        operation_scope=ContextViewOperationScope(operation_id="op", resource_scope="ctx"),
    )
    candidate = _scope()
    assert memory_source_candidate_scope_compatible(request, candidate)


def test_memory_false_work_item_claim_rejected() -> None:
    request = _scope(work_item_id="wi-1")
    candidate = _scope(work_item_id="wi-2")
    assert not memory_source_candidate_scope_compatible(request, candidate)


def test_default_policy_memory_category() -> None:
    policy = DefaultContextViewScopeCompatibilityPolicy()
    request = _scope(work_item_id="wi-1")
    candidate = _scope()
    assert policy.candidate_scope_compatible(
        category=ContextViewCategory.MEMORY,
        request_scope=request,
        candidate_scope=candidate,
    )


def test_knowledge_omits_unproven_work_item_when_document_operation_matches() -> None:
    op = ContextViewOperationScope(operation_id="op", resource_scope="doc-1")
    request = _scope(work_item_id="wi-1", operation_scope=op)
    candidate = _scope(operation_scope=op)
    assert source_candidate_scope_compatible(
        category=ContextViewCategory.KNOWLEDGE,
        request_scope=request,
        candidate_scope=candidate,
    )
