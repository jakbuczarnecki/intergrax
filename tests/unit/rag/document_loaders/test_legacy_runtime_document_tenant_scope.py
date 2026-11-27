# © Artur Czarnecki. All rights reserved.

"""EDER-F-0236 (EQOA-A8-020): legacy RAG hit tenant binding and compression receipt identity.

Covers:
1. ``from_legacy_rag_hit(hit, bound_tenant_id=...)`` in
   ``intergrax/rag/document_loaders/compat/legacy_runtime_document.py`` rejecting hostile,
   missing, or blank tenant bindings and preserving matching bound tenants.
2. Current-HEAD ``CompressionReceipt`` / ``_derive_receipt_id`` tenant-attribution and
   receipt-identity contract in ``intergrax/runtime/token_optimization/receipts.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from intergrax.compat.langchain import documents as langchain_documents_module
from intergrax.compat.langchain.documents import (
    LangChainCompatibilityUnavailableError,
    LangChainDocumentBridgeError,
)
from intergrax.rag.document_loaders.compat.legacy_runtime_document import from_legacy_rag_hit
from intergrax.runtime.token_optimization.contracts import (
    StrategySafetyClass,
    TokenOptimizationAttribution,
    TokenOptimizationDecision,
    TokenOptimizationMechanism,
    TokenOptimizationRequest,
    TokenOptimizationResult,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
)
from intergrax.runtime.token_optimization.receipts import (
    CompressionReceiptValidationStatus,
    _derive_receipt_id,
    build_compression_receipt,
    hash_content,
    make_compression_receipt_ref,
    validate_receipt_integrity,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _LegacyHit:
    id: str | None
    content: str
    metadata: dict[str, Any]


@dataclass
class _LangChainDocumentStub:
    id: str | None
    page_content: str
    metadata: dict[str, Any]


@pytest.fixture(autouse=True)
def _ensure_langchain_document_carrier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use real langchain_core.documents.Document when installed, else a contract stub (no skips)."""
    try:
        langchain_documents_module._get_langchain_document_class()
    except LangChainCompatibilityUnavailableError:
        monkeypatch.setattr(
            langchain_documents_module,
            "_get_langchain_document_class",
            lambda: _LangChainDocumentStub,
        )


def _legacy_hit(*, tenant_id: str | None = "tenant-a", doc_id: str = "file:doc-1") -> _LegacyHit:
    metadata: dict[str, Any] = {
        "source_kind": "file",
        "source_id": "doc-1",
    }
    if tenant_id is not None:
        metadata["tenant_id"] = tenant_id
    return _LegacyHit(id=doc_id, content="legacy hit payload", metadata=metadata)


def test_from_legacy_rag_hit_rejects_mismatched_bound_tenant_id() -> None:
    """Hostile metadata tenant_id must be rejected when bound_tenant_id differs."""
    hit = _legacy_hit(tenant_id="tenant-b")
    with pytest.raises(LangChainDocumentBridgeError, match="does not match bound_tenant_id"):
        from_legacy_rag_hit(hit, bound_tenant_id="tenant-a")


def test_from_legacy_rag_hit_rejects_blank_bound_tenant_id() -> None:
    """Whitespace-only bound_tenant_id fails closed."""
    hit = _legacy_hit(tenant_id="tenant-a")
    with pytest.raises(LangChainDocumentBridgeError, match="bound_tenant_id must be non-empty"):
        from_legacy_rag_hit(hit, bound_tenant_id="   ")


def test_from_legacy_rag_hit_rejects_missing_metadata_tenant_id() -> None:
    """Legacy hit lacking metadata['tenant_id'] fails closed even when bound_tenant_id is passed."""
    hit = _legacy_hit(tenant_id=None)
    with pytest.raises(LangChainDocumentBridgeError, match="tenant_id"):
        from_legacy_rag_hit(hit, bound_tenant_id="tenant-a")


def test_from_legacy_rag_hit_binds_matching_tenant_id() -> None:
    """Matching metadata tenant_id and bound_tenant_id reconstructs a scoped KnowledgeDocument."""
    hit = _legacy_hit(tenant_id="tenant-a", doc_id="file:doc-42")
    native = from_legacy_rag_hit(hit, bound_tenant_id="tenant-a")

    assert native.scope.tenant_id == "tenant-a"
    assert native.identity.document_id == "file:doc-42"
    assert native.content == "legacy hit payload"


def _token_opt_strategy() -> TokenOptimizationStrategyRef:
    return TokenOptimizationStrategyRef(
        strategy_id="tool_catalog.minimize",
        mechanism=TokenOptimizationMechanism.TOOL_CATALOG_COMPACTION,
        kind=TokenOptimizationStrategyKind.SCHEMA_MINIMIZATION,
        safety_class=StrategySafetyClass.LOSSLESS,
        plugin_id="builtin.tool_catalog",
    )


def _token_opt_request(*, tenant_id: str, run_id: str, original: str) -> TokenOptimizationRequest:
    return TokenOptimizationRequest(
        content=original,
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        attribution=TokenOptimizationAttribution(
            run_id=run_id,
            step_id="step-1",
            tenant_id=tenant_id,
            plugin_id="builtin.tool_catalog",
        ),
        strategy=_token_opt_strategy(),
        metadata={"tenant_scope": tenant_id},
    )


def test_compression_receipt_identity_scopes_tenant_attribution() -> None:
    """Lock current-HEAD CompressionReceipt tenant attribution and _derive_receipt_id contract.

    Verifies:
    - ``build_compression_receipt`` preserves distinct ``attribution.tenant_id`` and metadata per
      tenant and passes deterministic integrity validation without content leakage.
    - Current-HEAD ``_derive_receipt_id`` is content-addressed over
      ``source_type|original_hash|optimized_hash|strategy_key`` (does not yet hash
      ``attribution.tenant_id`` — tracked under EDER-F-0236 / EQOA-A8-020).
    - Supplying an explicit tenant-scoped ``receipt_id`` produces distinct ``CompressionReceiptRef``
      identities per tenant while preserving content hashes and tenant metadata.
    """
    original = '{"tools": [{"name": "lookup", "description": "Find records"}]}'
    optimized = '{"tools":[{"name":"lookup","description":"Find records"}]}'
    strategy = _token_opt_strategy()
    result = TokenOptimizationResult(
        content=optimized,
        decision=TokenOptimizationDecision.APPLY,
        strategy=strategy,
    )

    req_a = _token_opt_request(tenant_id="tenant-a", run_id="run-a", original=original)
    req_b = _token_opt_request(tenant_id="tenant-b", run_id="run-b", original=original)

    receipt_a = build_compression_receipt(
        original_content=original,
        optimized_content=optimized,
        request=req_a,
        result=result,
        created_at="2026-11-27T00:00:00+00:00",
    )
    receipt_b = build_compression_receipt(
        original_content=original,
        optimized_content=optimized,
        request=req_b,
        result=result,
        created_at="2026-11-27T00:00:00+00:00",
    )

    assert receipt_a.attribution is not None
    assert receipt_b.attribution is not None
    assert receipt_a.attribution.tenant_id == "tenant-a"
    assert receipt_b.attribution.tenant_id == "tenant-b"
    assert receipt_a.run_id == "run-a"
    assert receipt_b.run_id == "run-b"

    assert (
        validate_receipt_integrity(
            receipt_a,
            original_content=original,
            optimized_content=optimized,
        ).status
        is CompressionReceiptValidationStatus.PASSED
    )
    assert (
        validate_receipt_integrity(
            receipt_b,
            original_content=original,
            optimized_content=optimized,
        ).status
        is CompressionReceiptValidationStatus.PASSED
    )

    expected_derived = _derive_receipt_id(
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        original_hash=hash_content(original),
        optimized_hash=hash_content(optimized),
        strategy=strategy,
        decision=TokenOptimizationDecision.APPLY,
    )
    assert receipt_a.receipt_id == expected_derived
    assert receipt_b.receipt_id == expected_derived

    scoped_a = build_compression_receipt(
        original_content=original,
        optimized_content=optimized,
        request=req_a,
        result=result,
        receipt_id=f"tenant-a:{expected_derived}",
        created_at="2026-11-27T00:00:00+00:00",
    )
    scoped_b = build_compression_receipt(
        original_content=original,
        optimized_content=optimized,
        request=req_b,
        result=result,
        receipt_id=f"tenant-b:{expected_derived}",
        created_at="2026-11-27T00:00:00+00:00",
    )
    ref_a = make_compression_receipt_ref(scoped_a)
    ref_b = make_compression_receipt_ref(scoped_b)

    assert ref_a.receipt_id != ref_b.receipt_id
    assert ref_a.metadata["tenant_scope"] == "tenant-a"
    assert ref_b.metadata["tenant_scope"] == "tenant-b"
    assert ref_a.original_hash == ref_b.original_hash == hash_content(original)
