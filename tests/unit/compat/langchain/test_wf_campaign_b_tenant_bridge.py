# © Artur Czarnecki. All rights reserved.

"""PT-WF / Campaign B E3 — LangChain bridge tenant binding (A8-020)."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")
from intergrax.compat.langchain.documents import (
    LangChainDocumentBridgeError,
    from_langchain_document,
)
from langchain_core.documents import Document


def _metadata(*, tenant_id: str) -> dict[str, str]:
    return {
        "tenant_id": tenant_id,
        "source_kind": "file",
        "source_id": "abc123",
    }


@pytest.mark.unit
def test_nt_wf_neg_e3_01_foreign_metadata_rejected_with_bound_tenant() -> None:
    """NT-WF-NEG-E3-01 / Campaign B NT-11: hostile metadata tenant must not bind."""
    document = Document(
        id="file:abc123",
        page_content="payload",
        metadata=_metadata(tenant_id="tenant-b"),
    )
    with pytest.raises(LangChainDocumentBridgeError, match="does not match bound_tenant_id"):
        from_langchain_document(document, bound_tenant_id="tenant-a")


@pytest.mark.unit
def test_nt_wf_neg_e3_01_matching_bound_tenant_passes() -> None:
    document = Document(
        id="file:abc123",
        page_content="payload",
        metadata=_metadata(tenant_id="tenant-a"),
    )
    native = from_langchain_document(document, bound_tenant_id="tenant-a")
    assert native.scope.tenant_id == "tenant-a"
