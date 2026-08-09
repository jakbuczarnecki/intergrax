# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.governance.embedding_version_policy import (
    clear_reindex_queue_hooks,
    register_reindex_queue_hook,
)
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = pytest.mark.gate


@pytest.fixture(autouse=True)
def _clear_hooks() -> None:
    clear_reindex_queue_hooks()
    yield
    clear_reindex_queue_hooks()


class _StubLoader:
    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {
                        "document_id": "doc-1",
                        "root_document_id": "doc-1",
                    },
                    "scope": {"tenant_id": "tenant.test"},
                    "content": "body",
                    "metadata": {"source": source},
                    "provenance": {"source_kind": "test", "source_id": source},
                }
            )
        ]


class _StubSplitter:
    def split_documents(self, docs, strategy_id=None):
        return list(docs)


class _StubEmbedding:
    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


class _StubVectorstore:
    def add_records(self, records, *, scope=None) -> list[str]:
        return ["id-0"]

    def list_source_record_ids(self, *, source_id: str, scope: object) -> tuple[str, ...]:
        return ()

    def count(self, *, scope: object) -> int:
        return 0

    def delete(self, ids, *, scope=None) -> None:
        del ids, scope


def test_ingest_pipeline_records_version_warnings(tmp_path: Path) -> None:
    source = tmp_path / "doc.txt"
    source.write_text("content", encoding="utf-8")
    queued: list[str] = []
    register_reindex_queue_hook(lambda req: queued.append(req.source_path))

    pipeline = IngestPipeline(
        loader=_StubLoader(),
        splitter=_StubSplitter(),
        embedding_manager=_StubEmbedding(),
        vectorstore=_StubVectorstore(),
        profile=RagProfile(embedding_model_version="v2"),
    )
    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={
                "tenant_id": "tenant.test",
                "embedding_model_version": "v1",
            },
        )
    )

    assert result.used is True
    assert any("incoming_metadata_version_mismatch" in warning for warning in result.version_warnings)
    assert result.reindex_recommended is True
    assert queued == [str(source)]
