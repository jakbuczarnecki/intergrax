"""In-memory qualification backend for the reference provider."""

from __future__ import annotations

from dataclasses import dataclass

from acme_reference_vk_plugin.constants import (
    ACME_DEFAULT_COLLECTION_ID,
    ACME_REFERENCE_MARKER,
)


@dataclass(frozen=True, slots=True)
class AcmeReferenceDocument:
    remote_id: str
    title: str
    body: str
    revision: str


@dataclass(frozen=True, slots=True)
class AcmeReferenceCollection:
    collection_id: str
    safe_display_label: str
    documents: tuple[AcmeReferenceDocument, ...]


class AcmeReferenceBackend:
    """Deterministic, bounded, in-memory remote knowledge store."""

    def __init__(self) -> None:
        self._collections: dict[str, AcmeReferenceCollection] = {
            ACME_DEFAULT_COLLECTION_ID: AcmeReferenceCollection(
                collection_id=ACME_DEFAULT_COLLECTION_ID,
                safe_display_label="Acme Reference Collection",
                documents=(
                    AcmeReferenceDocument(
                        remote_id="doc-qual-001",
                        title="Reference qualification article",
                        body=(
                            "Synthetic knowledge article for VK-EXT-3.\n\n"
                            f"{ACME_REFERENCE_MARKER}\n"
                        ),
                        revision="acme-rev-qual-001",
                    ),
                ),
            )
        }

    def list_collections(self) -> tuple[AcmeReferenceCollection, ...]:
        return tuple(
            sorted(self._collections.values(), key=lambda item: item.collection_id)
        )

    def get_collection(self, collection_id: str) -> AcmeReferenceCollection | None:
        return self._collections.get(collection_id.strip())

    def list_documents(
        self,
        *,
        collection_id: str,
    ) -> tuple[AcmeReferenceDocument, ...]:
        collection = self.get_collection(collection_id)
        if collection is None:
            return ()
        return collection.documents

    def get_document(
        self,
        *,
        collection_id: str,
        remote_id: str,
    ) -> AcmeReferenceDocument | None:
        for document in self.list_documents(collection_id=collection_id):
            if document.remote_id == remote_id:
                return document
        return None
