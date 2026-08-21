# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
import threading

from proof_infrastructure.controlled_governance_approval_service.models import (
    GovernanceApprovalResponseV1,
)
from proof_infrastructure.controlled_governance_approval_service.state import (
    GovernanceApprovalStore,
)


class MongoGovernanceApprovalStore:
    """MongoDB-backed governance approval authority for Dockerized proof vendors."""

    def __init__(
        self,
        *,
        uri: str,
        database: str,
        collection: str,
    ) -> None:
        from pymongo import MongoClient

        self._client = MongoClient(uri)
        self._collection = self._client[database][collection]
        self._lock = threading.Lock()
        self._read_request_count = 0

    def put_governance(self, status: GovernanceApprovalResponseV1) -> None:
        with self._lock:
            self._collection.replace_one(
                {"subject_id": status.subject_id},
                status.model_dump(mode="json"),
                upsert=True,
            )

    def get_governance(self, subject_id: str) -> GovernanceApprovalResponseV1 | None:
        with self._lock:
            document = self._collection.find_one({"subject_id": subject_id})
            if document is None:
                return None
            payload = {key: value for key, value in document.items() if key != "_id"}
            return GovernanceApprovalResponseV1.model_validate(payload)

    def read_governance(self, subject_id: str) -> GovernanceApprovalResponseV1 | None:
        with self._lock:
            self._read_request_count += 1
            document = self._collection.find_one({"subject_id": subject_id})
            if document is None:
                return None
            payload = {key: value for key, value in document.items() if key != "_id"}
            return GovernanceApprovalResponseV1.model_validate(payload)

    def read_request_count(self) -> int:
        with self._lock:
            return self._read_request_count

    def reset_read_request_count(self) -> None:
        with self._lock:
            self._read_request_count = 0


def create_governance_approval_store_from_env() -> (
    GovernanceApprovalStore | MongoGovernanceApprovalStore
):
    uri = os.environ.get("GOVERNANCE_APPROVAL_MONGODB_URI", "").strip()
    if not uri:
        return GovernanceApprovalStore()
    database = os.environ.get(
        "GOVERNANCE_APPROVAL_MONGODB_DATABASE",
        "governed_proof",
    ).strip()
    collection = os.environ.get(
        "GOVERNANCE_APPROVAL_MONGODB_COLLECTION",
        "governance_approval_records",
    ).strip()
    return MongoGovernanceApprovalStore(
        uri=uri,
        database=database,
        collection=collection,
    )
