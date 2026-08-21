# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
import threading

from proof_infrastructure.controlled_change_approval_service.models import (
    ChangeApprovalResponseV1,
)
from proof_infrastructure.controlled_change_approval_service.state import ChangeApprovalStore


class MongoChangeApprovalStore:
    """MongoDB-backed change approval authority for Dockerized proof vendors."""

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

    def put_change(self, status: ChangeApprovalResponseV1) -> None:
        with self._lock:
            self._collection.replace_one(
                {"change_id": status.change_id},
                status.model_dump(mode="json"),
                upsert=True,
            )

    def get_change(self, change_id: str) -> ChangeApprovalResponseV1 | None:
        with self._lock:
            document = self._collection.find_one({"change_id": change_id})
            if document is None:
                return None
            payload = {key: value for key, value in document.items() if key != "_id"}
            return ChangeApprovalResponseV1.model_validate(payload)

    def read_change(self, change_id: str) -> ChangeApprovalResponseV1 | None:
        with self._lock:
            self._read_request_count += 1
            document = self._collection.find_one({"change_id": change_id})
            if document is None:
                return None
            payload = {key: value for key, value in document.items() if key != "_id"}
            return ChangeApprovalResponseV1.model_validate(payload)

    def read_request_count(self) -> int:
        with self._lock:
            return self._read_request_count

    def reset_read_request_count(self) -> None:
        with self._lock:
            self._read_request_count = 0


def create_change_approval_store_from_env() -> ChangeApprovalStore | MongoChangeApprovalStore:
    uri = os.environ.get("CHANGE_APPROVAL_MONGODB_URI", "").strip()
    if not uri:
        return ChangeApprovalStore()
    database = os.environ.get("CHANGE_APPROVAL_MONGODB_DATABASE", "governed_proof").strip()
    collection = os.environ.get(
        "CHANGE_APPROVAL_MONGODB_COLLECTION",
        "change_approval_records",
    ).strip()
    return MongoChangeApprovalStore(
        uri=uri,
        database=database,
        collection=collection,
    )
