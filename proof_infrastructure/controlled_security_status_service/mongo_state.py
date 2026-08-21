# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
import threading
from datetime import datetime

from proof_infrastructure.controlled_security_status_service.models import (
    SecurityStatusReadBehaviorV1,
    SecurityStatusResponseV1,
)
from proof_infrastructure.controlled_security_status_service.state import SecurityStatusStore


class MongoSecurityStatusStore:
    """MongoDB-backed security status authority for Dockerized proof vendors."""

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
        self._read_behavior = SecurityStatusReadBehaviorV1.NORMAL

    def put_security(self, status: SecurityStatusResponseV1) -> None:
        with self._lock:
            self._collection.replace_one(
                {"project_id": status.project_id},
                status.model_dump(mode="json"),
                upsert=True,
            )

    def get_security(self, project_id: str) -> SecurityStatusResponseV1 | None:
        with self._lock:
            document = self._collection.find_one({"project_id": project_id})
            if document is None:
                return None
            payload = {key: value for key, value in document.items() if key != "_id"}
            return SecurityStatusResponseV1.model_validate(payload)

    def read_security(self, project_id: str) -> SecurityStatusResponseV1 | None:
        with self._lock:
            self._read_request_count += 1
            document = self._collection.find_one({"project_id": project_id})
            if document is None:
                return None
            payload = {key: value for key, value in document.items() if key != "_id"}
            return SecurityStatusResponseV1.model_validate(payload)

    def read_request_count(self) -> int:
        with self._lock:
            return self._read_request_count

    def reset_read_request_count(self) -> None:
        with self._lock:
            self._read_request_count = 0

    def set_read_behavior(self, behavior: SecurityStatusReadBehaviorV1) -> None:
        with self._lock:
            self._read_behavior = behavior

    def read_behavior(self) -> SecurityStatusReadBehaviorV1:
        with self._lock:
            return self._read_behavior


def create_security_status_store_from_env() -> SecurityStatusStore | MongoSecurityStatusStore:
    uri = os.environ.get("SECURITY_STATUS_MONGODB_URI", "").strip()
    if not uri:
        return SecurityStatusStore()
    database = os.environ.get("SECURITY_STATUS_MONGODB_DATABASE", "governed_proof").strip()
    collection = os.environ.get(
        "SECURITY_STATUS_MONGODB_COLLECTION",
        "security_status_records",
    ).strip()
    return MongoSecurityStatusStore(
        uri=uri,
        database=database,
        collection=collection,
    )


def seed_security_status(
    store: SecurityStatusStore | MongoSecurityStatusStore,
    status: SecurityStatusResponseV1,
) -> SecurityStatusResponseV1:
    store.put_security(status)
    return status


def persisted_security_updated_at(
    store: SecurityStatusStore | MongoSecurityStatusStore,
    project_id: str,
) -> datetime | None:
    status = store.get_security(project_id)
    if status is None:
        return None
    return status.updated_at
