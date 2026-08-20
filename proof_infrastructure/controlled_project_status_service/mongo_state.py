# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
import threading
from datetime import datetime

from proof_infrastructure.controlled_project_status_service.models import (
    ProjectStatusReadBehaviorV1,
    ProjectStatusResponseV1,
    ProjectStatusControlUpdateV1,
)
from proof_infrastructure.controlled_project_status_service.state import ProjectStatusStore


class MongoProjectStatusStore:
    """MongoDB-backed project status authority for Dockerized proof vendors."""

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
        self._read_behavior = ProjectStatusReadBehaviorV1.NORMAL

    def put_status(self, status: ProjectStatusResponseV1) -> None:
        with self._lock:
            self._collection.replace_one(
                {"project_id": status.project_id},
                status.model_dump(mode="json"),
                upsert=True,
            )

    def get_status(self, project_id: str) -> ProjectStatusResponseV1 | None:
        with self._lock:
            document = self._collection.find_one({"project_id": project_id})
            if document is None:
                return None
            payload = {key: value for key, value in document.items() if key != "_id"}
            return ProjectStatusResponseV1.model_validate(payload)

    def read_status(self, project_id: str) -> ProjectStatusResponseV1 | None:
        with self._lock:
            self._read_request_count += 1
            document = self._collection.find_one({"project_id": project_id})
            if document is None:
                return None
            payload = {key: value for key, value in document.items() if key != "_id"}
            return ProjectStatusResponseV1.model_validate(payload)

    def update_status(
        self,
        project_id: str,
        update: ProjectStatusControlUpdateV1,
    ) -> ProjectStatusResponseV1:
        with self._lock:
            current = self.get_status(project_id)
            if current is None:
                raise KeyError("project_not_found")
            updated = current.model_copy(
                update={
                    "readiness_score": (
                        update.readiness_score
                        if update.readiness_score is not None
                        else current.readiness_score
                    ),
                    "blockers": (
                        update.blockers
                        if update.blockers is not None
                        else current.blockers
                    ),
                    "status": (
                        update.status if update.status is not None else current.status
                    ),
                    "updated_at": datetime.now().astimezone(),
                }
            )
            self.put_status(updated)
            return updated

    def read_request_count(self) -> int:
        with self._lock:
            return self._read_request_count

    def reset_read_request_count(self) -> None:
        with self._lock:
            self._read_request_count = 0

    def set_read_behavior(self, behavior: ProjectStatusReadBehaviorV1) -> None:
        with self._lock:
            self._read_behavior = behavior

    def read_behavior(self) -> ProjectStatusReadBehaviorV1:
        with self._lock:
            return self._read_behavior


def create_project_status_store_from_env() -> ProjectStatusStore | MongoProjectStatusStore:
    uri = os.environ.get("PROJECT_STATUS_MONGODB_URI", "").strip()
    if not uri:
        return ProjectStatusStore()
    database = os.environ.get("PROJECT_STATUS_MONGODB_DATABASE", "governed_proof").strip()
    collection = os.environ.get(
        "PROJECT_STATUS_MONGODB_COLLECTION",
        "project_status_records",
    ).strip()
    return MongoProjectStatusStore(
        uri=uri,
        database=database,
        collection=collection,
    )
