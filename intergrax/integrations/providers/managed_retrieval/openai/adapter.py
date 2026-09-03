# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenAI SDK adapter for hosted managed retrieval (vector stores + file_search)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

from intergrax.integrations.contracts.managed_retrieval import (
    ManagedRetrievalBackend,
    ManagedRetrievalProcessingTimeoutError,
    ManagedRetrievalQueryError,
    ManagedRetrievalQueryRequest,
    ManagedRetrievalResourceNotFoundError,
    ManagedRetrievalUploadError,
    ManagedRetrievalUploadResult,
)
from intergrax.integrations.providers.managed_retrieval.openai.config import (
    OpenAIManagedRetrievalConfig,
)
from intergrax.utils import attribute_access


class OpenAIManagedRetrievalAdapter(ManagedRetrievalBackend):
    """Maps neutral managed retrieval operations onto the OpenAI SDK."""

    def __init__(self, client: object, *, config: OpenAIManagedRetrievalConfig) -> None:
        self._client = client
        self._config = config

    def ensure_store_exists(self, store_id: str) -> None:
        try:
            self._client.vector_stores.retrieve(store_id)
        except Exception as exc:
            raise ManagedRetrievalResourceNotFoundError(
                f"managed store not found: {store_id}"
            ) from exc

    def list_attached_file_ids(self, store_id: str) -> Sequence[str]:
        files_page = self._client.vector_stores.files.list(
            vector_store_id=store_id,
            limit=100,
        )
        file_ids = [f.id for f in files_page.data]
        next_page = attribute_access.optional(files_page, "has_more", False)
        cursor = attribute_access.optional(files_page, "last_id", None)
        while next_page and cursor:
            page = self._client.vector_stores.files.list(
                vector_store_id=store_id,
                after=cursor,
                limit=100,
            )
            file_ids.extend([f.id for f in page.data])
            next_page = attribute_access.optional(page, "has_more", False)
            cursor = attribute_access.optional(page, "last_id", None)
        return file_ids

    def clear_store(self, store_id: str) -> int:
        file_ids = self.list_attached_file_ids(store_id)
        deleted = 0
        for fid in file_ids:
            try:
                self._client.vector_stores.files.delete(
                    vector_store_id=store_id,
                    file_id=fid,
                )
            except Exception:
                continue
            try:
                self._client.files.delete(file_id=fid)
                deleted += 1
            except Exception:
                continue
        return deleted

    def upload_folder(
        self,
        store_id: str,
        folder: str | Path,
        *,
        patterns: Sequence[str],
    ) -> ManagedRetrievalUploadResult:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {folder_path}")

        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(folder_path.glob(pattern))
        if not paths:
            return ManagedRetrievalUploadResult(uploaded_names=(), failed_names=())

        uploaded: list[str] = []
        failed: list[str] = []
        for path in paths:
            try:
                self._upload_single_file(store_id, path)
                uploaded.append(path.name)
            except Exception:
                failed.append(path.name)
        return ManagedRetrievalUploadResult(
            uploaded_names=tuple(uploaded),
            failed_names=tuple(failed),
        )

    def _upload_single_file(self, store_id: str, path: Path) -> None:
        with open(path, "rb") as handle:
            uploaded = self._client.files.create(file=handle, purpose="user_data")

        attempts = 0
        while attempts < self._config.max_poll_attempts:
            f_info = self._client.files.retrieve(uploaded.id)
            status = attribute_access.optional(f_info, "status", None)
            if status == "processed":
                break
            if status == "error":
                raise ManagedRetrievalUploadError(f"file processing failed for {path.name}")
            time.sleep(self._config.poll_interval_seconds)
            attempts += 1
        else:
            raise ManagedRetrievalProcessingTimeoutError(
                f"timed out waiting for file processing: {path.name}"
            )

        self._client.vector_stores.files.create(
            vector_store_id=store_id,
            file_id=uploaded.id,
        )

    def query(self, request: ManagedRetrievalQueryRequest) -> str:
        try:
            response = self._client.responses.create(
                model=request.model,
                instructions=request.instructions,
                input=request.question,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [request.store_id],
                        "max_num_results": request.max_results,
                        "ranking_options": {
                            "ranker": "auto",
                            "score_threshold": request.score_threshold,
                        },
                    }
                ],
            )
        except Exception as exc:
            raise ManagedRetrievalQueryError("hosted retrieval query failed") from exc
        return str(attribute_access.optional(response, "output_text", "") or "")


def create_openai_client(config: OpenAIManagedRetrievalConfig) -> object:
    from openai import OpenAI

    return OpenAI(api_key=config.api_key)


def create_openai_managed_retrieval_adapter(
    config: OpenAIManagedRetrievalConfig,
) -> OpenAIManagedRetrievalAdapter:
    return OpenAIManagedRetrievalAdapter(create_openai_client(config), config=config)
