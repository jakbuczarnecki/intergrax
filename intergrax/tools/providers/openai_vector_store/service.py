# © Artur Czarnecki. All rights reserved.

"""OpenAI managed vector store operations for catalog tools (not Tier-0 ``rag.*``)."""

from __future__ import annotations
from intergrax.utils import attribute_access

import os
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

from intergrax.tools.providers.openai_vector_store.config import (
    OpenAIVectorStoreToolConfig,
    openai_vector_store_config_from_env,
)
from intergrax.tools.providers.openai_vector_store.contracts import (
    OpenAiFileSearchQueryInput,
    OpenAiFileSearchQueryOutput,
    OpenAiVectorStoreClearInput,
    OpenAiVectorStoreClearOutput,
    OpenAiVectorStoreUploadInput,
    OpenAiVectorStoreUploadOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

OPENAI_FILE_SEARCH_QUERY_TOOL_ID = "openai.file_search.query"
OPENAI_VECTOR_STORE_UPLOAD_TOOL_ID = "openai.vector_store.upload"
OPENAI_VECTOR_STORE_CLEAR_TOOL_ID = "openai.vector_store.clear"

DEFAULT_FILE_SEARCH_INSTRUCTIONS = """
ROLE (STRICT RAG)

You are a Knowledge Retrieval Assistant. Your ONLY allowed source of truth is the content retrieved from documents via the file_search tool (vector store).
You MUST NOT use general knowledge, outside facts, assumptions, or world knowledge.

PURPOSE

Answer the user's questions using ONLY the retrieved document fragments.
Provide accurate, thorough, source-backed answers.

WORKFLOW (MANDATORY, STEP-BY-STEP)

1. Understand the question.
- If multi-part: split into sub-questions and address each one.

2. Retrieve context.
- Use file_search.
- Perform multiple differently-phrased queries if needed.
- Ensure you have enough coverage.

3. Verify consistency.
- Compare fragments.
- If contradictions appear: explicitly describe them and list possible interpretations (each with source reference).

4. Answer.
- Write concise conclusions.
- Then provide expanded explanation (definitions, context, consequences).
- ALL content must come from cited fragments.

5. Cite sources.
- After each important claim add a parenthetical reference:
    (Source: file_name, p. X) or (Source: file_name, section Y).
- For long answers: add a final "Sources" section.
- Use direct quotes only when truly necessary and keep them short.

UNCERTAINTY RULES

If the documents do NOT contain enough information:
- Say explicitly: "Based on the available documents, I cannot fully answer X."
- Specify what is missing (section name, document type, etc.).
- Suggest concrete search phrases or additional documents.

You MUST NOT:
- invent information
- speculate
- rely on prior knowledge
- fill gaps with assumptions

If you infer something from the provided fragments, label it clearly as:
"Conclusion based on sources."

RESPONSE STYLE

1. Start with a short, 2-4 sentence summary.
2. Then provide detailed explanation:
- step-by-step reasoning
- bullet lists
- small headings
3. Use precise terminology, no generalities or abstract phrasing.
4. For procedures or algorithms: produce a checklist or pseudo-procedure.
5. For numeric values: provide exact numbers and cite sources.

OUTPUT FORMAT

Summary
Detailed explanation (with inline citations)
Sources (file name + page/section)

PROHIBITED ACTIONS (ABSOLUTE)

- Do not use ANY information outside the retrieved documents.
- Do not rely on common knowledge, intuition, or the internet.
- Do not hide uncertainty.
- Do not strengthen or reinterpret claims beyond what is written.
""".strip()


class OpenAIVectorStoreOps:
    """Low-level OpenAI vector store + file_search operations."""

    def __init__(
        self,
        client: Any,
        vector_store_id: str,
        *,
        config: Optional[OpenAIVectorStoreToolConfig] = None,
    ) -> None:
        self._client = client
        self._vector_store_id = vector_store_id
        self._config = config or openai_vector_store_config_from_env()

    @property
    def vector_store_id(self) -> str:
        return self._vector_store_id

    def ensure_vector_store_exists(self) -> Any:
        return self._client.vector_stores.retrieve(self._vector_store_id)

    def list_all_file_ids(self) -> List[str]:
        files_page = self._client.vector_stores.files.list(
            vector_store_id=self._vector_store_id,
            limit=100,
        )
        file_ids = [f.id for f in files_page.data]
        next_page = attribute_access.optional(files_page, "has_more", False)
        cursor = attribute_access.optional(files_page, "last_id", None)
        while next_page and cursor:
            page = self._client.vector_stores.files.list(
                vector_store_id=self._vector_store_id,
                after=cursor,
                limit=100,
            )
            file_ids.extend([f.id for f in page.data])
            next_page = attribute_access.optional(page, "has_more", False)
            cursor = attribute_access.optional(page, "last_id", None)
        return file_ids

    def clear_vector_store_and_storage(self) -> int:
        file_ids = self.list_all_file_ids()
        deleted = 0
        for fid in file_ids:
            try:
                self._client.vector_stores.files.delete(
                    vector_store_id=self._vector_store_id,
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
        folder: str | Path,
        *,
        patterns: Sequence[str] = ("*.pdf", "*.txt", "*.doc", "*.docx"),
    ) -> tuple[list[str], list[str]]:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {folder_path}")

        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(folder_path.glob(pattern))
        if not paths:
            return [], []

        uploaded: list[str] = []
        failed: list[str] = []
        for path in paths:
            try:
                self._upload_single_file(path)
                uploaded.append(path.name)
            except Exception:
                failed.append(path.name)
        return uploaded, failed

    def _upload_single_file(self, path: Path) -> None:
        with open(path, "rb") as handle:
            uploaded = self._client.files.create(file=handle, purpose="user_data")

        attempts = 0
        while attempts < self._config.max_poll_attempts:
            f_info = self._client.files.retrieve(uploaded.id)
            status = attribute_access.optional(f_info, "status", None)
            if status == "processed":
                break
            if status == "error":
                raise RuntimeError(f"OpenAI file processing failed for {path.name}")
            time.sleep(self._config.poll_interval_seconds)
            attempts += 1
        else:
            raise TimeoutError(f"Timed out waiting for OpenAI file processing: {path.name}")

        self._client.vector_stores.files.create(
            vector_store_id=self._vector_store_id,
            file_id=uploaded.id,
        )

    def file_search_query(
        self,
        question: str,
        *,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        max_results: int = 10,
        score_threshold: float = 0.2,
    ) -> str:
        resolved_model = model or self._config.default_model
        resolved_instructions = instructions or DEFAULT_FILE_SEARCH_INSTRUCTIONS
        response = self._client.responses.create(
            model=resolved_model,
            instructions=resolved_instructions,
            input=question,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [self._vector_store_id],
                    "max_num_results": max_results,
                    "ranking_options": {
                        "ranker": "auto",
                        "score_threshold": score_threshold,
                    },
                }
            ],
        )
        return str(attribute_access.optional(response, "output_text", "") or "")


def resolve_openai_client(ctx: ToolWiringContext) -> Any | None:
    client = ctx.extras.get("openai_client")
    if client is not None:
        return client
    api_key = ctx.extras.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def resolve_vector_store_id(
    ctx: ToolWiringContext,
    override: Optional[str],
    *,
    config: Optional[OpenAIVectorStoreToolConfig] = None,
) -> Optional[str]:
    if override and override.strip():
        return override.strip()
    extra = ctx.extras.get("openai_vector_store_id")
    if extra and str(extra).strip():
        return str(extra).strip()
    resolved = config or openai_vector_store_config_from_env()
    return resolved.vector_store_id


def resolve_file_search_instructions(
    ctx: ToolWiringContext,
    override: Optional[str],
) -> str:
    if override and override.strip():
        return override.strip()
    custom = ctx.extras.get("openai_file_search_instructions")
    if custom and str(custom).strip():
        return str(custom).strip()
    registry = ctx.extras.get("prompt_registry")
    if registry is not None:
        try:
            localized = registry.resolve_localized(prompt_id="knowledge_openai_strict_system")
            if localized.system and localized.system.strip():
                return localized.system.strip()
        except Exception:
            pass
    return DEFAULT_FILE_SEARCH_INSTRUCTIONS


def resolve_ops(
    ctx: ToolWiringContext,
    vector_store_id: Optional[str],
) -> tuple[Optional[OpenAIVectorStoreOps], str]:
    client = resolve_openai_client(ctx)
    if client is None:
        return None, "openai_client_not_configured"
    vs_id = resolve_vector_store_id(ctx, vector_store_id)
    if not vs_id:
        return None, "vector_store_id_not_configured"
    config = ctx.extras.get("openai_vector_store_config")
    if config is not None and not isinstance(config, OpenAIVectorStoreToolConfig):
        config = None
    return OpenAIVectorStoreOps(client, vs_id, config=config), "ok"


def perform_openai_file_search_query(
    ctx: ToolWiringContext,
    params: OpenAiFileSearchQueryInput,
) -> OpenAiFileSearchQueryOutput:
    ops, reason = resolve_ops(ctx, params.vector_store_id)
    if ops is None:
        return OpenAiFileSearchQueryOutput(used=False, reason=reason)

    try:
        ops.ensure_vector_store_exists()
    except Exception:
        return OpenAiFileSearchQueryOutput(
            used=False,
            reason="vector_store_not_found",
            vector_store_id=ops.vector_store_id,
        )

    instructions = resolve_file_search_instructions(ctx, params.instructions)
    model = params.model or openai_vector_store_config_from_env().default_model
    try:
        answer = ops.file_search_query(
            params.query,
            model=model,
            instructions=instructions,
            max_results=params.max_results,
            score_threshold=params.score_threshold,
        )
    except Exception:
        return OpenAiFileSearchQueryOutput(
            used=False,
            reason="file_search_failed",
            vector_store_id=ops.vector_store_id,
            model=model,
        )

    if not answer.strip():
        return OpenAiFileSearchQueryOutput(
            used=False,
            reason="empty_response",
            vector_store_id=ops.vector_store_id,
            model=model,
        )

    return OpenAiFileSearchQueryOutput(
        used=True,
        answer_text=answer,
        context_text=answer,
        reason="ok",
        vector_store_id=ops.vector_store_id,
        model=model,
    )


def perform_openai_vector_store_upload(
    ctx: ToolWiringContext,
    params: OpenAiVectorStoreUploadInput,
) -> OpenAiVectorStoreUploadOutput:
    ops, reason = resolve_ops(ctx, params.vector_store_id)
    if ops is None:
        return OpenAiVectorStoreUploadOutput(used=False, reason=reason)

    try:
        ops.ensure_vector_store_exists()
    except Exception:
        return OpenAiVectorStoreUploadOutput(
            used=False,
            reason="vector_store_not_found",
            vector_store_id=ops.vector_store_id,
        )

    try:
        uploaded, failed = ops.upload_folder(params.folder_path, patterns=params.patterns)
    except FileNotFoundError:
        return OpenAiVectorStoreUploadOutput(
            used=False,
            reason="folder_not_found",
            vector_store_id=ops.vector_store_id,
        )
    except Exception:
        return OpenAiVectorStoreUploadOutput(
            used=False,
            reason="upload_failed",
            vector_store_id=ops.vector_store_id,
        )

    if not uploaded and not failed:
        return OpenAiVectorStoreUploadOutput(
            used=False,
            reason="no_matching_files",
            vector_store_id=ops.vector_store_id,
        )

    return OpenAiVectorStoreUploadOutput(
        used=bool(uploaded),
        uploaded_count=len(uploaded),
        file_names=uploaded,
        failed_files=failed,
        reason="ok" if uploaded else "all_files_failed",
        vector_store_id=ops.vector_store_id,
    )


def perform_openai_vector_store_clear(
    ctx: ToolWiringContext,
    params: OpenAiVectorStoreClearInput,
) -> OpenAiVectorStoreClearOutput:
    ops, reason = resolve_ops(ctx, params.vector_store_id)
    if ops is None:
        return OpenAiVectorStoreClearOutput(used=False, reason=reason)

    try:
        ops.ensure_vector_store_exists()
    except Exception:
        return OpenAiVectorStoreClearOutput(
            used=False,
            reason="vector_store_not_found",
            vector_store_id=ops.vector_store_id,
        )

    try:
        deleted = ops.clear_vector_store_and_storage()
    except Exception:
        return OpenAiVectorStoreClearOutput(
            used=False,
            reason="clear_failed",
            vector_store_id=ops.vector_store_id,
        )

    return OpenAiVectorStoreClearOutput(
        used=True,
        deleted_count=deleted,
        reason="ok" if deleted else "no_files_to_delete",
        vector_store_id=ops.vector_store_id,
    )
