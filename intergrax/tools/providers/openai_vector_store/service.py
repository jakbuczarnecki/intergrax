# © Artur Czarnecki. All rights reserved.

"""OpenAI managed vector store catalog tools — neutral managed retrieval boundary."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.managed_retrieval import (
    ManagedRetrievalBackend,
    ManagedRetrievalQueryRequest,
    ManagedRetrievalQueryError,
    ManagedRetrievalResourceNotFoundError,
    ManagedRetrievalUploadError,
)
from intergrax.integrations.providers.managed_retrieval.openai.bundle import (
    try_create_openai_managed_retrieval_from_env,
)
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


def resolve_managed_retrieval(ctx: ToolWiringContext) -> ManagedRetrievalBackend | None:
    if ctx.managed_retrieval is not None:
        return ctx.managed_retrieval
    extra = ctx.extras.get("managed_retrieval")
    if extra is not None:
        return extra
    return try_create_openai_managed_retrieval_from_env()


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


def resolve_tool_config(ctx: ToolWiringContext) -> OpenAIVectorStoreToolConfig:
    config = ctx.extras.get("openai_vector_store_config")
    if isinstance(config, OpenAIVectorStoreToolConfig):
        return config
    return openai_vector_store_config_from_env()


def resolve_backend_and_store(
    ctx: ToolWiringContext,
    vector_store_id: Optional[str],
) -> tuple[ManagedRetrievalBackend | None, str | None, str]:
    backend = resolve_managed_retrieval(ctx)
    if backend is None:
        return None, None, "managed_retrieval_not_configured"
    store_id = resolve_vector_store_id(ctx, vector_store_id, config=resolve_tool_config(ctx))
    if not store_id:
        return None, None, "vector_store_id_not_configured"
    return backend, store_id, "ok"


def perform_openai_file_search_query(
    ctx: ToolWiringContext,
    params: OpenAiFileSearchQueryInput,
) -> OpenAiFileSearchQueryOutput:
    backend, store_id, reason = resolve_backend_and_store(ctx, params.vector_store_id)
    if backend is None or store_id is None:
        return OpenAiFileSearchQueryOutput(used=False, reason=reason)

    try:
        backend.ensure_store_exists(store_id)
    except ManagedRetrievalResourceNotFoundError:
        return OpenAiFileSearchQueryOutput(
            used=False,
            reason="vector_store_not_found",
            vector_store_id=store_id,
        )

    instructions = resolve_file_search_instructions(ctx, params.instructions)
    tool_config = resolve_tool_config(ctx)
    model = params.model or tool_config.default_model
    try:
        answer = backend.query(
            ManagedRetrievalQueryRequest(
                store_id=store_id,
                question=params.query,
                model=model,
                instructions=instructions,
                max_results=params.max_results,
                score_threshold=params.score_threshold,
            )
        )
    except ManagedRetrievalQueryError:
        return OpenAiFileSearchQueryOutput(
            used=False,
            reason="file_search_failed",
            vector_store_id=store_id,
            model=model,
        )

    if not answer.strip():
        return OpenAiFileSearchQueryOutput(
            used=False,
            reason="empty_response",
            vector_store_id=store_id,
            model=model,
        )

    return OpenAiFileSearchQueryOutput(
        used=True,
        answer_text=answer,
        context_text=answer,
        reason="ok",
        vector_store_id=store_id,
        model=model,
    )


def perform_openai_vector_store_upload(
    ctx: ToolWiringContext,
    params: OpenAiVectorStoreUploadInput,
) -> OpenAiVectorStoreUploadOutput:
    backend, store_id, reason = resolve_backend_and_store(ctx, params.vector_store_id)
    if backend is None or store_id is None:
        return OpenAiVectorStoreUploadOutput(used=False, reason=reason)

    try:
        backend.ensure_store_exists(store_id)
    except ManagedRetrievalResourceNotFoundError:
        return OpenAiVectorStoreUploadOutput(
            used=False,
            reason="vector_store_not_found",
            vector_store_id=store_id,
        )

    try:
        upload_result = backend.upload_folder(
            store_id,
            params.folder_path,
            patterns=params.patterns,
        )
    except FileNotFoundError:
        return OpenAiVectorStoreUploadOutput(
            used=False,
            reason="folder_not_found",
            vector_store_id=store_id,
        )
    except ManagedRetrievalUploadError:
        return OpenAiVectorStoreUploadOutput(
            used=False,
            reason="upload_failed",
            vector_store_id=store_id,
        )

    uploaded = list(upload_result.uploaded_names)
    failed = list(upload_result.failed_names)
    if not uploaded and not failed:
        return OpenAiVectorStoreUploadOutput(
            used=False,
            reason="no_matching_files",
            vector_store_id=store_id,
        )

    return OpenAiVectorStoreUploadOutput(
        used=bool(uploaded),
        uploaded_count=len(uploaded),
        file_names=uploaded,
        failed_files=failed,
        reason="ok" if uploaded else "all_files_failed",
        vector_store_id=store_id,
    )


def perform_openai_vector_store_clear(
    ctx: ToolWiringContext,
    params: OpenAiVectorStoreClearInput,
) -> OpenAiVectorStoreClearOutput:
    backend, store_id, reason = resolve_backend_and_store(ctx, params.vector_store_id)
    if backend is None or store_id is None:
        return OpenAiVectorStoreClearOutput(used=False, reason=reason)

    try:
        backend.ensure_store_exists(store_id)
    except ManagedRetrievalResourceNotFoundError:
        return OpenAiVectorStoreClearOutput(
            used=False,
            reason="vector_store_not_found",
            vector_store_id=store_id,
        )

    try:
        deleted = backend.clear_store(store_id)
    except Exception:
        return OpenAiVectorStoreClearOutput(
            used=False,
            reason="clear_failed",
            vector_store_id=store_id,
        )

    return OpenAiVectorStoreClearOutput(
        used=True,
        deleted_count=deleted,
        reason="ok" if deleted else "no_files_to_delete",
        vector_store_id=store_id,
    )
