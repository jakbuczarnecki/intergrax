# © Artur Czarnecki. All rights reserved.

"""Pydantic contracts for OpenAI managed vector store catalog tools."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OpenAiFileSearchQueryInput(BaseModel):
    """Query a hosted OpenAI vector store via Responses API ``file_search``."""

    query: str = Field(..., min_length=1, description="Natural language question for file_search.")
    vector_store_id: Optional[str] = Field(
        default=None,
        description="OpenAI vector store id; falls back to wiring/env default.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Responses API model; defaults to INTERGRAX_OPENAI_FILE_SEARCH_MODEL or gpt-4o-mini.",
    )
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum file_search hits.")
    score_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="file_search ranking score threshold.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Override strict RAG system instructions; default from prompt registry or built-in.",
    )


class OpenAiFileSearchQueryOutput(BaseModel):
    used: bool
    answer_text: str = ""
    context_text: str = ""
    reason: str = ""
    vector_store_id: str = ""
    model: str = ""


class OpenAiVectorStoreUploadInput(BaseModel):
    """Upload local documents into an OpenAI managed vector store."""

    folder_path: str = Field(..., min_length=1, description="Directory containing files to upload.")
    vector_store_id: Optional[str] = Field(
        default=None,
        description="Target vector store id; falls back to wiring/env default.",
    )
    patterns: tuple[str, ...] = Field(
        default=("*.pdf", "*.txt", "*.doc", "*.docx"),
        description="Glob patterns applied within folder_path.",
    )


class OpenAiVectorStoreUploadOutput(BaseModel):
    used: bool
    uploaded_count: int = 0
    file_names: list[str] = Field(default_factory=list)
    failed_files: list[str] = Field(default_factory=list)
    reason: str = ""
    vector_store_id: str = ""


class OpenAiVectorStoreClearInput(BaseModel):
    """Remove all files from an OpenAI managed vector store and underlying storage."""

    vector_store_id: Optional[str] = Field(
        default=None,
        description="Vector store to clear; falls back to wiring/env default.",
    )


class OpenAiVectorStoreClearOutput(BaseModel):
    used: bool
    deleted_count: int = 0
    reason: str = ""
    vector_store_id: str = ""
