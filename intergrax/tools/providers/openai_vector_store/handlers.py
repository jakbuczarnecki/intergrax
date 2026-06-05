# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.openai_vector_store.contracts import (
    OpenAiFileSearchQueryInput,
    OpenAiFileSearchQueryOutput,
    OpenAiVectorStoreClearInput,
    OpenAiVectorStoreClearOutput,
    OpenAiVectorStoreUploadInput,
    OpenAiVectorStoreUploadOutput,
)
from intergrax.tools.providers.openai_vector_store.service import (
    perform_openai_file_search_query,
    perform_openai_vector_store_clear,
    perform_openai_vector_store_upload,
)


class OpenAiFileSearchQueryHandler(
    ServiceToolHandler[OpenAiFileSearchQueryInput, OpenAiFileSearchQueryOutput]
):
    _service = perform_openai_file_search_query


class OpenAiVectorStoreUploadHandler(
    ServiceToolHandler[OpenAiVectorStoreUploadInput, OpenAiVectorStoreUploadOutput]
):
    _service = perform_openai_vector_store_upload


class OpenAiVectorStoreClearHandler(
    ServiceToolHandler[OpenAiVectorStoreClearInput, OpenAiVectorStoreClearOutput]
):
    _service = perform_openai_vector_store_clear
