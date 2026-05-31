# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

import os
from typing import Dict, Optional

from openai import AzureOpenAI

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter


class AzureOpenAIChatAdapter(OpenAIChatCompletionsAdapter):
    """
    Azure OpenAI via Chat Completions (deployment name as ``model``).

    Delegates chat/tools/stream/structured to :class:`OpenAIChatCompletionsAdapter`.
    """

    _AZURE_CONTEXT_WINDOWS: Dict[str, int] = {}

    def __init__(
        self,
        client: Optional[AzureOpenAI] = None,
        deployment: Optional[str] = None,
        **defaults,
    ):
        endpoint = (os.getenv("INTERGRAX_DEFAULT_AZURE_OPENAI_ENDPOINT", "") or "").strip()
        api_version = (os.getenv("INTERGRAX_DEFAULT_AZURE_OPENAI_API_VERSION", "") or "").strip()
        default_deployment = (os.getenv("INTERGRAX_DEFAULT_AZURE_OPENAI_DEPLOYMENT", "") or "").strip()

        resolved_deployment = (deployment or default_deployment).strip()
        if not endpoint:
            raise RuntimeError(
                "INTERGRAX_DEFAULT_AZURE_OPENAI_ENDPOINT must be configured for Azure OpenAI adapter."
            )
        if not api_version:
            raise RuntimeError(
                "INTERGRAX_DEFAULT_AZURE_OPENAI_API_VERSION must be configured for Azure OpenAI adapter."
            )
        if not resolved_deployment:
            raise RuntimeError(
                "INTERGRAX_DEFAULT_AZURE_OPENAI_DEPLOYMENT or deployment= must be set."
            )

        azure_client = client or AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
        )

        super().__init__(
            client=azure_client,
            model=resolved_deployment,
            provider=LLMProvider.AZURE_OPENAI,
            context_windows=self._AZURE_CONTEXT_WINDOWS,
            **defaults,
        )
        self.deployment = resolved_deployment
