# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Google Vertex AI Gemini adapter (ADC / service account, no API key)."""

from __future__ import annotations

import os
from typing import Optional

from google import genai

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter


class VertexGeminiChatAdapter(GeminiChatAdapter):
    """
    Gemini on Vertex AI via Application Default Credentials.

    Requires ``INTERGRAX_VERTEX_PROJECT`` and optional ``INTERGRAX_VERTEX_LOCATION``.
    """

    ENV_PROJECT = "INTERGRAX_VERTEX_PROJECT"
    ENV_LOCATION = "INTERGRAX_DEFAULT_VERTEX_LOCATION"
    ENV_MODEL = "INTERGRAX_DEFAULT_VERTEX_GEMINI_MODEL"
    DEFAULT_LOCATION = "us-central1"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        **defaults,
    ):
        resolved_project = (project or os.getenv(self.ENV_PROJECT) or "").strip()
        if not resolved_project and client is None:
            raise RuntimeError(
                "INTERGRAX_VERTEX_PROJECT must be set for Vertex Gemini adapter."
            )

        resolved_location = (
            location or os.getenv(self.ENV_LOCATION) or self.DEFAULT_LOCATION
        ).strip()

        if client is None:
            client = genai.Client(
                vertexai=True,
                project=resolved_project,
                location=resolved_location,
            )

        resolved_model = model or os.getenv(self.ENV_MODEL) or self.DEFAULT_MODEL

        super().__init__(client=client, model=resolved_model, **defaults)
        self.provider = LLMProvider.VERTEX_GEMINI
