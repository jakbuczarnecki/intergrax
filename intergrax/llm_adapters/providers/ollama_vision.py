# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Ollama vision chat call for image captioning (A4-018, FRZ-CTR-02, FRZ-CTR-06).

The Ollama SDK is imported here, inside the provider package, and only when the call is made; modules
outside the provider package (for example ``intergrax.multimedia.images_loader``) delegate to this
function instead of importing the SDK.
"""

from __future__ import annotations

from collections.abc import Mapping


def chat_with_image(*, prompt: str, image_path: str, model: str) -> str:
    """Send one user message with one image to the local Ollama server and return the reply text."""
    try:
        import ollama
    except ModuleNotFoundError as exc:
        if exc.name != "ollama":
            raise
        from intergrax.llm_adapters.llm_provider_registry import LLMAdapterDependencyError

        raise LLMAdapterDependencyError(
            "Ollama image captioning requires dependency 'ollama'. "
            "Install it with 'Intergrax-ai[llm-ollama]' before using the Ollama vision bridge."
        ) from exc

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ],
    )
    message = response["message"]
    content = message["content"] if isinstance(message, Mapping) else message.content
    return str(content or "")


__all__ = ["chat_with_image"]
