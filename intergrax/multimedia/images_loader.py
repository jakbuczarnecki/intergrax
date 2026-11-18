# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Image captioning bridge used by ``ImageSmartLoader`` for Ollama adapters.

The Ollama SDK call lives in the provider package (``intergrax.llm_adapters.providers.ollama_vision``);
this module imports no vendor SDK and no notebook helper (A4-018, FRZ-CTR-02, FRZ-CTR-06).
"""

from __future__ import annotations


def transcribe_image(prompt: str, image_path: str, model: str = "llava-llama3:latest") -> str:
    """Describe the image at ``image_path`` with the Ollama vision model ``model``."""
    from intergrax.llm_adapters.providers.ollama_vision import chat_with_image

    return chat_with_image(prompt=prompt, image_path=image_path, model=model)
