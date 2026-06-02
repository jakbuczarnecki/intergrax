from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.llm.messages import AttachmentRef, ChatMessage
from intergrax.llm_adapters._shared.messages import map_chat_completion_messages
from intergrax.llm_adapters._shared.multimodal_content import DefaultAttachmentContentMapper
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter


def test_openai_adapter_declares_vision_support() -> None:
    adapter = OpenAIChatCompletionsAdapter(
        client=MagicMock(),
        model="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
    )
    assert adapter.supports_vision() is True


def test_map_chat_completion_includes_image_part_for_https_attachment() -> None:
    mapped = map_chat_completion_messages(
        system_text="",
        convo=[
            ChatMessage(
                role="user",
                content="describe this",
                attachments=[
                    AttachmentRef(
                        id="img-1",
                        type="image",
                        uri="https://example.com/sample.png",
                    )
                ],
            )
        ],
        include_multimodal=True,
        attachment_mapper=DefaultAttachmentContentMapper(),
    )
    content = mapped[0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"


def test_file_uri_image_maps_to_data_url(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    mapped = map_chat_completion_messages(
        system_text="",
        convo=[
            ChatMessage(
                role="user",
                content="inspect",
                attachments=[
                    AttachmentRef(
                        id="img-local",
                        type="png",
                        uri=image_path.as_uri(),
                    )
                ],
            )
        ],
        include_multimodal=True,
    )
    content = mapped[0]["content"]
    assert isinstance(content, list)
    assert str(content[1]["image_url"]["url"]).startswith("data:image/")
