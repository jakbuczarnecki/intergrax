# © Artur Czarnecki. All rights reserved.

"""Map AttachmentRef values to OpenAI-compatible multimodal content parts (Phase W-ML.1)."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from intergrax.llm.messages import AttachmentRef


class AttachmentContentMapper(Protocol):
    """Maps harness attachment references to vendor-neutral content parts."""

    def to_openai_content_parts(self, attachment: AttachmentRef) -> list[dict[str, Any]]:
        ...


_IMAGE_TYPES: frozenset[str] = frozenset(
    {"image", "png", "jpg", "jpeg", "gif", "webp", "image/png", "image/jpeg", "image/webp"}
)


class DefaultAttachmentContentMapper:
    """Default mapper: HTTPS URLs pass-through; file:// images become data URLs."""

    def to_openai_content_parts(self, attachment: AttachmentRef) -> list[dict[str, Any]]:
        attachment_type = attachment.type.lower()
        if attachment_type in _IMAGE_TYPES or attachment_type.startswith("image/"):
            return [self._image_part(attachment)]
        return [
            {
                "type": "text",
                "text": f"[attachment:{attachment.id} type={attachment.type}]",
            }
        ]

    def _image_part(self, attachment: AttachmentRef) -> dict[str, Any]:
        uri = attachment.uri
        if uri.startswith("http://") or uri.startswith("https://"):
            return {"type": "image_url", "image_url": {"url": uri}}
        if uri.startswith("file://"):
            data_url = self._file_uri_to_data_url(uri, attachment.type)
            return {"type": "image_url", "image_url": {"url": data_url}}
        return {
            "type": "text",
            "text": f"[unsupported-image-uri:{attachment.id}]",
        }

    def _file_uri_to_data_url(self, uri: str, attachment_type: str) -> str:
        path = _resolve_file_uri(uri)
        if not path.is_file():
            raise FileNotFoundError(f"Attachment file not found: {path}")
        mime = mimetypes.guess_type(path.name)[0]
        if mime is None:
            mime = attachment_type if "/" in attachment_type else f"image/{attachment_type}"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


def _resolve_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Not a file URI: {uri}")
    path_str = unquote(parsed.path)
    if path_str.startswith("/") and len(path_str) > 2 and path_str[2] == ":":
        path_str = path_str[1:]
    return Path(path_str)


def build_openai_user_content(
    *,
    text: str,
    attachments: list[AttachmentRef],
    mapper: AttachmentContentMapper,
) -> str | list[dict[str, Any]]:
    if not attachments:
        return text
    parts: list[dict[str, Any]] = []
    if text.strip():
        parts.append({"type": "text", "text": text})
    for attachment in attachments:
        parts.extend(mapper.to_openai_content_parts(attachment))
    return parts if parts else text
