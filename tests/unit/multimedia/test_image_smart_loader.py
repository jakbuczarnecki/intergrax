from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
from intergrax.multimedia import image_smart_loader
from intergrax.multimedia.image_smart_loader import ImageSmartLoader


@pytest.fixture
def image_fixture(monkeypatch):
    image = SimpleNamespace(
        size=(100, 50),
        info={"dpi": (72, 72)},
        format="PNG",
        _getexif=lambda: {},
    )
    monkeypatch.setattr(image_smart_loader.Image, "open", lambda _: image)
    monkeypatch.setattr(image_smart_loader, "pytesseract", None)
    return image


def _ollama_adapter(model: str | None = None) -> LangChainOllamaAdapter:
    adapter = object.__new__(LangChainOllamaAdapter)
    adapter.defaults = {"model": model} if model is not None else {}
    adapter.chat = SimpleNamespace()
    return adapter


def _load_caption(monkeypatch, tmp_path, adapter, image_fixture):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"not-read")
    caption_call = Mock(return_value="caption")
    images_loader = ModuleType("intergrax.multimedia.images_loader")
    images_loader.transcribe_image = caption_call
    monkeypatch.setitem(sys.modules, "intergrax.multimedia.images_loader", images_loader)

    document = ImageSmartLoader(
        str(image_path),
        text_mode="caption",
        caption_llm=adapter,
    ).load()[0]
    return document, caption_call


def test_ollama_explicit_model_is_used_for_runtime_and_metadata(
    monkeypatch, tmp_path, image_fixture
):
    document, caption_call = _load_caption(
        monkeypatch,
        tmp_path,
        _ollama_adapter("vision-custom:7b"),
        image_fixture,
    )

    assert caption_call.call_args.kwargs["model"] == "vision-custom:7b"
    assert document.metadata["caption_model_inferred"] == "vision-custom:7b"


def test_ollama_missing_model_uses_vision_fallback_for_runtime_and_metadata(
    monkeypatch, tmp_path, image_fixture
):
    document, caption_call = _load_caption(
        monkeypatch,
        tmp_path,
        _ollama_adapter(),
        image_fixture,
    )

    assert caption_call.call_args.kwargs["model"] == "llava-llama3:latest"
    assert document.metadata["caption_model_inferred"] == "llava-llama3:latest"


def test_non_ollama_caption_adapter_has_no_ollama_provenance(
    monkeypatch, tmp_path, image_fixture
):
    adapter = SimpleNamespace(describe_image=Mock(return_value="native caption"))

    document, caption_call = _load_caption(monkeypatch, tmp_path, adapter, image_fixture)

    assert adapter.describe_image.called
    assert not caption_call.called
    assert document.metadata["caption_model_inferred"] is None


def test_ocr_only_has_no_caption_model_provenance(monkeypatch, tmp_path, image_fixture):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"not-read")

    document = ImageSmartLoader(str(image_path), text_mode="ocr").load()[0]

    assert document.metadata["caption_model_inferred"] is None
