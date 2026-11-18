# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-D NT-RW-RUN-07 (NT-ROW-005 A4-018, NT-ROW-037 FRZ-CTR-02, NT-ROW-039 FRZ-CTR-06; alias NT-CTR-02-1,
NT-CFG-06-1): no ``ollama`` SDK import outside the provider package.

Before the change the Tier-0 image loader imported the Ollama SDK (and IPython) at module level and the Tier-2
model-routing qualifier imported ``ollama.ResponseError`` to classify failures. The vision call now lives in the
provider package and the qualifier classifies the typed provider errors. A meta-path finder that refuses the
SDK proves that the non-provider modules no longer need it.
"""

from __future__ import annotations

import ast
import importlib
import importlib.abc
import importlib.util
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PROVIDER_PACKAGE = _REPO / "intergrax" / "llm_adapters" / "providers"
_SDK = "ollama"


def _imports_of(path: Path) -> set[str]:
    """Top-level module names imported anywhere in ``path`` (any nesting level, including try blocks)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _scanned_files() -> list[Path]:
    files = [path for path in sorted((_REPO / "intergrax").rglob("*.py")) if _PROVIDER_PACKAGE not in path.parents]
    files += sorted((_REPO / "agents").rglob("*.py"))
    return files


def test_no_module_outside_the_provider_package_imports_the_ollama_sdk() -> None:
    offenders = [path.relative_to(_REPO).as_posix() for path in _scanned_files() if _SDK in _imports_of(path)]
    assert offenders == []


def test_the_image_loader_has_no_notebook_import() -> None:
    assert "IPython" not in _imports_of(_REPO / "intergrax" / "multimedia" / "images_loader.py")


class _RefuseSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None) -> None:
        if fullname == _SDK or fullname.startswith(f"{_SDK}."):
            raise ImportError(f"the {_SDK} SDK is refused outside the provider package in this test")
        return None


@pytest.fixture
def sdk_refused(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in list(sys.modules):
        if name == _SDK or name.startswith(f"{_SDK}."):
            monkeypatch.delitem(sys.modules, name)
    for name in (
        "intergrax.multimedia.images_loader",
        "model_routing_qualifier.qualification_types",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    finder = _RefuseSdk()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def test_the_image_loader_and_the_qualifier_import_without_the_sdk(sdk_refused: None) -> None:
    images_loader = importlib.import_module("intergrax.multimedia.images_loader")
    assert callable(images_loader.transcribe_image)
    qualification_types = importlib.import_module("model_routing_qualifier.qualification_types")
    assert callable(qualification_types.classify_invocation_failure)


def test_image_captioning_with_an_injected_adapter_needs_no_sdk(
    sdk_refused: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from intergrax.multimedia import image_smart_loader
    from intergrax.multimedia.image_smart_loader import ImageSmartLoader

    image = SimpleNamespace(size=(100, 50), info={"dpi": (72, 72)}, format="PNG", _getexif=lambda: {})
    monkeypatch.setattr(image_smart_loader.Image, "open", lambda _: image)
    monkeypatch.setattr(image_smart_loader, "pytesseract", None)
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"not-read")
    adapter = SimpleNamespace(describe_image=Mock(return_value="a caption"))

    document = ImageSmartLoader(
        str(image_path), text_mode="caption", caption_llm=adapter, tenant_id="tenant-test"
    ).load()[0]

    assert adapter.describe_image.called
    assert "a caption" in document.content


def test_the_vision_bridge_delegates_to_the_provider_package(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.find_spec("intergrax.llm_adapters.providers.ollama_vision")
    assert spec is not None, "the Ollama vision call must live in the provider package"
    calls: list[dict[str, Any]] = []

    fake_sdk = ModuleType(_SDK)

    def _chat(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return {"message": {"content": "described"}}

    fake_sdk.chat = _chat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _SDK, fake_sdk)
    monkeypatch.delitem(sys.modules, "intergrax.multimedia.images_loader", raising=False)
    images_loader = importlib.import_module("intergrax.multimedia.images_loader")

    text = images_loader.transcribe_image(prompt="Describe.", image_path="img.png", model="vision:7b")

    assert text == "described"
    assert calls == [
        {"model": "vision:7b", "messages": [{"role": "user", "content": "Describe.", "images": ["img.png"]}]}
    ]
    loader_source = (_REPO / "intergrax" / "multimedia" / "images_loader.py").read_text(encoding="utf-8")
    assert "intergrax.llm_adapters.providers.ollama_vision" in loader_source
