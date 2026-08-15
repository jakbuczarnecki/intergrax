from __future__ import annotations

from codecs import BOM_UTF8
from pathlib import Path

import pytest
from chardet import detect

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.parsers.text_smart_parser import TextLoaderParser


def test_loads_utf8_polish_text_and_preserves_newlines(tmp_path: Path):
    source = tmp_path / "sample.txt"
    content = "Zażółć gęślą jaźń\r\nlinia druga\n"
    source.write_text(content, encoding="utf-8", newline="")

    fragments = TextLoaderParser().load(str(source))

    assert len(fragments) == 1
    fragment = fragments[0]
    assert isinstance(fragment, ParsedDocumentFragment)
    assert fragment.text == content
    assert fragment.metadata["source"] == str(source)
    assert fragment.metadata["parser"] == "text_loader"
    assert fragment.metadata["position"] == 0


def test_loads_utf8_bom_without_leaking_bom_into_text(tmp_path: Path):
    source = tmp_path / "bom.txt"
    content = "BOM content\n"
    source.write_bytes(BOM_UTF8 + content.encode("utf-8"))

    fragments = TextLoaderParser().load(str(source))

    assert len(fragments) == 1
    assert fragments[0].text == content


def test_loads_cp1251_text_via_real_chardet_fallback(tmp_path: Path):
    source = tmp_path / "cp1251.txt"
    content = (
        "Это длинный текст на русском языке. Проверка кодировки, чтения и "
        "точного восстановления исходного текста. Москва, Санкт-Петербург "
        "и другие города.\n"
    ) * 8
    encoded = content.encode("cp1251")

    with pytest.raises(UnicodeDecodeError):
        encoded.decode("utf-8")

    detected = detect(encoded)
    assert detected["encoding"] == "windows-1251"
    assert detected["confidence"] >= 0.8

    source.write_bytes(encoded)
    fragments = TextLoaderParser().load(str(source))

    assert len(fragments) == 1
    assert isinstance(fragments[0], ParsedDocumentFragment)
    assert fragments[0].text == content


def test_loads_empty_file_as_one_empty_fragment(tmp_path: Path):
    source = tmp_path / "empty.txt"
    source.write_bytes(b"")

    fragments = TextLoaderParser().load(str(source))

    assert len(fragments) == 1
    assert fragments[0].text == ""


def test_missing_file_fails_fast(tmp_path: Path):
    source = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        TextLoaderParser().load(str(source))


def test_undecodable_bytes_fail_fast(tmp_path: Path):
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"\xff\xfe\xfa\xfb")

    with pytest.raises(UnicodeDecodeError):
        TextLoaderParser().load(str(source))
