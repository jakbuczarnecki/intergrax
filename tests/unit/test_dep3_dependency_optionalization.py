from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.chroma import opens as chroma_opens
from intergrax.integrations.providers.vector_store.pinecone import opens as pinecone_opens
from intergrax.integrations.providers.vector_store.qdrant import opens as qdrant_opens

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _project() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def test_dep3_capabilities_are_not_core_dependencies() -> None:
    project = _project()
    core = project["dependencies"]
    extras = project["optional-dependencies"]

    for package in (
        "chromadb",
        "qdrant-client",
        "pinecone",
        "beautifulsoup4",
        "trafilatura",
        "python-docx",
        "openpyxl",
        "xlrd",
        "pytesseract",
        "pillow",
        "PyMuPDF",
        "yt-dlp",
        "webvtt-py",
        "opencv-python-headless",
        "streamlit",
        "fastmcp",
    ):
        assert not any(dependency.lower().startswith(package.lower()) for dependency in core)

    assert extras["vector-chroma"] == ["chromadb==1.4.1"]
    assert extras["vector-qdrant"] == ["qdrant-client>=1.9,<2"]
    assert extras["vector-pinecone"] == ["pinecone>=3.0,<9"]
    assert "beautifulsoup4>=4.12,<5" in extras["parsing-web"]
    assert "trafilatura>=1.8,<3" in extras["parsing-web"]
    assert set(extras["parsing-office"]) == {
        "pandas>=2.1.4,<3",
        "python-docx>=1.1,<2",
        "openpyxl>=3.1,<4",
        "xlrd>=2.0,<3",
        "docx2txt>=0.8,<1",
        "langchain-community>=0.3,<0.5",
    }
    assert extras["parsing-pdf"] == ["PyMuPDF>=1.23,<2", "langchain-community>=0.3,<0.5"]
    assert set(extras["parsing-ocr"]) == {"pytesseract>=0.3,<1", "pillow>=11.0,<13"}
    assert extras["media-youtube"] == ["yt-dlp>=2024.0,<2027"]
    assert "opencv-python-headless==4.9.0.80" in extras["media-video"]
    assert "streamlit>=1.39,<2" in extras["ui-streamlit"]
    assert extras["mcp"] == ["fastmcp>=3.3.1,<4"]
    assert not any(dependency.lower().startswith("requests-cache") for dependency in core)


def test_core_imports_do_not_load_optional_capabilities() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import intergrax
import intergrax.runtime.nexus
import intergrax.harness

print(json.dumps({
    name: any(module == name or module.startswith(name + ".") for module in sys.modules)
    for name in (
        "chromadb", "qdrant_client", "pinecone", "streamlit", "fastmcp",
        "mcp", "yt_dlp", "cv2", "pytesseract", "fitz", "docx", "openpyxl",
        "xlrd", "trafilatura", "bs4",
    )
}))
""",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        name: False
        for name in (
            "chromadb", "qdrant_client", "pinecone", "streamlit", "fastmcp",
            "mcp", "yt_dlp", "cv2", "pytesseract", "fitz", "docx", "openpyxl",
            "xlrd", "trafilatura", "bs4",
        )
    }


@pytest.mark.parametrize(
    ("opener", "blocked", "extra"),
    (
        (chroma_opens._import_chromadb, "chromadb", "vector-chroma"),
        (qdrant_opens._import_qdrant_client, "qdrant_client", "vector-qdrant"),
        (pinecone_opens._import_pinecone, "pinecone", "vector-pinecone"),
    ),
)
def test_vector_missing_dependency_has_controlled_extra_error(
    opener, blocked: str, extra: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def block_dependency(name: str, *args: object, **kwargs: object) -> object:
        if name == blocked or name.startswith(f"{blocked}."):
            raise ModuleNotFoundError(f"No module named '{blocked}'", name=blocked)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_dependency)

    with pytest.raises(IntegrationConfigurationError, match=extra):
        opener()
