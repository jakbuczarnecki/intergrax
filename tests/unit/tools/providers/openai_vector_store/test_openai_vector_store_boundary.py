# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_openai_vector_store_tool_service_has_no_openai_sdk_import() -> None:
    root = Path(__file__).resolve().parents[5]
    service_path = (
        root / "intergrax" / "tools" / "providers" / "openai_vector_store" / "service.py"
    )
    source = service_path.read_text(encoding="utf-8")
    forbidden_tokens = (
        "from openai",
        "import openai",
        ".vector_stores",
        ".files.create",
        ".responses.create",
        "integrations/providers/managed_retrieval/openai",
        "integrations.providers.managed_retrieval.openai",
    )
    for token in forbidden_tokens:
        assert token not in source, f"forbidden token in tool service: {token}"


def test_openai_vector_store_package_has_no_direct_openai_import() -> None:
    root = Path(__file__).resolve().parents[5]
    package_dir = root / "intergrax" / "tools" / "providers" / "openai_vector_store"
    for path in package_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from openai" not in source
        assert "import openai" not in source
