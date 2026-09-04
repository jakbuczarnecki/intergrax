# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for S1 scale harness."""

from __future__ import annotations

from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parent
_FORBIDDEN_SNIPPETS = (
    "dict[str, Any]",
    "dict[str, object]",
    ": Any",
    "getattr(",
    "setattr(",
    "hasattr(",
    "importlib",
    "type: ignore",
    "except Exception:",
)


@pytest.mark.parametrize("relative_path", sorted(_PACKAGE.glob("*.py")))
def test_s1_scale_package_static_quality(relative_path: Path) -> None:
    if relative_path.name == "test_static_quality.py":
        return
    text = relative_path.read_text(encoding="utf-8")
    for snippet in _FORBIDDEN_SNIPPETS:
        assert snippet not in text, f"{relative_path.name} contains forbidden snippet: {snippet}"
