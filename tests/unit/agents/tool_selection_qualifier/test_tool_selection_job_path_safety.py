# © Artur Czarnecki. All rights reserved.

"""Path safety for Q2 wrong-tool write_file invocations."""

from __future__ import annotations

import pytest

from tool_selection_qualifier.steps.tool_selection_job import _qualification_safe_relative_path

pytestmark = pytest.mark.unit


def test_unsafe_absolute_path_falls_back_to_default() -> None:
    assert _qualification_safe_relative_path("/etc/passwd", "qualification-draft.md") == "qualification-draft.md"


def test_unsafe_traversal_path_falls_back_to_default() -> None:
    assert _qualification_safe_relative_path("../escape.md", "qualification-draft.md") == "qualification-draft.md"


def test_safe_relative_path_is_preserved() -> None:
    assert _qualification_safe_relative_path("notes/incident.md", "qualification-draft.md") == "notes/incident.md"
