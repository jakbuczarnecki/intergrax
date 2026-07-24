# © Artur Czarnecki. All rights reserved.

"""Unit tests for safe source label projection (LKW-WORKSPACE-CONTENTS-1A)."""

from __future__ import annotations

import pytest
from local_workspace_application.serving.source_projection import (
    local_folder_safe_label,
    safe_source_label,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"C:\Users\Artur\Documents\Contracts", "Contracts"),
        (r"D:\Cases\BUILDLOGIC", "BUILDLOGIC"),
        ("/home/user/projects/specifications", "specifications"),
        (r"C:\Users\Artur\Private\Client-X\Contracts", "Contracts"),
        ("//server/share/folder/specs", "specs"),
        ("/", "Local folder"),
        ("C:\\", "Local folder"),
        ("", "Local folder"),
        (None, "Local folder"),
    ],
)
def test_local_folder_safe_label(path: str | None, expected: str) -> None:
    assert local_folder_safe_label(path) == expected


def test_local_folder_label_strips_parent_fragments() -> None:
    label = local_folder_safe_label(r"C:\Users\Artur\Private\Client-X\Contracts")
    assert label == "Contracts"
    for fragment in ("C:\\", "Users", "Artur", "Private", "Client-X"):
        assert fragment not in label


def test_safe_source_label_unknown_type() -> None:
    assert safe_source_label(source_type="object_storage") == "Object storage"
    assert safe_source_label(source_type="sharepoint") == "SharePoint"
    assert safe_source_label(source_type="") == "Source"


def test_safe_source_label_local_folder_uses_basename() -> None:
    assert (
        safe_source_label(
            source_type="local_folder",
            path="/home/user/projects/specifications",
        )
        == "specifications"
    )
