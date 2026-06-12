# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from intergrax.scaffold.harness_adr import (
    deepen_docs_relative_links,
    harness_adr_entry_path,
    harness_adr_entry_relpath,
    relative_harness_adr_link,
    rewrite_harness_adr_cross_links,
)

pytestmark = pytest.mark.unit


def test_harness_adr_entry_relpath_uses_iso_day() -> None:
    rel = harness_adr_entry_relpath("ADR-FLOW-001.md", day=date(2026, 6, 7))
    assert rel == "entries/2026-06-07/ADR-FLOW-001.md"


def test_harness_adr_entry_path_validates_basename() -> None:
    with pytest.raises(ValueError, match="invalid harness ADR filename"):
        harness_adr_entry_path(Path("/tmp/docs/adr"), "not-an-adr.md")


def test_relative_harness_adr_link_same_day() -> None:
    index = {"ADR-FLOW-001.md": "2026-06-07", "ADR-FLOW-002.md": "2026-06-07"}
    assert (
        relative_harness_adr_link(
            from_day="2026-06-07",
            target_basename="ADR-FLOW-002.md",
            index=index,
        )
        == "ADR-FLOW-002.md"
    )


def test_relative_harness_adr_link_cross_day() -> None:
    index = {"ADR-CTX-001.md": "2026-06-12", "ADR-MEM-001.md": "2026-06-08"}
    assert (
        relative_harness_adr_link(
            from_day="2026-06-12",
            target_basename="ADR-MEM-001.md",
            index=index,
        )
        == "../2026-06-08/ADR-MEM-001.md"
    )


def test_rewrite_harness_adr_cross_links() -> None:
    index = {"ADR-CTX-001.md": "2026-06-12", "ADR-MEM-001.md": "2026-06-08"}
    text = "See [ADR-MEM-001](ADR-MEM-001.md) for budget semantics."
    out = rewrite_harness_adr_cross_links(text, from_day="2026-06-12", index=index)
    assert "](../2026-06-08/ADR-MEM-001.md)" in out


def test_deepen_docs_relative_links() -> None:
    text = "Canon: [MEMORY](../architecture/MEMORY.md)"
    assert "](../../architecture/MEMORY.md)" in deepen_docs_relative_links(text)
