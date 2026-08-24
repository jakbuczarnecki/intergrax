# © Artur Czarnecki. All rights reserved.

"""PROMO-P0-2: bounded public documentation local link integrity gate."""

from __future__ import annotations

import pytest

from tests.unit.docs.public_link_integrity import (
    PUBLIC_ROOTS,
    collect_public_link_integrity_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_public_documentation_local_link_integrity() -> None:
    report = collect_public_link_integrity_report()

    assert report.roots_checked == len(PUBLIC_ROOTS)
    assert not report.missing_roots, f"missing public roots: {report.missing_roots}"
    assert report.documents_checked >= len(PUBLIC_ROOTS)
    assert not report.broken_links, _format_broken_links(report)


def _format_broken_links(report) -> str:
    if not report.broken_links:
        return "no broken local links"
    lines = [
        (
            f"{item.source}: {item.target} "
            f"(docs={report.documents_checked}, links={report.local_links_checked}, assets={report.assets_checked})"
        )
        for item in report.broken_links
    ]
    return "broken public local links:\n" + "\n".join(lines)
