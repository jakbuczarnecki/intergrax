from __future__ import annotations

import pytest
from scripts.demo.helper import normalize

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Hello   World  ", "hello world"),
        ("FOO\tbar\nbaz", "foo bar baz"),
        ("ALREADY lower", "already lower"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_strips_collapses_and_lowercases(raw: str, expected: str) -> None:
    assert normalize(raw) == expected
