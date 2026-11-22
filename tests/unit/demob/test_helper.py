# © Artur Czarnecki. All rights reserved.

from scripts.demob.helper import normalize


def test_normalize_strips_and_lowercases() -> None:
    assert normalize("  Hello World  ") == "hello world"


def test_normalize_collapses_inner_whitespace() -> None:
    assert normalize("a\t\n  b   c") == "a b c"


def test_normalize_empty_and_whitespace_only() -> None:
    assert normalize("") == ""
    assert normalize("   \t  ") == ""
