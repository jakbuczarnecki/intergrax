# © Artur Czarnecki. All rights reserved.

"""Composition-bound secret resolution for authenticated diagnostic Problem list cursors."""

from __future__ import annotations

import os

_PROBLEM_LIST_CURSOR_SECRET_ENV = "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"


def resolve_problem_list_cursor_secret() -> bytes:
    """
    Resolve the production HMAC secret for diagnostic Problem list cursors.

    Restarting a host with a new secret invalidates previously issued cursors.
    """
    raw = os.environ.get(_PROBLEM_LIST_CURSOR_SECRET_ENV, "").strip()
    if not raw:
        raise ValueError(
            f"{_PROBLEM_LIST_CURSOR_SECRET_ENV} is required for authenticated "
            "diagnostic Problem list cursors",
        )
    encoded = raw.encode("utf-8")
    if not encoded:
        raise ValueError("problem_list_cursor_secret_invalid")
    return encoded


__all__ = [
    "_PROBLEM_LIST_CURSOR_SECRET_ENV",
    "resolve_problem_list_cursor_secret",
]
