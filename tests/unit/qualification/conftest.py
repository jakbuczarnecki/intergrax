# © Artur Czarnecki. All rights reserved.

"""Qualification test collection helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T[0-9:.+\-Z]+")


def pytest_make_parametrize_id(config: Any, val: object, argname: str) -> str | None:
    """Give parameters that embed wall-clock timestamps a run-independent id.

    ``pytest-xdist`` requires every worker to collect identical test ids; fixtures such as the operator-record texts of
    ``test_evol_x_closure_preflight_b4.py`` embed ``datetime.now()`` and would otherwise produce different ids per
    worker process (the parallel CI job ``evol-x-b4-preflight`` then fails at collection).
    """
    if isinstance(val, str) and _ISO_TIMESTAMP.search(val):
        stable = _ISO_TIMESTAMP.sub("<ts>", val)
        return f"{argname}-{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:10]}"
    return None
