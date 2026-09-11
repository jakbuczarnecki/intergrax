# © Artur Czarnecki. All rights reserved.

"""Atomic artifact writes for qualification sessions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
