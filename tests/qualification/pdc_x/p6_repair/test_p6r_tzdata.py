# © Artur Czarnecki. All rights reserved.

"""B-P6-R item 1 - IANA zone data is a declared dependency on Windows (no system zone database)."""

from __future__ import annotations

import tomllib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
_MARKER = "sys_platform == 'win32'"


def test_iana_zone_lookup_works_in_the_default_environment() -> None:
    zone = ZoneInfo("Europe/Warsaw")
    summer = datetime(2026, 7, 1, 12, 0, tzinfo=zone).utcoffset()
    winter = datetime(2026, 1, 1, 12, 0, tzinfo=zone).utcoffset()
    assert summer is not None and winter is not None
    assert summer != winter


def test_pyproject_declares_tzdata_for_windows_as_a_direct_dependency() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    normalized = [str(item).replace('"', "'").replace(" ", "") for item in data["project"]["dependencies"]]
    assert "tzdata;sys_platform=='win32'" in normalized


def test_lock_records_tzdata_as_a_windows_dependency_of_the_project() -> None:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    project = next(p for p in lock["package"] if p["name"] == "intergrax-ai" and p.get("source") == {"editable": "."})
    declared = [d for d in project["dependencies"] if d["name"] == "tzdata"]
    assert declared and declared[0].get("marker") == _MARKER
    requires = [r for r in project["metadata"]["requires-dist"] if r["name"] == "tzdata"]
    assert requires and requires[0].get("marker") == _MARKER
