# © Artur Czarnecki. All rights reserved.

"""Architecture gate: qualification core/domain binding must not embed vendor mechanics."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_PATTERNS = (
    "resolve_postgresql_config",
    "open_postgresql_collaborative_work_repositories",
    "psycopg",
    'provider_id == "postgresql"',
    'provider_id == "sqlite"',
    'resolved_provider_id == "postgresql"',
    'resolved_provider_id == "sqlite"',
)

_SCOPED_FILES = (
    "intergrax/core/qualification/execution.py",
    "intergrax/core/qualification/suite.py",
    "intergrax/collaborative_work/repository_qualification_suite.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("relative_path", _SCOPED_FILES)
def test_scoped_qualification_modules_have_no_vendor_execution_branches(
    relative_path: str,
) -> None:
    source = (_repo_root() / relative_path).read_text(encoding="utf-8")
    violations = [pattern for pattern in _FORBIDDEN_PATTERNS if pattern in source]
    assert not violations, f"{relative_path} contains forbidden vendor mechanics: {violations}"
