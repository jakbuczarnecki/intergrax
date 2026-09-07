# © Artur Czarnecki. All rights reserved.

"""Early pytest plugin hooks for invocation-scoped cache configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.pytest_temp_root import apply_invocation_pytest_cache_dir

_REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_load_initial_conftests(
    early_config: pytest.Config,
    parser: pytest.Parser,
    args: list[str],
) -> None:
    """Assign invocation-owned pytest cache before cacheprovider initializes."""
    (_REPO_ROOT / "build").mkdir(parents=True, exist_ok=True)
    apply_invocation_pytest_cache_dir(early_config, _REPO_ROOT)
