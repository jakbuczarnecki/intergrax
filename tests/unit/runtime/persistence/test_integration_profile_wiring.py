# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.persistence.integration_profile_wiring import (
    open_trace_store_from_profile,
    sqlite_runtime_persistence_for_profile,
)

pytestmark = pytest.mark.unit


def test_sqlite_runtime_persistence_for_profile(tmp_path: Path) -> None:
    profile = IntegrationProfile(relational_store="sqlite")
    profile = profile.model_copy(
        update={"options": {"sqlite": {"data_dir": str(tmp_path)}}},
    )
    bundle = sqlite_runtime_persistence_for_profile(profile)
    assert bundle is not None
    assert bundle.paths.data_dir == tmp_path


def test_open_trace_store_from_profile_sqlite(tmp_path: Path) -> None:
    profile = IntegrationProfile(relational_store="sqlite")
    profile = profile.model_copy(
        update={"options": {"sqlite": {"data_dir": str(tmp_path)}}},
    )
    store = open_trace_store_from_profile(profile)
    assert isinstance(store, SQLiteRunTraceStore)
