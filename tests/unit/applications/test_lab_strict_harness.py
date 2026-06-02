# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.lab_harness_context import LabHarnessContext
from intergrax.applications._shared.lab_runtime_config import build_lab_agent_runtime_config
from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_strict_harness_runtime_config_uses_production_mode_and_trace_path(
    tmp_path: Path,
) -> None:
    trace_db = tmp_path / "trace.db"
    harness = LabHarnessContext(
        policy_bundle=build_runtime_policy_bundle(),
        strict_harness=True,
        trace_db_path=trace_db,
    )
    config = build_lab_agent_runtime_config(
        request=RuntimeRequest(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            agent_id="echo",
            message="hi",
        ),
        llm_adapter=FakeLLMAdapter(),
        harness=harness,
    )
    assert config.production_mode is True
    assert config.trace_db_path == str(trace_db)


def test_application_build_context_strict_flag() -> None:
    manifest = ApplicationManifest.lab(
        app_id="lab",
        name="Lab",
        agents=[],
    )
    ctx = ApplicationBuildContext.for_manifest(
        manifest,
        strict_harness=True,
        trace_db_path=Path("/tmp/trace.db"),
    )
    assert ctx.strict_harness is True
    assert ctx.trace_db_path == Path("/tmp/trace.db")
