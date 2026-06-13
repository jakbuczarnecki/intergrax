# © Artur Czarnecki. All rights reserved.

"""Scaffold factory must use H-APP host runtime (Phase DX-1.6)."""

from __future__ import annotations

import pytest

from intergrax.scaffold.new_application import create_application

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_scaffold_lab_factory_uses_harness_host_runtime(tmp_path) -> None:
    target = create_application(
        name="align_lab",
        agents=["echo"],
        profile="lab",
        root=tmp_path,
        port=8199,
        force=True,
    )
    factory = (target / "host" / "factory.py").read_text(encoding="utf-8")
    assert "build_harness_host_runtime" in factory
    assert "nexus_loop = NexusLoop(" not in factory
    assert "host.integration_wiring import" not in factory
