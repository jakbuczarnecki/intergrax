# © Artur Czarnecki. All rights reserved.

"""NT-02 (A4-015): tracked contract vs alpha probe semantics."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

from testing_support.builder import build_runtime_request_for_tests


def test_nt02_bare_external_contractor_agent_default_is_relaxed() -> None:
    module = importlib.import_module(
        "external_contractor_adapter.external_contractor_adapter_agent",
    )
    agent = module.ExternalContractorAdapterAgent()
    assert agent._require_sanctioned_external_work_mutations is False


def test_nt02_strict_host_flag_prepares_production_mode() -> None:
    module = importlib.import_module(
        "external_contractor_adapter.external_contractor_adapter_agent",
    )
    agent = module.ExternalContractorAdapterAgent(require_sanctioned_external_work_mutations=True)
    request = build_runtime_request_for_tests(seed="nt02-strict-posture")
    with patch.object(module.RuntimeContext, "build", return_value=MagicMock()) as build_mock:
        agent.build_context(request)
    config = build_mock.call_args.kwargs["config"]
    assert config.production_mode is True
