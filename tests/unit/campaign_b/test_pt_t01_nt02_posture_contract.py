# © Artur Czarnecki. All rights reserved.

"""NT-02 (A4-015): tracked contract vs alpha probe semantics."""

from __future__ import annotations

import importlib


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
    assert agent._require_sanctioned_external_work_mutations is True
