# © Artur Czarnecki. All rights reserved.

"""NT-02 (A4-015): tracked contract vs alpha probe semantics.

Alpha probe ``pp_gov.py::test_v2_nt02`` instantiates a bare Tier-2 agent and FAILs when
``production_mode`` is False on a STRICT-reachable manifest binding. Post FC-1 Part C the
product derives posture from host injection (``require_sanctioned_external_work_mutations``),
not ``harness_production_mode()`` on a default instance.
"""

from __future__ import annotations

import importlib

from intergrax.contracts import execution_identity as ids
from intergrax.runtime.execution.agent_runtime_io import RuntimeRequest


def _runtime_request() -> RuntimeRequest:
    io_mod = importlib.import_module("intergrax.runtime.execution.agent_runtime_io")
    return io_mod.RuntimeRequest(
        agent_id="probe",
        user_id="u1",
        session_id="s1",
        message="m",
        task_id=ids.mint_task_id(),
        run_id=ids.mint_run_id(),
        tenant_id="t1",
    )


def test_nt02_bare_external_contractor_agent_stays_relaxed() -> None:
    module = importlib.import_module(
        "external_contractor_adapter.external_contractor_adapter_agent",
    )
    agent = module.ExternalContractorAdapterAgent()
    ctx = agent.build_context(_runtime_request())
    assert ctx.config.production_mode is False


def test_nt02_strict_host_injection_sets_production_mode() -> None:
    module = importlib.import_module(
        "external_contractor_adapter.external_contractor_adapter_agent",
    )
    agent = module.ExternalContractorAdapterAgent(require_sanctioned_external_work_mutations=True)
    ctx = agent.build_context(_runtime_request())
    assert ctx.config.production_mode is True
