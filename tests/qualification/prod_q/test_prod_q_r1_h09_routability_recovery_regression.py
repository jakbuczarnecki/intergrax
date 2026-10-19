# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1 H09 routability recovery — revision-bound registry routing regression (R01–R05)."""

from __future__ import annotations

from pathlib import Path

import pytest

from external_contractor_adapter.contract import build_agent_contract
from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from governed_contractor_application.host.strict_sanctioned_external_work_registry_projection import (
    reproject_registry_with_strict_sanctioned_external_work,
)
from intergrax.applications._shared.wiring import _index_manifest_bindings, binding_from_roster_entry
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.runtime.registry.agent_routing_policy import evaluate_agent_routing
from tests.qualification.prod_q.test_prod_q_r1_r7_q_strict_external_work_roster_projection_materialization_gate import (
    _host_runtime,
)
from tests.unit.applications.governed_contractor.test_prod_q_r1_r7_d4_strict_production_gca_launcher import (
    _qualified_canonical_bundle,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _EXTERNAL_CONTRACTOR_FACTORY_REF,
    _roster_with_strict_external_work,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_AGENT_ID = "external_contractor_adapter"


def _authority_contract_from_bundle(bundle) -> AgentContract:
    manifest = bundle.manifest
    roster = bundle.effective_roster
    authority = bundle.roster_contract_authority
    assert authority is not None
    manifest_bindings = _index_manifest_bindings(manifest)
    for entry in roster.entries:
        if entry.logical_agent_id != _AGENT_ID:
            continue
        binding = binding_from_roster_entry(entry, manifest_bindings)
        return authority.contract_for_binding(binding)
    raise AssertionError(f"missing roster entry {_AGENT_ID!r}")


def _project_qualified_bundle(bundle):
    host_runtime = _host_runtime()
    return reproject_registry_with_strict_sanctioned_external_work(
        bundle,
        bundle.factory_resolver,
        host_runtime=host_runtime,
        manifest=bundle.manifest,
    )


def test_r01_positive_revision_bound_external_contractor_production_routable(
    tmp_path: Path,
) -> None:
    bundle, *_rest = _qualified_canonical_bundle(tmp_path, revision_id="rev-h09-r01")
    registry = _project_qualified_bundle(bundle).agent_registry
    assert registry.has(_AGENT_ID)
    assert registry.is_routable(_AGENT_ID, production_mode=True)
    decision = evaluate_agent_routing(
        registry.get_contract(_AGENT_ID),
        production_mode=True,
    )
    assert decision.routable is True
    assert decision.reason is None


def test_r02_negative_lifecycle_development_denied_in_production_mode() -> None:
    contract = build_agent_contract().model_copy(
        update={"lifecycle_state": AgentLifecycleState.DEVELOPMENT},
    )
    decision = evaluate_agent_routing(contract, production_mode=True)
    assert decision.routable is False
    assert decision.reason is not None
    assert "production or staging" in decision.reason


def test_r03_negative_production_eligible_without_owner_metadata_denied() -> None:
    contract = AgentContract(
        id="metadata-negative",
        name="Metadata Negative",
        description="qualification negative",
        capabilities=["external_contractor.adapt"],
        lifecycle_state=AgentLifecycleState.PRODUCTION,
        production_eligible=True,
        owner_team="platform",
    )
    decision = evaluate_agent_routing(contract, production_mode=True)
    assert decision.routable is False
    assert decision.reason is not None
    assert "owner metadata" in decision.reason or "runbook" in decision.reason


def test_r04_strict_projection_registry_matches_roster_contract_authority(
    tmp_path: Path,
) -> None:
    bundle, *_ = _qualified_canonical_bundle(tmp_path, revision_id="rev-h09-r04")
    authority_contract = _authority_contract_from_bundle(bundle)
    registered = _project_qualified_bundle(bundle).agent_registry.get_contract(_AGENT_ID)
    assert registered.id == authority_contract.id
    assert registered.lifecycle_state == authority_contract.lifecycle_state
    assert registered.production_eligible == authority_contract.production_eligible


def test_r05_ac3_external_contractor_factory_provenance_unchanged() -> None:
    entry = next(
        e
        for e in _roster_with_strict_external_work().entries
        if e.logical_agent_id == _AGENT_ID
    )
    assert entry.factory_reference == _EXTERNAL_CONTRACTOR_FACTORY_REF
    agent = ExternalContractorAdapterAgent()
    assert agent.get_contract().id == _AGENT_ID
