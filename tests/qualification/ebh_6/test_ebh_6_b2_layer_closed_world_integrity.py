# © Artur Czarnecki. All rights reserved.

"""EBH-6 Batch B2 — T-LAY-* closed-world inventory + L3 linkage (16 layers)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_6._closed_world_discovery import (
    EBH6_AUDIT_LAYERS,
    T_LAY_PACKAGE_LAYERS,
    assert_ebh6_b2_layer_l3_inventory,
    assert_ebh6_b2_t_lay_package_surfaces,
    discover_agent_distribution_paths,
    discover_application_wiring_paths,
    discover_composition_root_paths,
    discover_contracts_surface_paths,
    discover_governance_execution_paths,
    discover_integration_bundle_paths,
    discover_rag_graph_bootstrap_paths,
    discover_resilience_observability_paths,
    discover_runtime_hosting_paths,
    discover_sanctioned_production_composition_roots,
    discover_semantic_core_paths,
    discover_tools_skills_runtime_paths,
)
from tests.qualification.ebh_6._l3_reachability_evidence import (
    l3_records_by_layer,
    mechanism_ids_for_layer,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh6_b2_exit_t_lay_inventory_and_l3() -> None:
    """B2 consolidated exit — all 8 T-LAY packages + 16-layer L3 annex."""
    assert_ebh6_b2_t_lay_package_surfaces()
    assert_ebh6_b2_layer_l3_inventory()


@pytest.mark.parametrize("package_id", tuple(T_LAY_PACKAGE_LAYERS.keys()))
def test_ebh6_b2_t_lay_package_covers_layers(package_id: str) -> None:
    layers = T_LAY_PACKAGE_LAYERS[package_id]
    assert len(layers) == 2, package_id
    for layer in layers:
        assert layer in EBH6_AUDIT_LAYERS, package_id
        records = l3_records_by_layer(layer)
        assert len(records) >= 3, f"{package_id}/{layer}"
        assert mechanism_ids_for_layer(layer), f"{package_id}/{layer}"


def test_ebh6_b2_t_lay_ctr_smc_surfaces() -> None:
    assert len(discover_contracts_surface_paths()) >= 10
    assert len(discover_semantic_core_paths()) >= 5


def test_ebh6_b2_t_lay_app_cmp_surfaces() -> None:
    assert discover_sanctioned_production_composition_roots()
    assert discover_application_wiring_paths()
    assert discover_composition_root_paths()


def test_ebh6_b2_t_lay_gov_exe_surfaces() -> None:
    assert len(discover_governance_execution_paths()) >= 10


def test_ebh6_b2_t_lay_int_sta_surfaces() -> None:
    assert len(discover_integration_bundle_paths()) >= 5


def test_ebh6_b2_t_lay_rec_obs_surfaces() -> None:
    assert len(discover_resilience_observability_paths()) >= 5


def test_ebh6_b2_t_lay_trc_wrk_reuses_spine_and_workers() -> None:
    for layer in ("events_traceability", "background_workers_infra_adapters"):
        assert len(l3_records_by_layer(layer)) >= 3, layer


def test_ebh6_b2_t_lay_agd_rag_surfaces() -> None:
    assert discover_rag_graph_bootstrap_paths()
    assert len(discover_agent_distribution_paths()) >= 3


def test_ebh6_b2_t_lay_hst_tlc_surfaces() -> None:
    assert len(discover_runtime_hosting_paths()) >= 3
    assert len(discover_tools_skills_runtime_paths()) >= 3
