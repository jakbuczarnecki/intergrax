#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Canonical known-red FIX row planning inventory for T-AF2-RW-PLAN (Package B)."""

from __future__ import annotations

KNOWN_RED_FIX_INVENTORY: list[dict[str, object]] = [
    {
        "kr_id": "KR-03",
        "owner_node": "T-AF2-RW-GATES",
        "file_scope": ["tests/unit/runtime/nexus/test_reference_harness_strict.py"],
        "negative_test_id": "NT-KR-03-01",
    },
    {
        "kr_id": "KR-04",
        "owner_node": "T-AF2-RW-GATES",
        "file_scope": ["tests/unit/runtime/policy/test_acp_token_budget_enforcement.py"],
        "negative_test_id": "NT-KR-04-01",
    },
    {
        "kr_id": "KR-07",
        "owner_node": "T-AF2-RW-SUPPLY",
        "file_scope": ["scripts/ci/durable_diagnostics_external_proof.py"],
        "negative_test_id": "NT-KR-07-01",
    },
    {
        "kr_id": "KR-08",
        "owner_node": "PT-Q03",
        "file_scope": ["scripts/governance/iiac_activation_gate.py"],
        "negative_test_id": "NT-KR-08-01",
    },
    {
        "kr_id": "KR-09",
        "owner_node": "PT-Q11",
        "file_scope": ["scripts/qualification/render_ebh7_f10_normative_matrix.py"],
        "negative_test_id": "NT-KR-09-01",
    },
    {
        "kr_id": "KR-10",
        "owner_node": "PT-Q04",
        "file_scope": ["tests/qualification/roadmap_replay_x/test_replay_x_closure_counters_computed.py"],
        "negative_test_id": "NT-KR-10-01",
    },
    {
        "kr_id": "KR-11",
        "owner_node": "PT-Q06",
        "file_scope": [
            "tests/qualification/trace_x/test_txp6_q02_restart_resume_closed_world_parity.py",
            "intergrax/platform/enterprise_audit/impact_map.py",
        ],
        "negative_test_id": "NT-KR-11-01",
    },
    {
        "kr_id": "KR-13",
        "owner_node": "T-AF2-RW-GATES",
        "file_scope": ["tests/qualification/platform_domain_completeness/"],
        "negative_test_id": "NT-KR-13-01",
    },
    {
        "kr_id": "KR-14",
        "owner_node": "T-AF2-RW-GATES",
        "file_scope": ["tests/unit/applications/test_scenario_runtime_baseline.py"],
        "negative_test_id": "NT-KR-14-01",
    },
    {
        "kr_id": "KR-15",
        "owner_node": "T-B7-CMP-REVAL",
        "file_scope": ["tests/qualification/compat_x/"],
        "negative_test_id": "NT-KR-15-01",
    },
    {
        "kr_id": "KR-16",
        "owner_node": "PT-WB",
        "file_scope": [
            "tests/unit/applications/test_fixture_dataset_ingestion.py",
            "tests/unit/applications/test_image_smart_document_handler.py",
        ],
        "negative_test_id": "NT-KR-16-01",
    },
    {
        "kr_id": "KR-17",
        "owner_node": "T-AF2-RW-RUNTIME",
        "file_scope": ["intergrax/integrations/providers/relational_store/mongodb/client.py"],
        "negative_test_id": "NT-KR-17-01",
    },
    {
        "kr_id": "KR-18",
        "owner_node": "PT-WA",
        "file_scope": ["intergrax/applications/contracts/environment_profile/root.py"],
        "negative_test_id": "NT-KR-18-01",
    },
    {
        "kr_id": "KR-19",
        "owner_node": "PT-WF",
        "file_scope": ["intergrax/runtime/observability_otlp/export_boundary.py"],
        "negative_test_id": "NT-KR-19-01",
    },
    {
        "kr_id": "KR-20",
        "owner_node": "PT-WF",
        "file_scope": ["tests/qualification/trace_x/", "intergrax/applications/local_workspace_application/"],
        "negative_test_id": "NT-KR-20-01",
    },
    {
        "kr_id": "KR-23",
        "owner_node": "PT-Q07",
        "file_scope": [
            "tests/unit/applications/test_mcp_surface_opt_in.py",
            "tests/unit/applications/test_mcp_surface_opt_in_runtime.py",
            "tests/unit/applications/test_mcp_catalog_e2e_hung_server_qualification.py",
        ],
        "negative_test_id": "NT-KR-23-01",
    },
    {
        "kr_id": "KR-24",
        "owner_node": "T-IGX-039-A",
        "file_scope": ["tests/unit/applications/test_mcp_surface_opt_in.py"],
        "negative_test_id": "NT-KR-24-01",
    },
    {
        "kr_id": "KR-26",
        "owner_node": "PT-WD",
        "file_scope": ["intergrax/platform/enterprise_audit/validators.py"],
        "negative_test_id": "NT-KR-26-01",
    },
    {
        "kr_id": "KR-27",
        "owner_node": "PT-G01",
        "file_scope": ["intergrax/integrations/providers/embedding_provider/ollama/register.py"],
        "negative_test_id": "NT-KR-27-01",
    },
    {
        "kr_id": "KR-28",
        "owner_node": "PT-G01",
        "file_scope": [
            "tests/acceptance/llm_routing/test_llm_routing_acceptance.py",
            "tests/acceptance/llm_routing/test_llm_routing_resolver_acceptance.py",
        ],
        "negative_test_id": "NT-KR-28-01",
    },
]

REQUIRED_KR_FIX_IDS = frozenset(row["kr_id"] for row in KNOWN_RED_FIX_INVENTORY)
