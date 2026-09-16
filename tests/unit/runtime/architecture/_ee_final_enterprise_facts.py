# © Artur Czarnecki. All rights reserved.

"""EE-FINAL — cross-session enterprise certification anchors and paths."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

FINAL_ARCHITECTURE_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_FINAL_ENTERPRISE_ARCHITECTURE.md"
)

FINAL_QUALIFICATION_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_FINAL_CROSS_SESSION_ENTERPRISE_EXECUTION_ENGINE_CERTIFICATION.md"
)

EE_FINAL_ARCH_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_FINAL_ARCH_UNIFIED_ENTRY_PLUGINABILITY_ZERO_BYPASS_CERTIFICATION.md"
)

P0_INVENTORY = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md"
)

EE_B2_FINAL_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B2_FINAL_CHAOS_FAULT_MATRIX_CLOSURE.md"
)

PLATFORM_REVALIDATION_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "INTEGRAX_CURRENT_HEAD_PLATFORM_REVALIDATION.md"
)

# Git-verified anchors on development (2026-09-14 session).
REVALIDATION_COMMIT = "170460148744ac2324e67b7e992e71883e443561"
EE_FINAL_ARCH_COMMIT = "1c1005e2f66447e3f19f9aba8c0020b13c944b72"
EE_B2_FINAL_COMMIT = "6d6f6544f9b0fc5d72f93fdd9a06c85de3e38359"
NPSC5F_REQUALIFICATION_COMMIT = "cd0217ef0cbf2386f5f6134c30cfb80adf6ecddb"

FINAL_ANCHOR_COMMITS: tuple[tuple[str, str], ...] = (
    ("EE-A1", "4b3102293982ceff352b2162108d66b6c4e2bbdc"),
    ("EE-A2", "d9c3653daf8a087e0380803f1f297d4cd4d0a046"),
    ("NPSC-4.2", "c185a82250698caf9c1475fdffe26d92390972f4"),
    ("NPSC-5A", "d61fe0b25194b8a229842c79fa884ecf721d711a"),
    ("NPSC-5B", "ce84900002c0d8b33f158e4c46ced3dc2353d499"),
    ("NPSC-5C", "d2cf64ce4e0e50dbe09ecddf57c292cefb52eed9"),
    ("NPSC-5D", "a4a1faca01cd5004e372f235132184a84aa5a6bd"),
    ("NPSC-5E", "fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7"),
    ("NPSC-5F R3+Final requalification", NPSC5F_REQUALIFICATION_COMMIT),
    ("EE-B1.1", "17db424fcf2369b1de0de368489a2d96e24564e5"),
    ("EE-B1.2", "b664029fb30e59ca0587569be3290a9d506301ec"),
    ("EE-B1.3", "471e183d5cd16639a63ced068f2b221be29903c1"),
    ("EE-B2-FINAL", EE_B2_FINAL_COMMIT),
    ("EE-B3-A", "fe2edc8077234437b13345daaf46633867fe8f31"),
    ("EE-B3-C", "e56579c8e2df06d12b0745ce47b5faf58e8efe0e"),
    ("EE-B4-A", "f7c17550d1ccb697ee6f46417d899a2f34da79c7"),
    ("EE-B4-B", "6baa9b4fa92addfad4e163330b3a7e75100b3629"),
    ("EE-B4-C", "edd44e2183c8ec78f6ca71697865d630ba221a9e"),
    ("EE-FINAL-ARCH", EE_FINAL_ARCH_COMMIT),
    ("Current HEAD Platform Revalidation", REVALIDATION_COMMIT),
)

FINAL_GATE_MODULES: tuple[str, ...] = (
    "tests/unit/runtime/architecture/test_ee_final_enterprise_execution_engine_certification.py",
    "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py",
    "tests/unit/runtime/architecture/test_ee_a2_h1_intake_identity_convergence_certification.py",
    "tests/unit/runtime/architecture/test_ee_a2_h2_identity_authority_global_freeze.py",
    "tests/unit/runtime/architecture/test_ee_a2_h3_identity_authority_frozen_plane_integration.py",
    "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py",
    "tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py",
    "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py",
    "tests/unit/runtime/architecture/test_ee_b1_2_capacity_architecture_gate.py",
    "tests/unit/runtime/architecture/test_ee_b1_3_worker_architecture_gate.py",
    "tests/unit/runtime/architecture/test_ee_b2_final_architecture_gate.py",
    "tests/unit/runtime/architecture/test_ee_b3_a_security_architecture_gate.py",
    "tests/unit/runtime/architecture/test_ee_b3_c_security_architecture_gate.py",
    "tests/unit/runtime/architecture/test_ee_b4_a_operational_architecture_gate.py",
    "tests/unit/runtime/architecture/test_ee_b4_b_shutdown_architecture_gate.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_operations_architecture_gate.py",
)

EE_FINAL_ARCH_GATE_GLOB = "tests/unit/runtime/architecture/test_ee_final_arch_*.py"
