# © Artur Czarnecki. All rights reserved.

"""COMPAT-X semantic owner matrix (mechanical current-HEAD evidence)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.qualification.compat_x._compat_x_types import CompatOwnerMatrixRow, OwnerResponsibilityState

_REPO_ROOT = Path(__file__).resolve().parents[3]

COMPAT_X_OWNER_MATRIX: Final[tuple[CompatOwnerMatrixRow, ...]] = (
    CompatOwnerMatrixRow(
        concern="public_contract_identity_version",
        responsibility_state=OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER,
        semantic_owner_path="intergrax/contracts/migrations/registry.py",
        composition_owner_path="intergrax/contracts/migrations/registry.py",
        evidence_paths=(
            "intergrax/contracts/migrations/registry.py",
            "tests/qualification/compat_x/test_compat_x_inventory_gates.py",
        ),
    ),
    CompatOwnerMatrixRow(
        concern="runtime_schema_registry",
        responsibility_state=OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER,
        semantic_owner_path="intergrax/runtime/schema/registry.py",
        composition_owner_path="intergrax/runtime/schema/registry.py",
        evidence_paths=("intergrax/runtime/schema/registry.py",),
    ),
    CompatOwnerMatrixRow(
        concern="persisted_schema_migration",
        responsibility_state=OwnerResponsibilityState.FRAGMENTED_UNOWNED,
        semantic_owner_path="FRAGMENTED — contracts registry vs runtime local migrations",
        composition_owner_path="intergrax/runtime/schema/registry.py",
        evidence_paths=(
            "intergrax/contracts/migrations/registry.py",
            "intergrax/runtime/observability/causal_evidence_index.py",
            "intergrax/runtime/diagnostics/problem_occurrence_migration.py",
        ),
    ),
    CompatOwnerMatrixRow(
        concern="versioning_policy",
        responsibility_state=OwnerResponsibilityState.FRAGMENTED_UNOWNED,
        semantic_owner_path="FRAGMENTED — no global versioning policy owner",
        composition_owner_path="UNOWNED — COMPAT-X-R1",
        evidence_paths=("tests/qualification/compat_x/_compat_x_inventory.py",),
    ),
    CompatOwnerMatrixRow(
        concern="event_schema_evolution",
        responsibility_state=OwnerResponsibilityState.CANDIDATE_OWNER_REMEDIATION_REQUIRED,
        semantic_owner_path="intergrax/runtime/events/payload_registry.py",
        composition_owner_path="intergrax/runtime/events/event_kind_registry.py",
        evidence_paths=(
            "intergrax/runtime/events/payload_registry.py",
            "intergrax/runtime/events/spine_payload_codec.py",
        ),
    ),
    CompatOwnerMatrixRow(
        concern="event_evolution_policy",
        responsibility_state=OwnerResponsibilityState.FRAGMENTED_UNOWNED,
        semantic_owner_path="FRAGMENTED — payload registry without evolution policy",
        composition_owner_path="UNOWNED — COMPAT-X-R3",
        evidence_paths=("intergrax/runtime/events/payload_registry.py",),
    ),
    CompatOwnerMatrixRow(
        concern="platform_plugin_manifest_compatibility",
        responsibility_state=OwnerResponsibilityState.CANDIDATE_OWNER_REMEDIATION_REQUIRED,
        semantic_owner_path="intergrax/core/plugins/package_contract.py",
        composition_owner_path="intergrax/core/plugins/package_contract.py",
        evidence_paths=("intergrax/core/plugins/package_contract.py",),
    ),
    CompatOwnerMatrixRow(
        concern="provider_contract_compatibility",
        responsibility_state=OwnerResponsibilityState.CANDIDATE_OWNER_REMEDIATION_REQUIRED,
        semantic_owner_path="intergrax/integrations/registry/catalog.py",
        composition_owner_path="intergrax/integrations/registry/factory.py",
        evidence_paths=("intergrax/integrations/registry/catalog.py",),
    ),
    CompatOwnerMatrixRow(
        concern="plugin_provider_compatibility_policy",
        responsibility_state=OwnerResponsibilityState.FRAGMENTED_UNOWNED,
        semantic_owner_path="FRAGMENTED — manifest vs catalog vs external assessment",
        composition_owner_path="UNOWNED — COMPAT-X-R4",
        evidence_paths=(
            "intergrax/core/plugins/package_contract.py",
            "intergrax/integrations/registry/catalog.py",
        ),
    ),
    CompatOwnerMatrixRow(
        concern="external_contract_semantic_assessment",
        responsibility_state=OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER,
        semantic_owner_path="intergrax/integrations/contracts/external_contract_compatibility.py",
        composition_owner_path="intergrax/integrations/contracts/external_contract_compatibility.py",
        evidence_paths=(
            "intergrax/integrations/contracts/external_contract_compatibility.py",
            "tests/qualification/external_contract_compatibility/test_external_contract_compatibility_certification.py",
        ),
    ),
    CompatOwnerMatrixRow(
        concern="deprecation_removal",
        responsibility_state=OwnerResponsibilityState.FRAGMENTED_UNOWNED,
        semantic_owner_path="UNOWNED — COMPAT-X-R5",
        composition_owner_path="UNOWNED — COMPAT-X-R5",
        evidence_paths=("tests/qualification/compat_x/_compat_x_inventory.py",),
    ),
    CompatOwnerMatrixRow(
        concern="compatibility_adapters",
        responsibility_state=OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER,
        semantic_owner_path="intergrax/compat/langchain/documents.py",
        composition_owner_path="intergrax/compat/langchain/documents.py",
        evidence_paths=("intergrax/compat/langchain/documents.py",),
    ),
)


def owner_matrix_paths_exist() -> tuple[str, ...]:
    missing: list[str] = []
    for row in COMPAT_X_OWNER_MATRIX:
        for rel in row.evidence_paths:
            if rel.startswith("tests/") or rel.startswith("UNOWNED") or rel.startswith("FRAGMENTED"):
                continue
            if not (_REPO_ROOT / rel).is_file():
                missing.append(rel)
        for rel in (row.semantic_owner_path, row.composition_owner_path):
            if rel.startswith(("UNOWNED", "FRAGMENTED")):
                continue
            if not (_REPO_ROOT / rel).is_file():
                missing.append(rel)
    return tuple(missing)
