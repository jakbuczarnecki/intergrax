# © Artur Czarnecki. All rights reserved.

"""COMPAT-X semantic owner matrix (mechanical current-HEAD evidence)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.qualification.compat_x._compat_x_types import CompatOwnerMatrixRow

_REPO_ROOT = Path(__file__).resolve().parents[3]

COMPAT_X_OWNER_MATRIX: Final[tuple[CompatOwnerMatrixRow, ...]] = (
    CompatOwnerMatrixRow(
        concern="public_contract_identity_version",
        semantic_owner_path="intergrax/contracts/migrations/registry.py",
        composition_owner_path="intergrax/contracts/migrations/registry.py",
        evidence_paths=(
            "intergrax/contracts/migrations/registry.py",
            "tests/qualification/compat_x/test_compat_x_inventory_gates.py",
        ),
    ),
    CompatOwnerMatrixRow(
        concern="runtime_schema_registry",
        semantic_owner_path="intergrax/runtime/schema/registry.py",
        composition_owner_path="intergrax/runtime/schema/registry.py",
        evidence_paths=("intergrax/runtime/schema/registry.py",),
    ),
    CompatOwnerMatrixRow(
        concern="persisted_schema_migration",
        semantic_owner_path="intergrax/contracts/migrations/registry.py",
        composition_owner_path="intergrax/runtime/schema/registry.py",
        evidence_paths=(
            "intergrax/contracts/migrations/registry.py",
            "intergrax/runtime/observability/causal_evidence_index.py",
        ),
    ),
    CompatOwnerMatrixRow(
        concern="event_schema_evolution",
        semantic_owner_path="intergrax/runtime/events/payload_registry.py",
        composition_owner_path="intergrax/runtime/events/event_kind_registry.py",
        evidence_paths=(
            "intergrax/runtime/events/payload_registry.py",
            "intergrax/runtime/events/spine_payload_codec.py",
        ),
    ),
    CompatOwnerMatrixRow(
        concern="platform_plugin_manifest_compatibility",
        semantic_owner_path="intergrax/core/plugins/package_contract.py",
        composition_owner_path="intergrax/core/plugins/package_contract.py",
        evidence_paths=("intergrax/core/plugins/package_contract.py",),
    ),
    CompatOwnerMatrixRow(
        concern="provider_contract_compatibility",
        semantic_owner_path="intergrax/integrations/registry/catalog.py",
        composition_owner_path="intergrax/integrations/registry/factory.py",
        evidence_paths=("intergrax/integrations/registry/catalog.py",),
    ),
    CompatOwnerMatrixRow(
        concern="external_contract_semantic_assessment",
        semantic_owner_path="intergrax/integrations/contracts/external_contract_compatibility.py",
        composition_owner_path="intergrax/integrations/contracts/external_contract_compatibility.py",
        evidence_paths=(
            "intergrax/integrations/contracts/external_contract_compatibility.py",
            "tests/qualification/external_contract_compatibility/test_external_contract_compatibility_certification.py",
        ),
    ),
    CompatOwnerMatrixRow(
        concern="deprecation_removal",
        semantic_owner_path="UNOWNED — COMPAT-X-R5",
        composition_owner_path="UNOWNED — COMPAT-X-R5",
        evidence_paths=("tests/qualification/compat_x/_compat_x_inventory.py",),
    ),
    CompatOwnerMatrixRow(
        concern="compatibility_adapters",
        semantic_owner_path="intergrax/compat/langchain/documents.py",
        composition_owner_path="intergrax/compat/langchain/documents.py",
        evidence_paths=("intergrax/compat/langchain/documents.py",),
    ),
)


def owner_matrix_paths_exist() -> tuple[str, ...]:
    missing: list[str] = []
    for row in COMPAT_X_OWNER_MATRIX:
        for rel in row.evidence_paths:
            if rel.startswith("tests/") or rel.startswith("UNOWNED"):
                continue
            if not (_REPO_ROOT / rel).is_file():
                missing.append(rel)
        for rel in (row.semantic_owner_path, row.composition_owner_path):
            if rel.startswith("UNOWNED"):
                continue
            if not (_REPO_ROOT / rel).is_file():
                missing.append(rel)
    return tuple(missing)
