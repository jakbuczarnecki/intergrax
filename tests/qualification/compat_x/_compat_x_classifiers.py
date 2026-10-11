# © Artur Czarnecki. All rights reserved.

"""Evidence-backed classifiers for migration and compatibility shim candidates."""

from __future__ import annotations

import ast
from typing import Final

from tests.qualification.compat_x._compat_x_ast_signals import (
    build_compatibility_candidate_context,
    extract_migration_signals,
    is_compatibility_adapter_candidate,
    module_exhibits_parallel_authority,
    module_is_translation_only_compat_adapter,
    parse_module,
)
from tests.qualification.compat_x._compat_x_types import MigrationMechanismClass, ShimClass

_CANONICAL_MIGRATION_PATHS: Final[frozenset[str]] = frozenset(
    {
        "intergrax/contracts/migrations/registry.py",
        "intergrax/runtime/schema/registry.py",
    }
)

_SANCTIONED_MIGRATION_CLASSES: Final[frozenset[MigrationMechanismClass]] = frozenset(
    {
        MigrationMechanismClass.CANONICAL_MIGRATION_OWNER,
        MigrationMechanismClass.LOCAL_FORMAT_MIGRATION,
        MigrationMechanismClass.LEGACY_COMPAT_READER,
        MigrationMechanismClass.UNOWNED_MIGRATION,
        MigrationMechanismClass.DUPLICATE_MIGRATION_AUTHORITY,
        MigrationMechanismClass.NOT_APPLICABLE,
    }
)


def sanctioned_migration_classes() -> frozenset[MigrationMechanismClass]:
    return _SANCTIONED_MIGRATION_CLASSES


def classify_migration_module(module_path: str, source: str) -> MigrationMechanismClass:
    normalized = module_path.replace("\\", "/")
    if normalized in _CANONICAL_MIGRATION_PATHS:
        return MigrationMechanismClass.CANONICAL_MIGRATION_OWNER
    if "quantum_state_rewriter" in source or "class QuantumMigrator" in source:
        return MigrationMechanismClass.UNSANCTIONED_MIGRATION_MECHANISM
    tree = parse_module(module_path, source)
    signals = extract_migration_signals(module_path, tree, source)
    if not signals:
        return MigrationMechanismClass.NOT_APPLICABLE
    if normalized.startswith("intergrax/compat/"):
        return MigrationMechanismClass.LEGACY_COMPAT_READER
    if normalized.startswith("intergrax/contracts/migrations/"):
        return MigrationMechanismClass.CANONICAL_MIGRATION_OWNER
    if normalized.startswith("intergrax/runtime/schema/"):
        return MigrationMechanismClass.CANONICAL_MIGRATION_OWNER
    if "registry.py" in normalized and "migrations" in normalized:
        return MigrationMechanismClass.CANONICAL_MIGRATION_OWNER
    return MigrationMechanismClass.LOCAL_FORMAT_MIGRATION


def classify_shim_module(module_path: str, source: str) -> ShimClass:
    normalized = module_path.replace("\\", "/")
    tree = parse_module(module_path, source)
    context = build_compatibility_candidate_context(module_path, source, tree)
    if not is_compatibility_adapter_candidate(context):
        return ShimClass.NOT_APPLICABLE
    if module_exhibits_parallel_authority(tree, context):
        return ShimClass.PARALLEL_AUTHORITY
    if module_is_translation_only_compat_adapter(tree, context):
        return ShimClass.TRANSLATION_ONLY
    lowered = source.lower()
    if "from_langchain" in lowered or "to_langchain" in lowered:
        return ShimClass.READ_COMPATIBILITY_ONLY
    if "translate" in lowered or "adapter" in lowered:
        return ShimClass.TRANSLATION_ONLY
    if normalized.startswith("intergrax/compat/"):
        return ShimClass.READ_COMPATIBILITY_ONLY
    return ShimClass.NOT_APPLICABLE


def langchain_documents_shim_evidence() -> tuple[str, ...]:
    """Code-evidence: translation bridge only (no provider/governance/persistence truth)."""
    return (
        "intergrax/compat/langchain/documents.py:from_langchain_document converts LangChain Document -> KnowledgeDocument",
        "intergrax/compat/langchain/documents.py:to_langchain_document converts KnowledgeDocument -> LangChain Document",
        "intergrax/compat/langchain/documents.py: does not define provider resolution; uses KnowledgeDocument.model_validate for semantics",
    )
