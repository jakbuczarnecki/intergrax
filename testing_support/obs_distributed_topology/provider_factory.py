# © Artur Czarnecki. All rights reserved.

"""Backward-compatible re-exports; SQLite lives under ``providers/``."""

from __future__ import annotations

from testing_support.obs_distributed_topology.provider_composition import (
    DEFAULT_DG005_EVIDENCE_PROVIDER_RESOLVER,
)
from testing_support.obs_distributed_topology.provider_contract import (
    EvidenceProviderDescriptor,
    EvidenceProviderFactory,
    QualificationEvidenceProviderResolver,
)
from testing_support.obs_distributed_topology.providers.sqlite_file import (
    SQLITE_FILE_PROVIDER_ID,
    sqlite_file_descriptor,
    sqlite_file_evidence_provider_factory,
)

DEFAULT_DG005_EVIDENCE_PROVIDER_FACTORY = sqlite_file_evidence_provider_factory

__all__ = [
    "DEFAULT_DG005_EVIDENCE_PROVIDER_FACTORY",
    "DEFAULT_DG005_EVIDENCE_PROVIDER_RESOLVER",
    "EvidenceProviderDescriptor",
    "EvidenceProviderFactory",
    "QualificationEvidenceProviderResolver",
    "SQLITE_FILE_PROVIDER_ID",
    "sqlite_file_descriptor",
    "sqlite_file_evidence_provider_factory",
]
