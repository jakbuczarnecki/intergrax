# © Artur Czarnecki. All rights reserved.

"""Default DG-005 qualification provider composition (explicit DI)."""

from __future__ import annotations

from testing_support.obs_distributed_topology.provider_contract import (
    QualificationEvidenceProviderResolver,
)
from testing_support.obs_distributed_topology.providers.sqlite_file import (
    SQLITE_FILE_PROVIDER_ID,
    sqlite_file_evidence_provider_factory,
)

DEFAULT_DG005_EVIDENCE_PROVIDER_RESOLVER = QualificationEvidenceProviderResolver(
    providers={
        SQLITE_FILE_PROVIDER_ID: sqlite_file_evidence_provider_factory,
    },
)
