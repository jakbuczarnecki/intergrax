"""VPI universal data pack contracts."""

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    EmbeddingModelIdentityError,
    VpiDataPackBuildError,
    VpiDataPackCompatibilityError,
    VpiDataPackError,
    VpiDataPackFormatError,
    VpiDataPackIntegrityError,
    VpiDataPackValidationError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.evidence import (
    DataPackProofReport,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    PROOF_50_RECORD_COUNT,
    PROOF_50_SAMPLE_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    SCENARIO_ID,
    semantic_text_hash,
    source_ref_key,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DEFAULT_PROOF_50_ROOT,
    DataPackPaths,
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)

__all__ = [
    "DATA_PACK_VERSION",
    "DEFAULT_PROOF_50_ROOT",
    "EMBEDDING_SCHEMA_VERSION",
    "RELATIONAL_SCHEMA_VERSION",
    "DataPackManifest",
    "DataPackPaths",
    "DataPackProofReport",
    "DataPackStatus",
    "EmbeddingModelIdentityError",
    "EmbeddingDataPackRecord",
    "PROOF_50_RECORD_COUNT",
    "PROOF_50_SAMPLE_VERSION",
    "RelationalDataPackRecord",
    "SCENARIO_ID",
    "VpiDataPackBuildError",
    "VpiDataPackCompatibilityError",
    "VpiDataPackError",
    "VpiDataPackFormatError",
    "VpiDataPackIntegrityError",
    "VpiDataPackValidationError",
    "resolve_data_pack_paths",
    "semantic_text_hash",
    "source_ref_key",
]
