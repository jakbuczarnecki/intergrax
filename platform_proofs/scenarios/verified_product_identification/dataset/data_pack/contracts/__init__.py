"""VPI universal data pack contracts."""

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
    VpiDataPackCompatibilityError,
    VpiDataPackError,
    VpiDataPackIntegrityError,
    VpiDataPackValidationError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.evidence import (
    DataPackProofReport,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    PROOF_50_RECORD_COUNT,
    PROOF_50_SAMPLE_VERSION,
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
    "DataPackManifest",
    "DataPackPaths",
    "DataPackProofReport",
    "DataPackStatus",
    "EmbeddingDataPackRecord",
    "PROOF_50_RECORD_COUNT",
    "PROOF_50_SAMPLE_VERSION",
    "RelationalDataPackRecord",
    "SCENARIO_ID",
    "VpiDataPackBuildError",
    "VpiDataPackCompatibilityError",
    "VpiDataPackError",
    "VpiDataPackIntegrityError",
    "VpiDataPackValidationError",
    "resolve_data_pack_paths",
    "semantic_text_hash",
    "source_ref_key",
]
