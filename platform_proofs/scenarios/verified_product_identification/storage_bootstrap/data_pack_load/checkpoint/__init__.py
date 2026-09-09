"""Durable checkpoint coordination for storage bootstrap resume."""

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.compatibility import (
    advance_checkpoint_after_batch,
    build_run_identity,
    compute_resume_decision,
    initial_checkpoint_state,
    utc_now_iso,
    validate_checkpoint_compatibility,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
    VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
    VPI_STORAGE_BOOTSTRAP_ORDERING_POLICY,
    BootstrapCheckpointCompatibility,
    BootstrapCheckpointIdentity,
    BootstrapCheckpointState,
    BootstrapResumeDecision,
    BootstrapRunIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.errors import (
    BootstrapCheckpointError,
    CheckpointAlreadyExists,
    CheckpointConcurrentModification,
    CheckpointCorrupt,
    CheckpointIncompatible,
    CheckpointNotFound,
    CheckpointPersistenceError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.filesystem_store import (
    FilesystemBootstrapCheckpointStore,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.ports import (
    BootstrapCheckpointStorePort,
)

__all__ = (
    "VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION",
    "VPI_STORAGE_BOOTSTRAP_ORDERING_POLICY",
    "BootstrapCheckpointCompatibility",
    "BootstrapCheckpointError",
    "BootstrapCheckpointIdentity",
    "BootstrapCheckpointState",
    "BootstrapCheckpointStorePort",
    "BootstrapResumeDecision",
    "BootstrapRunIdentity",
    "CheckpointAlreadyExists",
    "CheckpointConcurrentModification",
    "CheckpointCorrupt",
    "CheckpointIncompatible",
    "CheckpointNotFound",
    "CheckpointPersistenceError",
    "FilesystemBootstrapCheckpointStore",
    "advance_checkpoint_after_batch",
    "build_run_identity",
    "compute_resume_decision",
    "initial_checkpoint_state",
    "utc_now_iso",
    "validate_checkpoint_compatibility",
)
