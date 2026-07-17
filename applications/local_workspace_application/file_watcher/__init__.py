# © Artur Czarnecki. All rights reserved.

"""LKW file-watcher package (LKW.7A/7B1/7B2A).

LKW.7A: snapshot contracts and job construction.
LKW.7B1: deterministic runtime state machine and enqueue binding.
LKW.7B2A: durable checkpoint contracts, atomic JSON store, export/restore.
No OS watcher process, sleep loop, or automatic checkpoint lifecycle lives here.
"""

from local_workspace_application.file_watcher.batching import (
    build_file_watcher_ingest_job,
    build_incremental_file_change_batch,
    file_change_token,
)
from local_workspace_application.file_watcher.checkpoint import (
    LKW_FILE_WATCHER_CHECKPOINT_SCHEMA_VERSION,
    FileWatcherCheckpoint,
    JsonFileWatcherCheckpointStore,
    build_file_watcher_checkpoint,
    decode_file_watcher_checkpoint,
    encode_file_watcher_checkpoint,
    file_watcher_checkpoint_path,
    restore_file_watcher_runtime,
)
from local_workspace_application.file_watcher.contracts import (
    FileChange,
    FileChangeKind,
    FileSnapshot,
    IncrementalFileChangeBatch,
)
from local_workspace_application.file_watcher.runtime import (
    BackgroundIngestEnqueuer,
    FileSnapshotProvider,
    FileWatcherCycleResult,
    FileWatcherCycleStatus,
    FileWatcherRuntime,
    FileWatcherRuntimeConfig,
    build_file_watcher_runtime,
)
from local_workspace_application.file_watcher.snapshot import (
    detect_file_changes,
    snapshot_allowed_roots,
    snapshot_file,
)

__all__ = [
    "BackgroundIngestEnqueuer",
    "FileChange",
    "FileChangeKind",
    "FileSnapshot",
    "FileSnapshotProvider",
    "FileWatcherCheckpoint",
    "FileWatcherCycleResult",
    "FileWatcherCycleStatus",
    "FileWatcherRuntime",
    "FileWatcherRuntimeConfig",
    "IncrementalFileChangeBatch",
    "JsonFileWatcherCheckpointStore",
    "LKW_FILE_WATCHER_CHECKPOINT_SCHEMA_VERSION",
    "build_file_watcher_checkpoint",
    "build_file_watcher_ingest_job",
    "build_file_watcher_runtime",
    "build_incremental_file_change_batch",
    "decode_file_watcher_checkpoint",
    "detect_file_changes",
    "encode_file_watcher_checkpoint",
    "file_change_token",
    "file_watcher_checkpoint_path",
    "restore_file_watcher_runtime",
    "snapshot_allowed_roots",
    "snapshot_file",
]
