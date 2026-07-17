# © Artur Czarnecki. All rights reserved.

"""LKW file-watcher package (LKW.7A/7B1).

LKW.7A: snapshot contracts and job construction.
LKW.7B1: deterministic runtime state machine and enqueue binding.
No OS watcher process, sleep loop, or checkpoint persistence lives here.
"""

from local_workspace_application.file_watcher.batching import (
    build_file_watcher_ingest_job,
    build_incremental_file_change_batch,
    file_change_token,
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
    "FileWatcherCycleResult",
    "FileWatcherCycleStatus",
    "FileWatcherRuntime",
    "FileWatcherRuntimeConfig",
    "IncrementalFileChangeBatch",
    "build_file_watcher_ingest_job",
    "build_file_watcher_runtime",
    "build_incremental_file_change_batch",
    "detect_file_changes",
    "file_change_token",
    "snapshot_allowed_roots",
    "snapshot_file",
]
