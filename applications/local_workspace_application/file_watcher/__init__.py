# © Artur Czarnecki. All rights reserved.

"""LKW file-watcher package (LKW.7A) — snapshot contracts and job construction only.

No watcher process, debounce loop, or message-bus enqueue lives here.
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
from local_workspace_application.file_watcher.snapshot import (
    detect_file_changes,
    snapshot_allowed_roots,
    snapshot_file,
)

__all__ = [
    "FileChange",
    "FileChangeKind",
    "FileSnapshot",
    "IncrementalFileChangeBatch",
    "build_file_watcher_ingest_job",
    "build_incremental_file_change_batch",
    "detect_file_changes",
    "file_change_token",
    "snapshot_allowed_roots",
    "snapshot_file",
]
