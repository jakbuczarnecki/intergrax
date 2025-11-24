# © Artur Czarnecki. All rights reserved.
# Intergrax framework - proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Attachment resolution utilities for Drop-In Knowledge Mode.

This module defines:
  - `AttachmentResolver` protocol - an abstraction that knows how to turn
    an `AttachmentRef` into a local `Path` (or raise if it cannot).
  - `FileSystemAttachmentResolver` - a minimal implementation that handles
    local filesystem-based URIs, such as `file:///...`.

The goal is to decouple:
  - how and where attachments are stored (filesystem, DB, object storage),
  - from how the RAG pipeline consumes them (Intergrax document loaders).

In other words:
  Runtime deals with AttachmentRef -> AttachmentResolver -> Path,
  and then passes the resolved Paths to Intergrax RAG components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from intergrax.llm.conversational_memory import AttachmentRef


@runtime_checkable
class AttachmentResolver(Protocol):
    """
    Resolves an AttachmentRef into a local file path that can be passed
    to Intergrax document loaders.

    Implementations may:
      - download from object storage (S3, GCS, etc.),
      - fetch from a database and materialize as a temporary file,
      - validate and return local filesystem paths.

    The only hard requirement for the RAG pipeline is that the returned
    object can be consumed by the Intergrax documents loader (typically
    a string path).
    """

    async def resolve_to_path(self, attachment: AttachmentRef) -> Path:
        """
        Resolve the given AttachmentRef into a local filesystem Path.

        Raises:
          - FileNotFoundError if the attachment cannot be found.
          - ValueError for unsupported URI schemes.
        """
        ...


@dataclass
class FileSystemAttachmentResolver:
    """
    Minimal resolver for URIs like:

      - file:///absolute/path/to/file.pdf
      - file://C:/path/to/file.pdf
      - /relative/or/absolute/path (with empty scheme)

    This implementation is intended for:
      - local experiments,
      - Jupyter notebooks,
      - simple on-prem setups.

    In production you are expected to provide additional resolvers
    (e.g. S3AttachmentResolver, DBAttachmentResolver, etc.).
    """

    def _from_file_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)

        # Allow both explicit "file" scheme and an empty scheme (raw path).
        if parsed.scheme not in ("", "file"):
            raise ValueError(f"Unsupported URI scheme for file resolver: {parsed.scheme}")

        # Case 1: raw path without scheme (e.g. "D:/..." or "C:\\...")
        if parsed.scheme == "":
            # `uri` is a plain path string in this branch.
            return Path(uri).expanduser()

        # Case 2: proper file:// URI
        # Example on Windows:
        #   file:///D:/Projekty/intergrax/PROJECT_STRUCTURE.md
        #   -> parsed.path == "/D:/Projekty/intergrax/PROJECT_STRUCTURE.md"
        path_str = parsed.path

        # Fix Windows-style drive letters:
        # If path looks like "/D:/something", strip the leading slash.
        if (
            len(path_str) >= 4
            and path_str[0] == "/"
            and path_str[2] == ":"
        ):
            path_str = path_str[1:]  # "D:/Projekty/..."

        # UNC / netloc case (rare in this context, but we keep it for completeness)
        if parsed.netloc:
            # e.g. file://server/share/path
            # You can adapt this logic if you want UNC support.
            path_str = f"//{parsed.netloc}{path_str}"

        return Path(path_str).expanduser()

    async def resolve_to_path(self, attachment: AttachmentRef) -> Path:
        """
        Resolve the AttachmentRef's URI into an existing filesystem Path.

        This implementation assumes that `attachment.uri` is either:
          - a raw filesystem path, or
          - a `file://` URI pointing to a local file.
        """
        path = self._from_file_uri(attachment.uri)
        if not path.exists():
            raise FileNotFoundError(f"Attachment path does not exist: {path}")
        return path
