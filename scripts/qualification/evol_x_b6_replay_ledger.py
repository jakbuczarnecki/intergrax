# © Artur Czarnecki. All rights reserved.

"""Append-only replay ledger for B6 attestations (path must be outside implementer repo)."""

from __future__ import annotations

import json
from pathlib import Path


class ReplayLedgerError(ValueError):
    """Replay ledger contract violation."""


def load_ledger(path: Path) -> list[str]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(item, str) for item in data):
        raise ReplayLedgerError("ledger must be a JSON array of replay_binding strings")
    return list(data)


def append_binding(path: Path, replay_binding: str, *, repo_root: Path) -> None:
    """Append one binding; fail-closed on duplicate or repo-internal ledger path."""
    resolved = path.resolve()
    repo = repo_root.resolve()
    try:
        resolved.relative_to(repo)
        in_repo = True
    except ValueError:
        in_repo = False
    if in_repo:
        raise ReplayLedgerError("replay ledger must not live inside the implementer repository")
    if len(replay_binding) != 64:
        raise ReplayLedgerError("replay_binding must be 64 hex chars")
    entries = load_ledger(path)
    if replay_binding in entries:
        raise ReplayLedgerError("duplicate replay_binding — replay attack or retry without new attestation_id")
    entries.append(replay_binding)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
