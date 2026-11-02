from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.repo_identity import read_repo_identity


def test_repo_identity_reads_head_and_branch() -> None:
    root = Path(__file__).resolve().parents[4]
    identity = read_repo_identity(root)
    assert len(identity.head_sha) == 40
    assert identity.branch
    assert identity.remote_url
