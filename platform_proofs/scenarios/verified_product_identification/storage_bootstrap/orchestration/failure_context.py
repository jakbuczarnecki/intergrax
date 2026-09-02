"""Actionable ingest failure messages without credential leakage."""

from __future__ import annotations


def format_ingest_failure(
    *,
    stage: str,
    batch_ordinal: int,
    checkpoint_rows: int,
    provider_role: str,
    detail: str,
) -> str:
    return (
        f"{stage} failed: batch={batch_ordinal} checkpoint_rows={checkpoint_rows} "
        f"provider={provider_role} detail={detail}"
    )
