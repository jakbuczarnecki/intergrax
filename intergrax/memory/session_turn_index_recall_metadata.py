# © Artur Czarnecki. All rights reserved.

"""Runtime recall metadata adapter for typed session turn index hits."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.memory.contracts.session_turn_index import SessionTurnIndexHit


def session_turn_index_hits_as_recall_metadata(
    hits: Sequence[SessionTurnIndexHit],
) -> list[dict[str, str | float]]:
    """Serialize typed hits for legacy runtime / CE recall handles."""
    rows: list[dict[str, str | float]] = []
    for hit in hits:
        text = (hit.message.content or "").strip()
        rows.append(
            {
                "text": text,
                "score": hit.score,
                "message_id": hit.message.entry_id,
                "entry_id": hit.entry_id,
                "session_id": hit.session_id,
            }
        )
    return rows
