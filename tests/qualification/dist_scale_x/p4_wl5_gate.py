# © Artur Czarnecki. All rights reserved.

"""Aggregate rate gate for WL-5 OS workers (≤40 offers/s)."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class Wl5RateGate:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS offered (seq INTEGER PRIMARY KEY)",
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS claimed (seq INTEGER PRIMARY KEY, worker TEXT NOT NULL)",
            )
            conn.commit()

    def offer(self, seq: int) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute("INSERT OR IGNORE INTO offered(seq) VALUES (?)", (seq,))
            conn.commit()

    def try_claim(self, worker: str) -> int | None:
        with sqlite3.connect(self._path) as conn:
            row = conn.execute(
                """
                SELECT o.seq FROM offered o
                LEFT JOIN claimed c ON c.seq = o.seq
                WHERE c.seq IS NULL
                ORDER BY o.seq ASC
                LIMIT 1
                """,
            ).fetchone()
            if row is None:
                return None
            seq = int(row[0])
            try:
                conn.execute(
                    "INSERT INTO claimed(seq, worker) VALUES (?, ?)",
                    (seq, worker),
                )
                conn.commit()
                return seq
            except sqlite3.IntegrityError:
                conn.rollback()
                return None

    def pending_offers(self) -> int:
        with sqlite3.connect(self._path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM offered o
                LEFT JOIN claimed c ON c.seq = o.seq
                WHERE c.seq IS NULL
                """,
            ).fetchone()
            return int(row[0]) if row else 0
