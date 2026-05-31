# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite-backed delivery receipts and dead-letter ledger (Appendix B.13)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from intergrax.runtime.notifications.deliveries.delivery_ledger import DeliveryReceipt, _utc_now


class SQLiteDeliveryLedger:
    """Durable delivery ledger for production Tier-3 hosts."""

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS delivery_receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    delivery_id TEXT NOT NULL UNIQUE,
                    destination TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    delivered_at_utc TEXT NOT NULL,
                    last_error TEXT,
                    payload_summary_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_delivery_receipts_status
                ON delivery_receipts (status, id DESC);
                """
            )

    def record_success(
        self,
        *,
        destination: str,
        task_id: str,
        channel: str,
        attempts: int,
        payload_summary: Optional[Dict[str, str]] = None,
    ) -> DeliveryReceipt:
        receipt = DeliveryReceipt(
            delivery_id=f"dlv_{uuid4().hex}",
            destination=destination,
            task_id=task_id,
            channel=channel,
            status="delivered",
            attempts=attempts,
            delivered_at_utc=_utc_now(),
            payload_summary=dict(payload_summary or {}),
        )
        self._insert(receipt)
        return receipt

    def record_dead_letter(
        self,
        *,
        destination: str,
        task_id: str,
        channel: str,
        attempts: int,
        last_error: str,
        payload_summary: Optional[Dict[str, str]] = None,
    ) -> DeliveryReceipt:
        receipt = DeliveryReceipt(
            delivery_id=f"dlv_{uuid4().hex}",
            destination=destination,
            task_id=task_id,
            channel=channel,
            status="dead_letter",
            attempts=attempts,
            delivered_at_utc=_utc_now(),
            last_error=last_error,
            payload_summary=dict(payload_summary or {}),
        )
        self._insert(receipt)
        return receipt

    def _insert(self, receipt: DeliveryReceipt) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO delivery_receipts (
                    delivery_id, destination, task_id, channel, status,
                    attempts, delivered_at_utc, last_error, payload_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.delivery_id,
                    receipt.destination,
                    receipt.task_id,
                    receipt.channel,
                    receipt.status,
                    receipt.attempts,
                    receipt.delivered_at_utc,
                    receipt.last_error,
                    json.dumps(receipt.payload_summary),
                ),
            )

    def list_receipts(self, *, limit: int = 100) -> List[DeliveryReceipt]:
        return self._list(status="delivered", limit=limit)

    def list_dead_letters(self, *, limit: int = 100) -> List[DeliveryReceipt]:
        return self._list(status="dead_letter", limit=limit)

    def _list(self, *, status: str, limit: int) -> List[DeliveryReceipt]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM delivery_receipts
                WHERE status = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        return [_row_to_receipt(row) for row in rows]


def _row_to_receipt(row: sqlite3.Row) -> DeliveryReceipt:
    summary_raw = row["payload_summary_json"] or "{}"
    try:
        summary = json.loads(summary_raw)
    except json.JSONDecodeError:
        summary = {}
    return DeliveryReceipt(
        delivery_id=row["delivery_id"],
        destination=row["destination"],
        task_id=row["task_id"],
        channel=row["channel"],
        status=row["status"],
        attempts=int(row["attempts"]),
        delivered_at_utc=row["delivered_at_utc"],
        last_error=row["last_error"],
        payload_summary={str(k): str(v) for k, v in dict(summary).items()},
    )
