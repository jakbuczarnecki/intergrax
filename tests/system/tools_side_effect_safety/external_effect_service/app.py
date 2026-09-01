"""External side-effect oracle service for TOOLS-SIDE-EFFECT-SAFETY Docker proof."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field


DATABASE_URL = os.environ.get(
    "EFFECT_DATABASE_URL",
    "postgresql://intergrax:intergrax@effect-postgres:5432/side_effect_proof",
)


class ChargeRequest(BaseModel):
    business_operation_id: str = Field(min_length=1)
    amount: int = Field(ge=0)
    worker_source: str | None = None


class ChargeResponse(BaseModel):
    effect_id: int
    business_operation_id: str
    committed_at: str


@dataclass
class HoldState:
    before_commit: asyncio.Event = field(default_factory=asyncio.Event)
    after_commit: asyncio.Event = field(default_factory=asyncio.Event)
    released_before: bool = False
    released_after: bool = False


class _Runtime:
    def __init__(self) -> None:
        self._holds: dict[str, HoldState] = {}
        self._lock = threading.Lock()

    def hold_for(self, business_operation_id: str) -> HoldState:
        with self._lock:
            state = self._holds.setdefault(business_operation_id, HoldState())
            state.before_commit.clear()
            state.after_commit.clear()
            state.released_before = False
            state.released_after = False
            return state

    def release_before(self, business_operation_id: str) -> bool:
        with self._lock:
            state = self._holds.get(business_operation_id)
            if state is None:
                return False
            state.released_before = True
            state.before_commit.set()
            return True

    def release_after(self, business_operation_id: str) -> bool:
        with self._lock:
            state = self._holds.get(business_operation_id)
            if state is None:
                return False
            state.released_after = True
            state.after_commit.set()
            return True


runtime = _Runtime()


def _ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(
        "ALTER TABLE effect_attempts ADD COLUMN IF NOT EXISTS received_delay_ms INTEGER",
    )
    conn.execute(
        "ALTER TABLE effect_attempts ADD COLUMN IF NOT EXISTS effect_committed_at TIMESTAMPTZ",
    )
    conn.execute(
        "ALTER TABLE effect_attempts ADD COLUMN IF NOT EXISTS response_started_at TIMESTAMPTZ",
    )
    conn.execute(
        "ALTER TABLE effect_attempts ADD COLUMN IF NOT EXISTS response_finished_at TIMESTAMPTZ",
    )
    conn.commit()


def _connect() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL)


def _wait_for_db(deadline_s: float = 60.0) -> None:
    import time

    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        try:
            with _connect() as conn:
                conn.execute("SELECT 1")
                _ensure_schema(conn)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("effect database did not become ready")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _wait_for_db()
    yield


app = FastAPI(title="Side Effect Oracle", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    with _connect() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/admin/reset")
def admin_reset() -> dict[str, str]:
    with _connect() as conn:
        conn.execute("TRUNCATE effect_attempts RESTART IDENTITY")
        conn.execute("TRUNCATE effects RESTART IDENTITY")
        conn.commit()
    return {"status": "reset"}


@app.post("/admin/release-before/{business_operation_id}")
def release_before(business_operation_id: str) -> dict[str, bool]:
    return {"released": runtime.release_before(business_operation_id)}


@app.post("/admin/release-after/{business_operation_id}")
def release_after(business_operation_id: str) -> dict[str, bool]:
    return {"released": runtime.release_after(business_operation_id)}


@app.get("/admin/effects/{business_operation_id}")
def get_effects(business_operation_id: str) -> dict[str, Any]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, business_operation_id, payload, worker_source, created_at
            FROM effects
            WHERE business_operation_id = %s
            ORDER BY id
            """,
            (business_operation_id,),
        ).fetchall()
        attempts = conn.execute(
            """
            SELECT id, proof_mode, received_delay_ms, worker_source,
                   received_at, effect_committed_at, response_started_at, response_finished_at
            FROM effect_attempts
            WHERE business_operation_id = %s
            ORDER BY id
            """,
            (business_operation_id,),
        ).fetchall()
    return {
        "effects": [
            {
                "id": row[0],
                "business_operation_id": row[1],
                "payload": row[2],
                "worker_source": row[3],
                "created_at": row[4].isoformat(),
            }
            for row in rows
        ],
        "attempt_count": len(attempts),
        "attempts": [
            {
                "id": row[0],
                "proof_mode": row[1],
                "received_delay_ms": row[2],
                "worker_source": row[3],
                "received_at": row[4].isoformat() if row[4] else None,
                "effect_committed_at": row[5].isoformat() if row[5] else None,
                "response_started_at": row[6].isoformat() if row[6] else None,
                "response_finished_at": row[7].isoformat() if row[7] else None,
            }
            for row in attempts
        ],
    }


@app.get("/admin/duplicates")
def duplicate_scan() -> dict[str, Any]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT business_operation_id, COUNT(*) AS cnt
            FROM effects
            GROUP BY business_operation_id
            HAVING COUNT(*) > 1
            ORDER BY business_operation_id
            """,
        ).fetchall()
    return {"duplicates": [{"business_operation_id": r[0], "count": r[1]} for r in rows]}


@app.post("/charge")
async def charge(request: Request, body: ChargeRequest) -> Response:
    proof_mode = request.headers.get("X-Proof-Mode", "normal")
    delay_ms = int(request.headers.get("X-Proof-Delay-Ms", "0"))
    worker_source = body.worker_source or request.headers.get("X-Worker-Source")
    received_at = datetime.now(UTC)

    with _connect() as conn:
        attempt_row = conn.execute(
            """
            INSERT INTO effect_attempts (
                business_operation_id, request_payload, worker_source,
                proof_mode, received_delay_ms, received_at
            )
            VALUES (%s, %s::jsonb, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                body.business_operation_id,
                json.dumps(body.model_dump()),
                worker_source,
                proof_mode,
                delay_ms,
                received_at,
            ),
        ).fetchone()
        conn.commit()
    attempt_id = int(attempt_row[0])

    if proof_mode == "fail_before_commit":
        raise HTTPException(status_code=500, detail="fail_before_commit")

    hold = None
    if proof_mode in {"hold_before_commit", "hold_after_commit"}:
        hold = runtime.hold_for(body.business_operation_id)

    if proof_mode == "hold_before_commit" and hold is not None:
        try:
            await asyncio.wait_for(hold.before_commit.wait(), timeout=300.0)
        except TimeoutError:
            raise HTTPException(status_code=504, detail="hold_before_commit_timeout") from None

    committed_at = datetime.now(UTC)
    with _connect() as conn:
        row = conn.execute(
            """
            INSERT INTO effects (business_operation_id, payload, worker_source, created_at)
            VALUES (%s, %s::jsonb, %s, %s)
            RETURNING id
            """,
            (
                body.business_operation_id,
                json.dumps({"amount": body.amount}),
                worker_source,
                committed_at,
            ),
        ).fetchone()
        conn.execute(
            "UPDATE effect_attempts SET effect_committed_at = %s WHERE id = %s",
            (committed_at, attempt_id),
        )
        conn.commit()
    effect_id = int(row[0])

    if proof_mode == "hold_after_commit" and hold is not None:
        try:
            await asyncio.wait_for(hold.after_commit.wait(), timeout=300.0)
        except TimeoutError:
            raise HTTPException(status_code=504, detail="hold_after_commit_timeout") from None

    response_started_at = datetime.now(UTC)
    if proof_mode in {"delay_after_commit", "timeout_response"}:
        await asyncio.sleep(max(delay_ms, 0) / 1000.0)

    if proof_mode == "abort_connection":
        # Force connection close without a complete HTTP response.
        request.scope["transport"].close()  # type: ignore[attr-defined]
        await asyncio.sleep(3600)

    if proof_mode == "bad_output_after_commit":
        response_finished_at = datetime.now(UTC)
        with _connect() as conn:
            conn.execute(
                """
                UPDATE effect_attempts
                SET response_started_at = %s, response_finished_at = %s
                WHERE id = %s
                """,
                (response_started_at, response_finished_at, attempt_id),
            )
            conn.commit()
        return Response(
            content=json.dumps({"unexpected": True}),
            media_type="application/json",
            status_code=200,
        )

    payload = ChargeResponse(
        effect_id=effect_id,
        business_operation_id=body.business_operation_id,
        committed_at=committed_at.isoformat(),
    )
    response_finished_at = datetime.now(UTC)
    with _connect() as conn:
        conn.execute(
            """
            UPDATE effect_attempts
            SET response_started_at = %s, response_finished_at = %s
            WHERE id = %s
            """,
            (response_started_at, response_finished_at, attempt_id),
        )
        conn.commit()
    return Response(content=payload.model_dump_json(), media_type="application/json")
