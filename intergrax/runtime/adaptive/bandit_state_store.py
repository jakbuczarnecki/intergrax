# © Artur Czarnecki. All rights reserved.

"""Thompson sampling bandit state persistence (Phase W-ADAPT-2.1)."""

from __future__ import annotations

import random
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from intergrax.runtime.adaptive.adaptation_models import BanditArmState


class BanditStateStore(Protocol):
    """Store for contextual bandit arm state partitioned by tenant."""

    def get_arm(
        self,
        *,
        tenant_id: str,
        task_class: str,
        arm_id: str,
    ) -> BanditArmState: ...

    def save_arm(self, state: BanditArmState) -> None: ...

    def list_arms(
        self,
        *,
        tenant_id: str,
        task_class: str | None = None,
    ) -> list[BanditArmState]: ...

    def record_reward(
        self,
        *,
        tenant_id: str,
        task_class: str,
        arm_id: str,
        reward: float,
    ) -> BanditArmState: ...

    def sample_arm_score(
        self,
        *,
        tenant_id: str,
        task_class: str,
        arm_id: str,
    ) -> float: ...

    def clear(self) -> None: ...


def _clamp_reward(reward: float) -> float:
    return max(0.0, min(1.0, reward))


def default_bandit_store_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "adaptive_harness" / "bandit_state.db"


class InMemoryBanditStateStore:
    """In-process bandit store for unit tests."""

    def __init__(self) -> None:
        self._arms: dict[tuple[str, str, str], BanditArmState] = {}

    def get_arm(
        self,
        *,
        tenant_id: str,
        task_class: str,
        arm_id: str,
    ) -> BanditArmState:
        key = (tenant_id, task_class, arm_id)
        if key not in self._arms:
            self._arms[key] = BanditArmState(
                tenant_id=tenant_id,
                task_class=task_class,
                arm_id=arm_id,
            )
        return self._arms[key]

    def save_arm(self, state: BanditArmState) -> None:
        self._arms[(state.tenant_id, state.task_class, state.arm_id)] = state

    def list_arms(
        self,
        *,
        tenant_id: str,
        task_class: str | None = None,
    ) -> list[BanditArmState]:
        items = [state for state in self._arms.values() if state.tenant_id == tenant_id]
        if task_class is not None:
            items = [state for state in items if state.task_class == task_class]
        return sorted(items, key=lambda item: item.arm_id)

    def record_reward(
        self,
        *,
        tenant_id: str,
        task_class: str,
        arm_id: str,
        reward: float,
    ) -> BanditArmState:
        clamped = _clamp_reward(reward)
        state = self.get_arm(tenant_id=tenant_id, task_class=task_class, arm_id=arm_id)
        updated = state.model_copy(
            update={
                "alpha": state.alpha + clamped,
                "beta": state.beta + (1.0 - clamped),
                "observation_count": state.observation_count + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        self.save_arm(updated)
        return updated

    def sample_arm_score(
        self,
        *,
        tenant_id: str,
        task_class: str,
        arm_id: str,
    ) -> float:
        state = self.get_arm(tenant_id=tenant_id, task_class=task_class, arm_id=arm_id)
        return random.betavariate(max(state.alpha, 1e-6), max(state.beta, 1e-6))

    def clear(self) -> None:
        self._arms.clear()


class SQLiteBanditStateStore:
    """SQLite-backed bandit store under ``build/adaptive_harness/``."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or default_bandit_store_path()
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit_arm_state (
                    tenant_id TEXT NOT NULL,
                    task_class TEXT NOT NULL,
                    arm_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, task_class, arm_id)
                );
                """
            )

    def get_arm(
        self,
        *,
        tenant_id: str,
        task_class: str,
        arm_id: str,
    ) -> BanditArmState:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM bandit_arm_state
                WHERE tenant_id = ? AND task_class = ? AND arm_id = ?
                """,
                (tenant_id, task_class, arm_id),
            ).fetchone()
        if row is None:
            return BanditArmState(tenant_id=tenant_id, task_class=task_class, arm_id=arm_id)
        return BanditArmState.model_validate_json(row["payload_json"])

    def save_arm(self, state: BanditArmState) -> None:
        payload = state.model_dump_json()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO bandit_arm_state (tenant_id, task_class, arm_id, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant_id, task_class, arm_id)
                DO UPDATE SET payload_json = excluded.payload_json
                """,
                (state.tenant_id, state.task_class, state.arm_id, payload),
            )

    def list_arms(
        self,
        *,
        tenant_id: str,
        task_class: str | None = None,
    ) -> list[BanditArmState]:
        query = "SELECT payload_json FROM bandit_arm_state WHERE tenant_id = ?"
        params: list[str] = [tenant_id]
        if task_class is not None:
            query += " AND task_class = ?"
            params.append(task_class)
        query += " ORDER BY arm_id ASC"
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [BanditArmState.model_validate_json(row["payload_json"]) for row in rows]

    def record_reward(
        self,
        *,
        tenant_id: str,
        task_class: str,
        arm_id: str,
        reward: float,
    ) -> BanditArmState:
        clamped = _clamp_reward(reward)
        state = self.get_arm(tenant_id=tenant_id, task_class=task_class, arm_id=arm_id)
        updated = state.model_copy(
            update={
                "alpha": state.alpha + clamped,
                "beta": state.beta + (1.0 - clamped),
                "observation_count": state.observation_count + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        self.save_arm(updated)
        return updated

    def sample_arm_score(
        self,
        *,
        tenant_id: str,
        task_class: str,
        arm_id: str,
    ) -> float:
        state = self.get_arm(tenant_id=tenant_id, task_class=task_class, arm_id=arm_id)
        return random.betavariate(max(state.alpha, 1e-6), max(state.beta, 1e-6))

    def clear(self) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM bandit_arm_state")
