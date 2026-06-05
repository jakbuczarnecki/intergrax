# © Artur Czarnecki. All rights reserved.

"""SQLite-backed user profile store (Phase MEM-2.1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from intergrax.memory.user_profile_serialization import user_profile_from_json, user_profile_to_json
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.utils.time_provider import SystemTimeProvider


class SQLiteUserProfileStore(UserProfileStore):
    """Persist ``UserProfile`` aggregates per tenant/user pair."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._connection = self._create_connection(db_path)
        self._initialize_schema()

    def _create_connection(self, db_path: str) -> sqlite3.Connection:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path))
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def _initialize_schema(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (tenant_id, user_id)
            );
            """
        )
        self._connection.commit()

    async def get_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> UserProfile:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT profile_json
            FROM user_profiles
            WHERE tenant_id = ? AND user_id = ?
            """,
            (tenant_id, user_id),
        )
        row = cursor.fetchone()
        if row is None:
            identity = UserIdentity(user_id=user_id)
            preferences = UserPreferences()
            return UserProfile(identity=identity, preferences=preferences)
        return user_profile_from_json(str(row[0]))

    async def save_profile(
        self,
        *,
        tenant_id: str,
        profile: UserProfile,
    ) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO user_profiles (
                tenant_id,
                user_id,
                profile_json,
                updated_at_utc
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                tenant_id,
                profile.identity.user_id,
                user_profile_to_json(profile),
                SystemTimeProvider.utc_now().isoformat(),
            ),
        )
        self._connection.commit()

    async def delete_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            DELETE FROM user_profiles
            WHERE tenant_id = ? AND user_id = ?
            """,
            (tenant_id, user_id),
        )
        self._connection.commit()
