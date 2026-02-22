# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from datetime import datetime

from intergrax.runtime.organization.organization_profile import (
    OrganizationProfile,
    OrganizationIdentity,
    OrganizationPreferences,
)
from intergrax.runtime.organization.organization_profile_store import (
    OrganizationProfileStore,
)
from intergrax.utils.time_provider import SystemTimeProvider


class SQLiteOrganizationProfileStore(OrganizationProfileStore):
    """
    SQLite-backed implementation of OrganizationProfileStore.

    - One table: organization_profiles
    - Full aggregate persistence (overwrite semantics)
    - JSON columns for list/dict fields
    - ISO 8601 datetime storage
    """

    def __init__(self, db_path: str) -> None:
        self._db_path: str = db_path
        self._connection: sqlite3.Connection = self._create_connection(db_path)
        self._initialize_schema()

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

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
            CREATE TABLE IF NOT EXISTS organization_profiles (
                organization_id TEXT PRIMARY KEY,

                name TEXT NOT NULL,
                legal_name TEXT,
                slug TEXT,
                primary_domain TEXT,
                industry TEXT,
                headquarters_location TEXT,
                default_timezone TEXT,

                default_language TEXT NOT NULL,
                default_output_format TEXT NOT NULL,
                tone_of_voice TEXT NOT NULL,
                allow_web_search INTEGER NOT NULL,
                allow_tools INTEGER NOT NULL,
                sensitive_topics_json TEXT,
                hard_constraints_json TEXT,
                soft_guidelines_json TEXT,

                system_instructions TEXT,
                summary_instructions TEXT,
                domain_summary TEXT,
                knowledge_summary TEXT,
                knowledge_sources_json TEXT,
                tags_json TEXT,

                extra_json TEXT,
                last_updated_utc TEXT NOT NULL,
                modified INTEGER NOT NULL
            );
            """
        )

        self._connection.commit()

    # ------------------------------------------------------------------
    # OrganizationProfileStore implementation
    # ------------------------------------------------------------------

    async def get_profile(self, organization_id: str) -> OrganizationProfile:
        cursor = self._connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM organization_profiles
            WHERE organization_id = ?
            """,
            (organization_id,),
        )

        row = cursor.fetchone()

        if row is None:
            # Return initialized default profile
            identity = OrganizationIdentity(
                organization_id=organization_id,
                name=organization_id,
            )
            return OrganizationProfile(identity=identity)

        (
            organization_id,
            name,
            legal_name,
            slug,
            primary_domain,
            industry,
            headquarters_location,
            default_timezone,
            default_language,
            default_output_format,
            tone_of_voice,
            allow_web_search,
            allow_tools,
            sensitive_topics_json,
            hard_constraints_json,
            soft_guidelines_json,
            system_instructions,
            summary_instructions,
            domain_summary,
            knowledge_summary,
            knowledge_sources_json,
            tags_json,
            extra_json,
            last_updated_utc,
            modified,
        ) = row

        identity = OrganizationIdentity(
            organization_id=organization_id,
            name=name,
            legal_name=legal_name,
            slug=slug,
            primary_domain=primary_domain,
            industry=industry,
            headquarters_location=headquarters_location,
            default_timezone=default_timezone,
        )

        preferences = OrganizationPreferences(
            default_language=default_language,
            default_output_format=default_output_format,
            tone_of_voice=tone_of_voice,
            allow_web_search=bool(allow_web_search),
            allow_tools=bool(allow_tools),
            sensitive_topics=json.loads(sensitive_topics_json)
            if sensitive_topics_json
            else [],
            hard_constraints=json.loads(hard_constraints_json)
            if hard_constraints_json
            else [],
            soft_guidelines=json.loads(soft_guidelines_json)
            if soft_guidelines_json
            else [],
        )

        profile = OrganizationProfile(
            identity=identity,
            preferences=preferences,
            system_instructions=system_instructions,
            summary_instructions=summary_instructions or "",
            domain_summary=domain_summary or "",
            knowledge_summary=knowledge_summary or "",
            knowledge_sources=json.loads(knowledge_sources_json)
            if knowledge_sources_json
            else [],
            tags=json.loads(tags_json) if tags_json else [],
            extra=json.loads(extra_json) if extra_json else {},
        )

        profile.last_updated_utc = datetime.fromisoformat(last_updated_utc)
        profile.modified = bool(modified)

        return profile

    async def save_profile(self, profile: OrganizationProfile) -> None:
        cursor = self._connection.cursor()

        profile.last_updated_utc = SystemTimeProvider.utc_now()

        cursor.execute(
            """
            INSERT OR REPLACE INTO organization_profiles (
                organization_id,
                name,
                legal_name,
                slug,
                primary_domain,
                industry,
                headquarters_location,
                default_timezone,
                default_language,
                default_output_format,
                tone_of_voice,
                allow_web_search,
                allow_tools,
                sensitive_topics_json,
                hard_constraints_json,
                soft_guidelines_json,
                system_instructions,
                summary_instructions,
                domain_summary,
                knowledge_summary,
                knowledge_sources_json,
                tags_json,
                extra_json,
                last_updated_utc,
                modified
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.identity.organization_id,
                profile.identity.name,
                profile.identity.legal_name,
                profile.identity.slug,
                profile.identity.primary_domain,
                profile.identity.industry,
                profile.identity.headquarters_location,
                profile.identity.default_timezone,
                profile.preferences.default_language,
                profile.preferences.default_output_format,
                profile.preferences.tone_of_voice,
                1 if profile.preferences.allow_web_search else 0,
                1 if profile.preferences.allow_tools else 0,
                json.dumps(profile.preferences.sensitive_topics)
                if profile.preferences.sensitive_topics
                else None,
                json.dumps(profile.preferences.hard_constraints)
                if profile.preferences.hard_constraints
                else None,
                json.dumps(profile.preferences.soft_guidelines)
                if profile.preferences.soft_guidelines
                else None,
                profile.system_instructions,
                profile.summary_instructions,
                profile.domain_summary,
                profile.knowledge_summary,
                json.dumps(profile.knowledge_sources)
                if profile.knowledge_sources
                else None,
                json.dumps(profile.tags) if profile.tags else None,
                json.dumps(profile.extra) if profile.extra else None,
                profile.last_updated_utc.isoformat(),
                1 if profile.modified else 0,
            ),
        )

        self._connection.commit()

    async def delete_profile(self, organization_id: str) -> None:
        cursor = self._connection.cursor()

        cursor.execute(
            """
            DELETE FROM organization_profiles
            WHERE organization_id = ?
            """,
            (organization_id,),
        )

        self._connection.commit()