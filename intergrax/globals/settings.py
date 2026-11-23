# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GlobalSettings:
    """
    Global framework-wide configuration.

    This is the single source of truth for defaults such as:
      - default language,
      - default locale,
      - default timezone,
      - default LLM model,
      - default memory/consolidation intervals, etc.

    Local modules (runtime, memory, tools) should import and use these
    defaults instead of hardcoding values.

    Persistent storage and audit timestamps use UTC unless a tenant or host
    profile supplies an explicit override. Presentation locale/timezone may
    differ per tenant via environment or host configuration.
    """

    # ------------------------------------------------------------------
    # Language, locale, timezone defaults (neutral platform baseline)
    # ------------------------------------------------------------------

    default_language: str = os.getenv("INTERGRAX_DEFAULT_LANGUAGE", "en")

    default_locale: str = os.getenv("INTERGRAX_DEFAULT_LOCALE", "en-US")

    default_region: str = os.getenv("INTERGRAX_DEFAULT_REGION", "en-US")

    default_timezone: str = os.getenv("INTERGRAX_DEFAULT_TIMEZONE", "UTC")

    # ------------------------------------------------------------------
    # Session memory / consolidation defaults
    # ------------------------------------------------------------------

    # Default interval (in user turns) for mid-session consolidation.
    # If None or non-integer, the runtime components should treat it as disabled
    # or fall back to a module-local constant. Since os.getenv ALWAYS returns
    # strings, we cast to int when possible; otherwise None.
    raw_interval = os.getenv("INTERGRAX_DEFAULT_USER_TURNS_CONSOLIDATION_INTERVAL", None)

    default_user_turns_consolidation_interval: int | None = (
        int(raw_interval) if raw_interval and raw_interval.isdigit() else None
    )

    raw_cooldown_sec = os.getenv("INTERGRAX_DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS", None)

    default_consolidation_cooldown_seconds: int | None = (
        int(raw_cooldown_sec) if raw_cooldown_sec and raw_cooldown_sec.isdigit() else None
    )


# Singleton-style global settings instance used across the framework.
GLOBAL_SETTINGS = GlobalSettings()
