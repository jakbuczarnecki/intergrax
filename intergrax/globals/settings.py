# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class GlobalSettings:
    """
    Global framework-wide configuration.

    This is the single source of truth for defaults such as:
      - default language,
      - default locale,
      - default timezone,
      - default LLM model (if needed),
      - etc.

    Local modules (runtime, memory, tools) should import and use these
    defaults instead of hardcoding values.
    """

    # Default language for prompts, instructions and extracted content.
    # Can be overridden via environment variable.
    default_language: str = os.getenv("INTERGRAX_DEFAULT_LANGUAGE", "pl")    

    # Default locale (used only if no more specific user/org-level value).
    default_locale: str = os.getenv("INTERGRAX_DEFAULT_LOCALE", "pl-PL")

    # Default region (used only if no more specific user/org-level value).
    default_region: str = os.getenv("INTERGRAX_DEFAULT_REGION", "pl-PL")

    # Default timezone (used only if no more specific user/org-level value).
    default_timezone: str = os.getenv("INTERGRAX_DEFAULT_TIMEZONE", "Europe/Warsaw")


# Singleton-style global settings instance used across the framework.
GLOBAL_SETTINGS = GlobalSettings()