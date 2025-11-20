# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


"""Provider factory for Drop-In Knowledge Mode runtime.

Instantiates websearch provider instances based on the RuntimeConfig.websearch_providers
setting. Accepts either provider classes, already-instantiated providers, or simple
string names (legacy support for "google_cse", "bing").
"""
from __future__ import annotations

import logging
from typing import List, Union, Type

from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig
from intergrax.websearch.providers.base import WebSearchProvider
from intergrax.websearch.providers.google_cse_provider import GoogleCSEProvider
from intergrax.websearch.providers.bing_provider import BingWebProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    @staticmethod
    def instantiate_web_providers(config: RuntimeConfig) -> List[WebSearchProvider]:
        """Return list of WebSearchProvider instances according to config.

        Supports entries in `config.websearch_providers` as:
        - provider class (subclass of WebSearchProvider)
        - provider instance (already constructed)
        - string name ("google_cse", "bing")

        The factory logs warnings on failures but continues with remaining providers.
        """
        providers_spec = getattr(config, "websearch_providers", None)
        if not providers_spec:
            # fallback defaults
            return [GoogleCSEProvider(), BingWebProvider()]

        instances: List[WebSearchProvider] = []

        for p in providers_spec:
            try:
                # class -> instantiate
                if isinstance(p, type) and issubclass(p, WebSearchProvider):
                    instances.append(p())
                    continue

                # already an instance
                if isinstance(p, WebSearchProvider):
                    instances.append(p)
                    continue

                # string keys
                if isinstance(p, str):
                    key = p.lower()
                    if key in ("google_cse", "googlecse", "google-cse"):
                        instances.append(GoogleCSEProvider())
                        continue
                    if key in ("bing", "bing_web", "bingweb"):
                        instances.append(BingWebProvider())
                        continue
                    logger.warning("ProviderFactory: unknown provider name '%s'", p)
                    continue

                logger.warning("ProviderFactory: unsupported provider spec: %r", p)

            except Exception as e:
                logger.exception("ProviderFactory: failed to instantiate provider %r: %s", p, e)

        if not instances:
            logger.warning("ProviderFactory: no providers successfully instantiated; falling back to defaults")
            try:
                instances = [GoogleCSEProvider(), BingWebProvider()]
            except Exception:
                # last resort: empty list
                instances = []

        return instances


__all__ = ["ProviderFactory"]
