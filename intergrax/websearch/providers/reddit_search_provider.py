# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Backward-compatible re-export — implementation lives in Integration Library."""

from intergrax.integrations.providers.search_provider.reddit.web_client import RedditAPIProvider

__all__ = ["RedditAPIProvider"]
