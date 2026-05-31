# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Browser automation integration contract (§7.1.2, Phase M.6 P3)."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    html: str = ""
    status_code: int = 200
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class BrowserAutomation(Protocol):
    """Headless browser facade for dynamic web research."""

    def fetch_page(self, url: str, *, wait_until: str = "load") -> PageContent:
        """Navigate to ``url`` and return normalized page content."""

    def close(self) -> None:
        """Release browser resources."""
