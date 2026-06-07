# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class BrowserFetchPageInput(BaseModel):
    url: str = Field(..., min_length=1)
    wait_until: str = Field(default="load", description="Navigation wait condition (load, domcontentloaded, networkidle).")


class BrowserFetchPageOutput(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    status_code: int = 200
    html_length: int = 0
