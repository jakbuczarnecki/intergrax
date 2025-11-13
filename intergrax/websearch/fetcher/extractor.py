# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Dict, Any
from bs4 import BeautifulSoup

from intergrax.websearch.schemas.page_content import PageContent


def extract_basic(page: PageContent) -> PageContent:
    """
    Performs lightweight HTML extraction on a PageContent instance.

    Responsibilities:
      - Extract <title>.
      - Extract meta description.
      - Extract <html lang> attribute.
      - Extract Open Graph meta tags (og:*).
      - Produce a plain-text version of the page.

    This function is intentionally conservative:
      - It does not perform advanced readability or boilerplate removal.
      - It does not modify HTTP-related fields (status_code, final_url, etc.).
    """
    if not page or not page.html:
        return page

    soup = BeautifulSoup(page.html, "lxml")

    # Title
    if not page.title:
        if soup.title and soup.title.string:
            page.title = soup.title.string.strip()

    # Description: meta[name="description"] or meta[property="og:description"]
    if not page.description:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if not meta_desc:
            meta_desc = soup.find("meta", attrs={"property": "og:description"})
        if meta_desc and meta_desc.get("content"):
            page.description = meta_desc["content"].strip()

    # Language: <html lang="...">
    if not page.lang:
        html_tag = soup.find("html")
        if html_tag:
            lang_attr = html_tag.get("lang") or html_tag.get("xml:lang")
            if lang_attr:
                page.lang = lang_attr.strip()

    # Open Graph tags: meta[property^="og:"]
    if not page.og:
        og: Dict[str, Any] = {}
        for tag in soup.find_all("meta"):
            prop = tag.get("property") or tag.get("name")
            if not prop:
                continue
            prop_lower = prop.lower()
            if prop_lower.startswith("og:") and tag.get("content"):
                og[prop_lower] = tag["content"].strip()
        page.og = og

    # Plain text extraction
    if not page.text:
        # Using '\n' as separator to preserve some structure.
        raw_text = soup.get_text(separator="\n")
        page.text = raw_text.strip()

    return page
