# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Dict, Any, Optional

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.websearch.schemas.page_content import PageContent


HAS_TRAFILATURA = None
trafilatura = None


def _beautiful_soup(html: str):
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as exc:
        if exc.name == "bs4":
            raise IntegrationDependencyError(
                "Web parsing requires optional dependency 'beautifulsoup4'. "
                "Install Intergrax-ai[parsing-web].",
                integration_name="web_parser",
            ) from exc
        raise
    return BeautifulSoup(html, "lxml")


def _load_trafilatura():
    global HAS_TRAFILATURA, trafilatura
    if HAS_TRAFILATURA is False:
        return None
    if HAS_TRAFILATURA and trafilatura is not None:
        return trafilatura
    try:
        import trafilatura as module
    except ImportError:
        HAS_TRAFILATURA = False
        trafilatura = None
        return None
    HAS_TRAFILATURA = True
    trafilatura = module
    return module


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

    soup = _beautiful_soup(page.html)

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


def extract_advanced(
    page: PageContent,
    *,
    min_text_chars: int = 300,
    overwrite_existing_text: bool = True,
) -> PageContent:
    """
    Performs advanced readability-based extraction on a PageContent instance.

    This step is applied AFTER lightweight metadata extraction (extract_basic).

    Responsibilities:
    - Remove obvious boilerplate elements (scripts, styles, iFrames, navigation).
    - Prefer trafilatura (when available) to extract primary readable content.
    - Fallback to BeautifulSoup plain-text extraction if trafilatura fails.
    - Normalize whitespace and reduce noise.
    - Optionally overwrite existing text if it already exists.

    This function is intentionally synchronous to allow use in both
    synchronous and async pipelines.
    """
    if not page or not page.html:
        return page

    html = page.html
    extracted_text: Optional[str] = None
    extraction_method = "beautifulsoup"

    # ---------------------------------
    # STEP 1: Try readability extraction via trafilatura (if installed)
    # ---------------------------------
    trafilatura_module = _load_trafilatura()
    if trafilatura_module is not None:
        try:
            extracted_text = trafilatura_module.extract(
                html,
                url=page.final_url or None,
                include_comments=False,
                favor_precision=True,
            )
            if extracted_text:
                extraction_method = "trafilatura"
        except Exception:
            extracted_text = None

    # ---------------------------------
    # STEP 2: Fallback to manual HTML cleanup and extraction
    # ---------------------------------
    if not extracted_text:
        soup = _beautiful_soup(html)

        # Remove non-content HTML nodes
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "iframe",
                "header",
                "footer",
                "nav",
                "aside",
                "form",
                "button",
                "svg",
                "canvas",
            ]
        ):
            tag.decompose()

        # Extract readable text and normalize formatting
        raw_text = soup.get_text(separator="\n", strip=True)
        extracted_text = _normalize_whitespace(raw_text)
        extraction_method = "beautifulsoup"

    # ---------------------------------
    # STEP 3: Respect overwrite mode
    # If page already contains text and overwrite is disabled,
    # attach metadata and return without replacing.
    # ---------------------------------
    if page.text and not overwrite_existing_text:
        _attach_extraction_metadata(
            page,
            extraction_method=extraction_method,
            overwritten=False,
            length_before=len(page.text),
            length_after=len(extracted_text or ""),
        )
        return page

    # ---------------------------------
    # STEP 4: Replace or set PageContent.text with extracted form
    # ---------------------------------
    length_before = len(page.text) if page.text else 0
    page.text = extracted_text

    # Optional threshold check: flag unusually small content bodies
    if extracted_text and len(extracted_text) < min_text_chars:
        extra = page.extra or {}
        extra["advanced_extraction_warning"] = (
            f"Extracted content too short ({len(extracted_text)} chars; "
            f"min={min_text_chars})."
        )
        page.extra = extra

    # ---------------------------------
    # STEP 5: Attach trace/debug metadata for observability
    # ---------------------------------
    _attach_extraction_metadata(
        page,
        extraction_method=extraction_method,
        overwritten=True,
        length_before=length_before,
        length_after=len(extracted_text or ""),
    )

    return page


# =====================================================================
# Helper Functions
# =====================================================================

def _normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespace by stripping extra line breaks,
    trimming trailing/leading spaces, and collapsing empty lines.
    """
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


def _attach_extraction_metadata(
    page: PageContent,
    *,
    extraction_method: str,
    overwritten: bool,
    length_before: int,
    length_after: int,
) -> None:
    """
    Attaches metadata to PageContent.extra to help debugging and analysis.

    Metadata recorded includes:
    - extraction method used,
    - whether existing text was overwritten,
    - before/after text length comparison.

    This supports observability in multi-source pipelines.
    """
    extra = page.extra or {}
    extraction_info: Dict[str, Any] = extra.get("advanced_extraction", {})

    extraction_info.update(
        {
            "extraction_method": extraction_method,
            "overwritten_existing_text": overwritten,
            "length_before": length_before,
            "length_after": length_after,
        }
    )

    extra["advanced_extraction"] = extraction_info
    page.extra = extra