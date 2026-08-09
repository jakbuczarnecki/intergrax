# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PyMuPDF / Tesseract openers — only module with fitz / pytesseract imports."""

from __future__ import annotations

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.pymupdf.config import PymupdfIntegrationConfig


def parse_pymupdf_file(config: PymupdfIntegrationConfig, source: str) -> list[ParsedDocumentFragment]:
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
    except ModuleNotFoundError as exc:
        if exc.name == "langchain_community":
            raise RuntimeError(
                "Provider 'pymupdf' requires optional dependency group "
                "'rag-langchain-loaders'. Install Intergrax with "
                "'rag-langchain-loaders'."
            ) from exc
        raise

    docs = PyMuPDFLoader(source).load()
    if not docs:
        return []

    if config.enable_ocr:
        docs = _apply_ocr(config, source, docs)

    fragments: list[ParsedDocumentFragment] = []
    for index, doc in enumerate(docs):
        metadata = dict(doc.metadata or {})
        metadata["parser_backend"] = "pymupdf"
        fragments.append(
            ParsedDocumentFragment(
                text=doc.page_content or "",
                metadata=metadata,
                native_handle=None,
            )
        )
    return fragments


def _apply_ocr(config: PymupdfIntegrationConfig, source: str, docs: list) -> list:
    import fitz
    from PIL import Image
    import pytesseract

    pdf = None
    try:
        pdf = fitz.open(source)
    except Exception:
        return docs

    ocr_done = 0
    for doc in docs:
        content = (doc.page_content or "").strip()
        if content:
            continue
        if config.ocr_max_pages is not None and ocr_done >= config.ocr_max_pages:
            break
        page_index = doc.metadata.get("page") or doc.metadata.get("page_index")
        if page_index is None or pdf is None:
            continue
        try:
            page = pdf.load_page(int(page_index))
        except Exception:
            continue
        pix = page.get_pixmap(dpi=config.ocr_dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        cfg_parts = []
        if config.ocr_psm is not None:
            cfg_parts.append(f"--psm {int(config.ocr_psm)}")
        if config.ocr_oem is not None:
            cfg_parts.append(f"--oem {int(config.ocr_oem)}")
        ocr_config = " ".join(cfg_parts) if cfg_parts else None
        try:
            ocr_text = pytesseract.image_to_string(img, lang=config.ocr_lang, config=ocr_config) or ""
        except Exception:
            ocr_text = ""
        ocr_text = ocr_text.strip()
        if not ocr_text:
            continue
        doc.page_content = ocr_text
        md = dict(doc.metadata or {})
        md["ocr"] = True
        md["ocr_lang"] = config.ocr_lang
        md["ocr_dpi"] = config.ocr_dpi
        doc.metadata = md
        ocr_done += 1

    if pdf is not None:
        try:
            pdf.close()
        except Exception:
            pass
    return docs
