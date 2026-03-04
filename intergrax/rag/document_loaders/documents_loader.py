# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
import logging
import docx
from intergrax.logging import IntergraxLogging


from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.multimedia.audio_smart_loader import AudioSmartLoader


from pathlib import Path


from typing import (
    Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union,
)

from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document
from tqdm.auto import tqdm
from typing import Literal

from intergrax.multimedia.image_smart_loader import ImageSmartLoader
from intergrax.multimedia.video_smart_loader import VideoSmartLoader
from intergrax.rag.document_loaders.handlers.doc_smart_document_handler import DocxParagraphLoader
from intergrax.rag.document_loaders.handlers.excel_smart_document_handler import ExcelSmartLoader
from intergrax.rag.document_loaders.handlers.pdf_smart_document_handler import PdfSmartLoader


try:
    import pandas as pd
except Exception:
    pd = None

try:
    import openpyxl  # engine for .xlsx (used by pandas)
except Exception:
    openpyxl = None

try:
    import xlrd  # engine for .xls (legacy Excel files)
except Exception:
    xlrd = None

try:
    import pytesseract
except Exception:
    pytesseract = None


logger = IntergraxLogging.get_logger(__name__, component="rag")

MetadataFn = Callable[[Document, Path], Optional[Dict]]

DOCX_MODE = Literal["fulltext", "paragraphs", "headings"]

EXCEL_MODE = Literal["rows", "sheets", "markdown"]

class DocumentsLoader:
    """Robust, extensible document loader with metadata injection and safety guards."""

    def __init__(
        self,
        *,
        file_patterns: Iterable[str] = ("**/*",),  # include files without extension too
        extensions_map: Optional[Mapping[str, Callable[[str], object]]] = None,
        exclude_globs: Optional[Iterable[str]] = None,
        follow_symlinks: bool = False,
        max_files: Optional[int] = None,
        max_file_size_mb: Optional[int] = 64,
        docx_mode: DOCX_MODE = "fulltext",
        pdf_enable_ocr: bool = False,
        pdf_ocr_lang: str = "eng",
        pdf_ocr_dpi: int = 200,
        pdf_ocr_psm: Optional[int] = None,
        pdf_ocr_oem: Optional[int] = None,
        pdf_ocr_max_pages: Optional[int] = None,
        excel_mode: EXCEL_MODE = "rows",           # "rows" | "sheets" | "markdown"
        excel_header: int = 0,
        excel_sheet: str | int | None = None,
        excel_na_filter: bool = True,
        excel_max_rows_per_sheet: Optional[int] = None,
        csv_encoding: Optional[str] = None,
        csv_delimiter: Optional[str] = None,
        # --- image (existing) ---
        image_ocr_lang: str = "eng",
        image_ocr_psm: Optional[int] = None,
        image_ocr_oem: Optional[int] = None,
        image_extract_exif: bool = True,
        image_max_dim: Optional[int] = 2000,
        # --- image NEW: captioning modes via framework adapter ---
        image_text_mode: Literal["ocr", "caption", "both"] = "both",
        image_caption_llm: Optional[LLMAdapter] = None, 
        image_both_joiner: str = "\n\n---\n\n",
    ):
        self._file_patterns = tuple(file_patterns)
        self._exclude_globs = tuple(exclude_globs or ())
        self._follow_symlinks = follow_symlinks
        self._max_files = max_files
        self._max_file_size_mb = max_file_size_mb
        self._docx_mode = docx_mode 

        self._pdf_enable_ocr = bool(pdf_enable_ocr)
        self._pdf_ocr_lang = pdf_ocr_lang
        self._pdf_ocr_dpi = int(pdf_ocr_dpi)
        self._pdf_ocr_psm = pdf_ocr_psm
        self._pdf_ocr_oem = pdf_ocr_oem
        self._pdf_ocr_max_pages = pdf_ocr_max_pages

        self._excel_mode = excel_mode
        self._excel_header = excel_header
        self._excel_sheet = excel_sheet
        self._excel_na_filter = excel_na_filter
        self._excel_max_rows_per_sheet = excel_max_rows_per_sheet
        self._csv_encoding = csv_encoding
        self._csv_delimiter = csv_delimiter

        # images (existing)
        self._image_ocr_lang = image_ocr_lang
        self._image_ocr_psm = image_ocr_psm
        self._image_ocr_oem = image_ocr_oem
        self._image_extract_exif = bool(image_extract_exif)
        self._image_max_dim = image_max_dim
        # images NEW (captioning via adapter)
        self._image_text_mode = image_text_mode
        self._image_caption_llm = image_caption_llm
        self._image_both_joiner = image_both_joiner


        default_map: Dict[str, Callable[[str], object]] = {
            ".txt":  lambda p: TextLoader(p, autodetect_encoding=True),  # autodetect
            
            ".md":       lambda p: TextLoader(p, autodetect_encoding=True),
            ".markdown": lambda p: TextLoader(p, autodetect_encoding=True),

            ".docx": lambda p: Docx2txtLoader(p),            
            ".htm":  lambda p: UnstructuredHTMLLoader(p),
            ".html": lambda p: UnstructuredHTMLLoader(p),
            ".pdf":  lambda p: PdfSmartLoader(
                p,
                enable_ocr=self._pdf_enable_ocr,
                ocr_lang=self._pdf_ocr_lang,
                ocr_dpi=self._pdf_ocr_dpi,
                ocr_psm=self._pdf_ocr_psm,
                ocr_oem=self._pdf_ocr_oem,
                ocr_max_pages=self._pdf_ocr_max_pages,
            ),
            ".xlsx": lambda p: ExcelSmartLoader(
                p,
                mode=self._excel_mode,
                header=self._excel_header,
                sheet=self._excel_sheet,
                na_filter=self._excel_na_filter,
                max_rows_per_sheet=self._excel_max_rows_per_sheet,
            ),
            ".xls":  lambda p: ExcelSmartLoader(
                p,
                mode=self._excel_mode,
                header=self._excel_header,
                sheet=self._excel_sheet,
                na_filter=self._excel_na_filter,
                max_rows_per_sheet=self._excel_max_rows_per_sheet,
            ),
            ".csv":  lambda p: ExcelSmartLoader(
                p,
                mode=self._excel_mode,
                header=self._excel_header,
                na_filter=self._excel_na_filter,
                max_rows_per_sheet=self._excel_max_rows_per_sheet,
                encoding=self._csv_encoding,
                delimiter=self._csv_delimiter or ",",
            ),
            ".tsv":  lambda p: ExcelSmartLoader(
                p,
                mode=self._excel_mode,
                header=self._excel_header,
                na_filter=self._excel_na_filter,
                max_rows_per_sheet=self._excel_max_rows_per_sheet,
                encoding=self._csv_encoding,
                delimiter=self._csv_delimiter or "\t",
            ),
        }

        # images
        image_exts = (
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".bmp",
            ".webp",
            ".heic",  # optional modern iPhone format
            ".heif",  # optional, same family
        )

        for ext in image_exts:
            default_map[ext] = lambda p, _ext=ext: ImageSmartLoader(
                p,
                ocr_lang=self._image_ocr_lang,
                ocr_psm=self._image_ocr_psm,
                ocr_oem=self._image_ocr_oem,
                extract_exif=self._image_extract_exif,
                max_image_dim=self._image_max_dim,                
                text_mode=self._image_text_mode,                # "ocr" | "caption" | "both"
                caption_llm=self._image_caption_llm,
                both_joiner=self._image_both_joiner,
            )

        # videos
        video_exts = (
            ".mp4",
            ".mkv",
            ".mov",
            ".avi",
            ".webm",
            ".m4v",
            ".flv",
            ".wmv",
            ".ts",
            ".3gp",
            ".ogv",
        )

        for ext in video_exts:
            default_map[ext] = lambda p, _ext=ext: VideoSmartLoader(
                p,
                out_dir=None,                # optional, saves frames/metadata next to the video
                frames_subdir="frames",
                meta_subdir="video_meta",
                transcribe_if_missing=True,  # automatically generates VTT if missing
                whisper_model_size="base",
                whisper_language=None,       # e.g. "pl" if you want to force Polish
                frame_target_height=350,
            )


        # Audio
        audio_exts = (
            ".wav",
            ".mp3",
            ".m4a",
            ".flac",
            ".ogg",
            ".opus",
            ".aac",
            ".wma",
            ".aiff",  # aka .aif
            ".aif",
            ".mka",   # Matroska audio
        )

        for ext in audio_exts:
            default_map[ext] = lambda p, _ext=ext: AudioSmartLoader(
                path=p,
                out_dir=None,
                audio_format=_ext.lstrip("."),
                whisper_model="medium",
                whisper_language=None,
                translate=True,
            )


        if self._docx_mode == "fulltext":
            # original behavior: single Document with full text
            default_map[".docx"] = lambda p: Docx2txtLoader(p)
        elif self._docx_mode in ("paragraphs", "headings"):
            # custom loader returning a list of Documents per paragraph / heading
            default_map[".docx"] = lambda p: DocxParagraphLoader(p, mode=self._docx_mode)
        else:
            raise ValueError("docx_mode must be one of: 'fulltext', 'paragraphs', 'headings'")


        self._extensions_map: Dict[str, Callable[[str], object]] = dict(default_map)

        if extensions_map:
            for k, v in extensions_map.items():
                if not callable(v):
                    raise TypeError(f"extensions_map['{k}'] must be callable(path)->Loader")
                self._extensions_map[k.lower()] = v

        self._allowed_exts = set(self._extensions_map.keys())

    def _is_within_limits(self, file_path: Path) -> bool:
        # size guard
        if self._max_file_size_mb is not None:
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > self._max_file_size_mb:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[intergraxDocumentsLoader] Skipping large file (%.1f MB): %s",
                            size_mb,
                            file_path,
                            extra={"data": {"size_mb": float(size_mb)}},
                        )
                    return False
            except OSError as e:
                logger.warning("[intergraxDocumentsLoader] Could not stat file %s: %s", file_path, e)
                return False
        return True

    def _is_excluded(self, file_path: Path, root: Path) -> bool:
        # apply exclude globs relative to root
        if not self._exclude_globs:
            return False
        rel = file_path.relative_to(root)
        for pat in self._exclude_globs:
            if rel.match(pat):
                return True
        return False

    @staticmethod
    def _stable_parent_id(path: Path) -> str:
        # stable id from absolute path
        return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


    def load_document(
        self,
        file_path: str,
        *,
        use_default_metadata: bool = True,
        call_custom_metadata: Optional[Union[MetadataFn, Sequence[MetadataFn]]] = None,
    ) -> List[Document]:
        """
        Load a SPECIFIC file (single-file) and enrich with metadata, like in load_documents().
        Returns a list of Document (e.g., PDF → 1 per page, DOCX paragraphs → many, etc.).
        """
        p = Path(file_path).resolve()
        docs: List[Document] = []

        if not p.exists() or not p.is_file():
            logger.error("[intergraxDocumentsLoader] File not found: %s", p)
            return docs

        # size / exclude / extension
        if not self._is_within_limits(p):
            return docs

        ext = p.suffix.lower()
        if ext not in self._allowed_exts:
            logger.warning("[intergraxDocumentsLoader] Unsupported extension for single-file load: %s", p)
            return docs

        # callbacks → list
        callbacks: List[MetadataFn] = []
        if call_custom_metadata:
            callbacks = (
                list(call_custom_metadata)
                if isinstance(call_custom_metadata, (list, tuple))
                else [call_custom_metadata]
            )

        try:
            loader_factory = self._extensions_map.get(ext)
            if loader_factory is None:
                logger.warning("[intergraxDocumentsLoader] No loader for extension: %s", ext)
                return docs

            loader = loader_factory(str(p))
            loaded = loader.load()  # typically List[Document]
            if not loaded:
                return docs

            parent_id = self._stable_parent_id(p)

            for d in loaded:
                if use_default_metadata:
                    d.metadata.setdefault("source_path", str(p))
                    d.metadata.setdefault("source_name", p.name)
                    d.metadata.setdefault("ext", ext)
                    if "page" in d.metadata and "page_index" not in d.metadata:
                        d.metadata["page_index"] = d.metadata["page"]
                    d.metadata.setdefault("parent_id", parent_id)

                for cb in callbacks:
                    try:
                        extra = cb(d, p)
                        if isinstance(extra, dict):
                            d.metadata.update({k: v for k, v in extra.items() if v is not None})
                    except Exception as cb_e:
                        logger.exception("[intergraxDocumentsLoader] Callback error for %s: %s", p, cb_e)

            docs.extend(loaded)
            return docs

        except Exception as e:
            logger.exception("[intergraxDocumentsLoader] Error while loading file %s: %s", p, e)
            return docs


    def load_documents(
        self,
        directory_path: str,
        *,
        use_default_metadata: bool = True,
        call_custom_metadata: Optional[Union[MetadataFn, Sequence[MetadataFn]]] = None
    ) -> List[Document]:
        """
        Scans a directory according to file_patterns/exclusions/limits and
        delegates each file to load_document(...).
        """

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[intergraxDocumentsLoader] Loading documents from %s", directory_path)

        docs: List[Document] = []
        root = Path(directory_path).resolve()
        if not root.exists():
            logger.error("[intergraxDocumentsLoader] Directory not found: %s", root)
            return docs

        # Gather candidates by patterns
        all_files: List[Path] = []
        for pattern in self._file_patterns:
            for f in root.glob(pattern):
                try:
                    if not self._follow_symlinks and f.is_symlink():
                        continue
                    all_files.append(f)
                except OSError:
                    continue

        # Filtering: file, extension, exclude, size
        candidate_files: List[Path] = []
        for f in all_files:
            try:
                if not f.is_file():
                    continue
                if self._is_excluded(f, root):
                    continue
                if f.suffix.lower() not in self._allowed_exts:
                    continue
                if not self._is_within_limits(f):
                    continue
                candidate_files.append(f)
            except OSError:
                continue

        # File count limit
        if self._max_files is not None and len(candidate_files) > self._max_files:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[intergraxDocumentsLoader] Too many files (%d). Truncating to %d.",
                    len(candidate_files),
                    self._max_files,
                    extra={"data": {"total": len(candidate_files), "limit": self._max_files}},
                )
            candidate_files = candidate_files[: self._max_files]


        # Progress bar + delegation to single-file loader
        with tqdm(
            desc=f"Loading files from {directory_path}",
            unit="file",
            leave=False,
            total=len(candidate_files),
            disable=not logger.isEnabledFor(logging.DEBUG),
        ) as pbar:
            for file in candidate_files:
                try:
                    file_docs = self.load_document(
                        str(file),
                        use_default_metadata=use_default_metadata,
                        call_custom_metadata=call_custom_metadata,
                    )
                    if file_docs:
                        docs.extend(file_docs)
                except Exception as e:
                    logger.exception("[intergraxDocumentsLoader] Error while loading file %s: %s", file, e)
                finally:
                    pbar.update(1)

        logger.debug(
            "[intergraxDocumentsLoader] Done. Loaded documents: %d",
            len(docs),
            extra={"data": {"loaded": len(docs)}},
        )


        return docs

