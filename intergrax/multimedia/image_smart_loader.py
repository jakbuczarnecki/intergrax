# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access
import json
import hashlib
import os
from pathlib import Path
from typing import Any, Literal, Optional

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


class _PillowModuleProxy:
    def open(self, *args: Any, **kwargs: Any):
        return _import_pillow()[0].open(*args, **kwargs)


class _PillowExifTagsProxy:
    def __getattr__(self, name: str) -> Any:
        return attribute_access.optional(_import_pillow()[1], name)


def _import_pillow():
    try:
        from PIL import ExifTags, Image as PillowImage
    except ModuleNotFoundError as exc:
        if exc.name == "PIL":
            raise IntegrationDependencyError(
                "Image parsing requires optional dependency 'pillow'. "
                "Install Intergrax-ai[media-image].",
                integration_name="image",
            ) from exc
        raise
    return PillowImage, ExifTags


Image = _PillowModuleProxy()
ExifTags = _PillowExifTagsProxy()
pytesseract = None

class ImageSmartLoader:
    """
    Universal image loader: JPG, PNG, TIFF, BMP, WEBP, HEIC/HEIF.
    Modes:
      - OCR:     extract visible text with Tesseract
      - Caption: call your framework LLM adapter (e.g., Ollama) to describe the image
      - Both:    combine caption + OCR (with a joiner)
    Always returns 1 Document per image with clear provenance metadata.
    """

    def __init__(
        self,
        path: str,
        *,
        ocr_lang: str = "eng",
        ocr_psm: int | None = None,
        ocr_oem: int | None = None,
        extract_exif: bool = True,
        max_image_dim: int | None = None,  # e.g., 2000 – downscale if larger
        # NEW:
        text_mode: Literal["ocr", "caption", "both"] = "both",
        caption_llm: Optional[LLMAdapter] = None,
        both_joiner: str = "\n\n---\n\n",
        tenant_id: str = "default",
        namespace: str | None = None,
        workspace_id: str | None = None,
    ):
        self.path = path
        self.ocr_lang = ocr_lang
        self.ocr_psm = ocr_psm
        self.ocr_oem = ocr_oem
        self.extract_exif = bool(extract_exif)
        self.max_image_dim = max_image_dim

        self.text_mode = text_mode
        self.caption_llm = caption_llm
        self.both_joiner = both_joiner
        self.tenant_id = tenant_id
        self.namespace = namespace
        self.workspace_id = workspace_id

    # ---------- helpers ----------
    def _resize_if_needed(self, img: Any) -> Any:
        if self.max_image_dim is None:
            return img
        w, h = img.size
        if max(w, h) <= self.max_image_dim:
            return img
        ratio = self.max_image_dim / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        return img.resize(new_size)

    def _ocr(self, img: Any) -> str:
        if pytesseract is None:
            return ""
        cfg_parts = []
        if self.ocr_psm is not None:
            cfg_parts.append(f"--psm {int(self.ocr_psm)}")
        if self.ocr_oem is not None:
            cfg_parts.append(f"--oem {int(self.ocr_oem)}")
        config = " ".join(cfg_parts) if cfg_parts else None
        try:
            return pytesseract.image_to_string(img, lang=self.ocr_lang, config=config) or ""
        except Exception:
            return ""

    def _resolved_ollama_model(self) -> Optional[str]:
        """
        Resolve the model name for an Ollama adapter.
        - Prefer the adapter's stable public model attribute
        - Then prefer adapter.defaults.get("model")
        - Fallbacks are possible (chat.model)
        - Use the vision bridge default when no model is exposed
        - Return None for non-Ollama adapters
        """
        if not self._is_ollama_adapter(self.caption_llm):
            return None
        model = attribute_access.optional(self.caption_llm, "model", None)
        if isinstance(model, str) and model.strip():
            return model.strip()
        defaults = attribute_access.optional(self.caption_llm, "defaults", {}) or {}
        model = defaults.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        chat = attribute_access.optional(self.caption_llm, "chat", None)
        if chat is not None:
            for attr in ("model", "model_name", "model_id"):
                if hasattr(chat, attr):
                    try:
                        val = attribute_access.optional(chat, attr)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                    except Exception:
                        pass
            for attr in ("kwargs", "config", "client"):
                try:
                    obj = attribute_access.optional(chat, attr, None)
                    if isinstance(obj, dict):
                        for k in ("model", "model_name", "model_id"):
                            if isinstance(obj.get(k), str) and obj[k].strip():
                                return obj[k].strip()
                except Exception:
                    pass
        return "llava-llama3:latest"

    @staticmethod
    def _is_ollama_adapter(adapter: object | None) -> bool:
        provider = attribute_access.optional(adapter, "provider", None)
        return provider in (LLMProvider.OLLAMA, LLMProvider.OLLAMA.value)

    def _caption_via_ollama(self, img_path: str) -> str:
        """
        Vision caption bridge for Ollama (local REST).
        Uses model inferred from the adapter. Endpoint defaults to localhost.
        """
        from intergrax.multimedia.images_loader import transcribe_image

        model = self._resolved_ollama_model()
        prompt = "Describe the image in detail."


        resp = transcribe_image(
            prompt=prompt, 
            model=model,
            image_path=img_path,
        )

        return resp

    def _caption_via_adapter(self, img_path: str) -> str:
        """
        Generic bridge:
        - If adapter exposes describe_image(path) → use it.
        - Else if it's an Ollama adapter → use REST vision bridge.
        - Else raise (you can extend here for OpenAI/Gemini Vision).
        """
        if self.caption_llm is None:
            return ""

        # 1) Native helper, if adapter ją posiada
        if hasattr(self.caption_llm, "describe_image"):
            try:
                txt = self.caption_llm.describe_image(img_path)
                return (txt or "").strip()
            except Exception as e:
                raise RuntimeError(f"LLMAdapter.describe_image failed: {e}")
        # 2) Ollama vision fallback
        if self._is_ollama_adapter(self.caption_llm):
            return self._caption_via_ollama(img_path)
        # 3) Not supported yet
        raise ValueError("Captioning supported for adapters exposing describe_image(...) or an Ollama adapter (vision).")

    def _exif_dict(self, img: Any) -> dict:
        out = {}
        if not (self.extract_exif and ExifTags and hasattr(img, "_getexif")):
            return out
        try:
            exif_raw = img._getexif() or {}
            for tag, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag, str(tag))
                out[tag_name] = str(value)
        except Exception:
            pass
        return out

    # ---------- main ----------
    def load(self) -> list[KnowledgeDocument]:
        if Image is None:
            raise ImportError("Pillow (PIL) is required for ImageSmartLoader")

        img = Image.open(self.path)
        img = self._resize_if_needed(img)
        width, height = img.size
        dpi = img.info.get("dpi", None)
        exif = self._exif_dict(img)

        # Decide which mechanisms to run
        run_ocr = (self.text_mode in ("ocr", "both")) and (pytesseract is not None)
        run_caption = (self.text_mode in ("caption", "both")) and (self.caption_llm is not None)

        ocr_text = self._ocr(img) if run_ocr else ""
        caption_text = self._caption_via_adapter(self.path) if run_caption else ""

        # Assemble content
        if self.text_mode == "ocr":
            content = (ocr_text or "").strip() or "(No visible text detected.)"
        elif self.text_mode == "caption":
            content = (caption_text or "").strip() or "(No caption generated.)"
        else:
            left = (caption_text or "").strip()
            right = (ocr_text or "").strip()
            if left and right:
                content = f"{left}{self.both_joiner}{right}"
            elif left:
                content = left
            elif right:
                content = right
            else:
                content = "(No caption nor OCR text produced.)"

        caption_model = (
            self._resolved_ollama_model()
            if (
                self.text_mode in ("caption", "both")
                and self.caption_llm is not None
            )
            else None
        )

        # Metadata
        meta = {
            "source_name": os.path.basename(self.path),
            "source_path": self.path,
            "format": img.format,
            "width": width,
            "height": height,
            "dpi": (dpi[0] if isinstance(dpi, tuple) and len(dpi) > 0 else dpi),
            "exif_json": json.dumps(exif, ensure_ascii=False) if exif else None,

            # Provenance & modes
            "image_text_mode": self.text_mode,                 # "ocr" | "caption" | "both"
            "ocr_lang": self.ocr_lang if (self.text_mode in ("ocr", "both")) else None,
            "caption_llm": type(self.caption_llm).__name__ if (self.text_mode in ("caption", "both") and self.caption_llm) else None,
            "caption_model_inferred": caption_model,
        }

        source_id = str(Path(self.path).resolve())
        document_id = "image:" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()
        document = KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {
                    "document_id": document_id,
                    "root_document_id": document_id,
                },
                "scope": {
                    "tenant_id": self.tenant_id,
                    "namespace": self.namespace,
                    "workspace_id": self.workspace_id,
                },
                "content": content,
                "metadata": meta,
                "provenance": {
                    "source_kind": "image",
                    "source_id": source_id,
                },
            }
        )
        return [document]