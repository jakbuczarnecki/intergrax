# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import json
import os
from typing import Literal, Optional
from PIL import Image, ExifTags
from langchain_core.documents import Document

from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter


try:
    import pytesseract
except Exception:
    pytesseract = None

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

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

    # ---------- helpers ----------
    def _resize_if_needed(self, img: Image) -> Image:
        if self.max_image_dim is None:
            return img
        w, h = img.size
        if max(w, h) <= self.max_image_dim:
            return img
        ratio = self.max_image_dim / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        return img.resize(new_size)

    def _ocr(self, img: Image) -> str:
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

    def _infer_ollama_model(self) -> Optional[str]:
        """
        Try to infer model name from the LangChainOllamaAdapter.
        - Prefer adapter.defaults.get("model")
        - Fallbacks are possible (chat.model), else None
        """
        if not isinstance(self.caption_llm, LangChainOllamaAdapter):
            return None
        defaults = getattr(self.caption_llm, "defaults", {}) or {}
        model = defaults.get("model")
        if model:
            return model
        chat = getattr(self.caption_llm, "chat", None)
        if chat is not None:
            for attr in ("model", "model_name", "model_id"):
                if hasattr(chat, attr):
                    try:
                        val = getattr(chat, attr)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                    except Exception:
                        pass
            for attr in ("kwargs", "config", "client"):
                try:
                    obj = getattr(chat, attr, None)
                    if isinstance(obj, dict):
                        for k in ("model", "model_name", "model_id"):
                            if isinstance(obj.get(k), str) and obj[k].strip():
                                return obj[k].strip()
                except Exception:
                    pass
        return None

    def _caption_via_ollama(self, img_path: str) -> str:
        """
        Vision caption bridge for Ollama (local REST).
        Uses model inferred from the adapter. Endpoint defaults to localhost.
        """
        from intergrax.multimedia.images_loader import transcribe_image

        model = self._infer_ollama_model() or "llava-llama3:latest"
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
        - Else if it's LangChainOllamaAdapter → use REST vision bridge.
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
        if isinstance(self.caption_llm, LangChainOllamaAdapter):
            return self._caption_via_ollama(img_path)
        # 3) Not supported yet
        raise ValueError("Captioning supported for adapters exposing describe_image(...) or LangChainOllamaAdapter (vision).")

    def _exif_dict(self, img: Image) -> dict:
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
    def load(self) -> list[Document]:
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
            "caption_model_inferred": (self._infer_ollama_model() or "llava-llama3:latest")
                if (self.text_mode in ("caption", "both") and isinstance(self.caption_llm, LangChainOllamaAdapter))
                else None,
        }

        return [Document(page_content=content, metadata=meta)]