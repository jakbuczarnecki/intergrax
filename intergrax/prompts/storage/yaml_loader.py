# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import yaml

from intergrax.prompts.schema.prompt_schema import (
    LocalizedContent,
    LocalizedPromptDocument,
    PromptMeta,
)
from intergrax.prompts.storage.models import (
    LoadedPrompt,
    PromptParseError,
    PromptValidationError,
)


class YamlPromptLoader:
    """
    Production-grade YAML loader for prompt documents.

    No dynamic typing, no getattr, explicit contract only.
    """

    def load(self, path: Path) -> LoadedPrompt:

        try:
            raw = self._read_yaml(path)
        except Exception as e:
            raise PromptParseError(str(e)) from e

        if "id" not in raw or "version" not in raw:
            raise PromptValidationError("Missing required fields")

        # Validation errors must NOT be wrapped
        doc = self._parse_localized_document(raw)

        try:
            h = self._compute_hash(doc)
        except Exception as e:
            raise PromptParseError(str(e)) from e

        return LoadedPrompt(document=doc, content_hash=h)


    # ---------------------------------------------------------------------

    def _read_yaml(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise PromptValidationError("YAML root must be a mapping")

        return data

        
    def _parse_localized_document(
        self, data: Dict[str, Any]
    ) -> LocalizedPromptDocument:

        # --- basic required fields --------------------------------------------

        if "id" not in data:
            raise PromptValidationError("Missing required field: 'id'")

        if "version" not in data:
            raise PromptValidationError("Missing required field: 'version'")

        pid = str(data["id"])
        version = int(data["version"])

        # --- 1) NEW FORMAT: explicit locales ----------------------------------

        locales_raw = data.get("locales")

        if isinstance(locales_raw, dict):
            locales: Dict[str, LocalizedContent] = {}

            for lang, wrapper in locales_raw.items():
                if not isinstance(wrapper, dict):
                    raise PromptValidationError(
                        f"Invalid locale '{lang}' structure"
                    )

                content = wrapper.get("content")
                if not isinstance(content, dict):
                    raise PromptValidationError(
                        f"Locale '{lang}' must contain 'content' mapping"
                    )

                if "system" not in content:
                    raise PromptValidationError(
                        f"Locale '{lang}' content must contain 'system'"
                    )

                locales[str(lang)] = LocalizedContent(
                    system=str(content["system"]),
                    developer=self._opt_str(content.get("developer")),
                    user_template=self._opt_str(content.get("user_template")),
                )


        # --- 2) LEGACY FORMAT: top-level 'content' ---------------------------

        elif "content" in data:
            c = data["content"]

            if not isinstance(c, dict):
                raise PromptValidationError(
                    "Field 'content' must be mapping"
                )

            locales = {
                "en": LocalizedContent(
                    system=str(c["system"]),
                    developer=self._opt_str(c.get("developer")),
                    user_template=self._opt_str(c.get("user_template")),
                )
            }

        else:
            raise PromptValidationError(
                "Prompt must contain either 'locales' or legacy 'content'"
            )

        # --- meta -------------------------------------------------------------

        if "meta" not in data:
            raise PromptValidationError("Missing required field: 'meta'")

        meta = data["meta"]

        return LocalizedPromptDocument(
            id=pid,
            version=version,
            locales=locales,
            meta=PromptMeta(
                model_family=str(meta["model_family"]),
                output_schema_id=str(meta["output_schema_id"]),
                tags=frozenset(str(t) for t in meta.get("tags", [])),
                description=self._opt_str(meta.get("description")),
            ),
        )




    # ---------------------------------------------------------------------

    def _compute_hash(self, doc: LocalizedPromptDocument) -> str:
        h = hashlib.sha256()

        h.update(doc.id.encode("utf-8"))
        h.update(str(doc.version).encode("utf-8"))

        for lang in sorted(doc.locales.keys()):
            lc = doc.locales[lang]
            h.update(lang.encode("utf-8"))
            h.update(lc.system.encode("utf-8"))

            if lc.developer:
                h.update(lc.developer.encode("utf-8"))

            if lc.user_template:
                h.update(lc.user_template.encode("utf-8"))

        h.update(doc.meta.model_family.encode("utf-8"))
        h.update(doc.meta.output_schema_id.encode("utf-8"))

        for tag in sorted(doc.meta.tags):
            h.update(tag.encode("utf-8"))

        return h.hexdigest()


    # ---------------------------------------------------------------------

    def _opt_str(self, v: Any) -> str | None:
        if v is None:
            return None
        return str(v)
